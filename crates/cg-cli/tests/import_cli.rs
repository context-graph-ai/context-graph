mod support;

use context_graph::{DecisionId, DecisionStatus, DiscoveryFilter, IntentionId, Store, StoreConfig};
use contextdb_core::Value;
use contextdb_engine::Database;
use serde_json::Value as JsonValue;
use support::{CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider, write_file};

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
{"type":"user","message":{"content":[{"type":"text","text":"For Acme, we need password-less login to ship this sprint, and we need an audit trail for every auth migration."}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"grep","input":{"query":"Decision: raw tool-use decoy should not persist"}},{"type":"text","text":"I found the current auth module at src/auth.py."}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","content":"Observation: raw tool-result decoy should not persist"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Let's use JWT with 15-min access tokens for the login flow."}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Write the auth migration notes to docs/auth-migration.md so reviewers can audit the change."}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Current facts: the auth module file is observed at src/auth.py, auth tests cover password-less login, JWT refresh token TTL remains 7 days, and the audit trail lives in docs/auth-migration.md."}]}}
"#;

const EXPECTED_CLEAN_CLAUDE_SESSION: &str = "User: For Acme, we need password-less login to ship this sprint, and we need an audit trail for every auth migration.\nAssistant: I found the current auth module at src/auth.py.\nAssistant: Let's use JWT with 15-min access tokens for the login flow.\nAssistant: Write the auth migration notes to docs/auth-migration.md so reviewers can audit the change.\nAssistant: Current facts: the auth module file is observed at src/auth.py, auth tests cover password-less login, JWT refresh token TTL remains 7 days, and the audit trail lives in docs/auth-migration.md.\n";

const SECOND_CLAUDE_SESSION: &str = r#"{"type":"user","message":{"content":[{"type":"text","text":"Checkout is slow; the goal is to reduce checkout latency before the next launch."}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"We should cache the product catalog in memory for the checkout service."}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"After the change, checkout latency p95 dropped to 220ms."}]}}
"#;

const EXPECTED_CLEAN_SECOND_CLAUDE_SESSION: &str = "User: Checkout is slow; the goal is to reduce checkout latency before the next launch.\nAssistant: We should cache the product catalog in memory for the checkout service.\nAssistant: After the change, checkout latency p95 dropped to 220ms.\n";

const AUTH_REPLACEMENT_CLAUDE_SESSION: &str = r#"{"type":"user","message":{"content":[{"type":"text","text":"We are migrating the auth handler this week."}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Replace the auth handler with src/authentication.py."}]}}
"#;

fn compact_expected_session(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn large_chunked_statement(index: usize) -> String {
    let filler = format!(" background-{index}").repeat(500);
    format!(
        "Goal G{index}: keep service-{index} reliable. \
Entity E{index}: service-{index} path=services/{index}. \
Decision D{index}: route service-{index} through queue serves G{index} based_on E{index}. \
Observation O{index}: E{index} queue verified. {filler}"
    )
}

fn large_chunked_session(lines: usize) -> String {
    let mut session = String::new();
    for index in 0..lines {
        let text = large_chunked_statement(index);
        let encoded = serde_json::to_string(&text).expect("json string");
        session.push_str(&format!(
            r#"{{"type":"user","message":{{"content":[{{"type":"text","text":{encoded}}}]}}}}"#
        ));
        session.push('\n');
    }
    session
}

fn large_chunked_clean_session(lines: usize) -> String {
    (0..lines)
        .map(|index| format!("User: {}\n", large_chunked_statement(index)))
        .collect()
}

fn plain_sentinel_session(lines: usize) -> (String, String) {
    let mut session = String::new();
    let mut clean = String::new();
    for index in 0..lines {
        let marker = match index {
            0 => "SENTINEL_OPEN",
            value if value + 1 == lines => "SENTINEL_CLOSE",
            _ => "middle",
        };
        let text = format!(
            "The operator discussed {marker} while evaluating long-session memory chunk {index}. {}",
            "ordinary prose without ontology labels ".repeat(30)
        );
        let encoded = serde_json::to_string(&text).expect("json string");
        session.push_str(&format!(
            r#"{{"type":"user","message":{{"content":[{{"type":"text","text":{encoded}}}]}}}}"#
        ));
        session.push('\n');
        clean.push_str("User: ");
        clean.push_str(&text);
        clean.push('\n');
    }
    (session, clean)
}

fn extraction_prompt_transcript(prompt: &str) -> Option<String> {
    if !prompt.contains("Extract context-graph memory") {
        return None;
    }
    let transcript = prompt
        .split_once("Transcript:\n")
        .map(|(_, tail)| tail)
        .unwrap_or(prompt);
    let start = "<cg_transcript>\n";
    let end = "\n</cg_transcript>";
    let transcript = transcript
        .split_once(start)
        .map(|(_, tail)| tail)
        .unwrap_or(transcript)
        .split_once(end)
        .map(|(head, _)| head)
        .unwrap_or(transcript);
    Some(transcript.to_string())
}

fn source_chunk_from_prompt(prompt: &str) -> Option<String> {
    let transcript = extraction_prompt_transcript(prompt)?;
    let source = transcript
        .rsplit_once("Next session chunk:\n")
        .map(|(_, chunk)| chunk)
        .unwrap_or(transcript.as_str())
        .trim()
        .to_string();
    (!source.is_empty()).then_some(source)
}

fn source_chunks_after(harness: &CliHarness, before_calls: usize) -> Vec<String> {
    harness
        .providers
        .completion_records()
        .into_iter()
        .skip(before_calls)
        .filter_map(|record| {
            record
                .get("prompt")
                .and_then(JsonValue::as_str)
                .and_then(source_chunk_from_prompt)
        })
        .collect()
}

#[test]
fn cli_import_claude_session_becomes_memory() {
    let harness = extraction_harness();
    let session = harness.data_root.path().join("session.jsonl");
    write_file(&session, CLAUDE_SESSION);

    let before = harness.run(&["status"]);
    let dry_run = harness.run(&[
        "import",
        "--understand-now",
        "--dry-run",
        session.to_str().expect("utf8 path"),
    ]);
    let after_dry_run = harness.run(&["status"]);
    let import = harness.run(&[
        "import",
        "--understand-now",
        session.to_str().expect("utf8 path"),
    ]);
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
    drop(store);
    let second_session = harness.data_root.path().join("session-two.jsonl");
    write_file(&second_session, SECOND_CLAUDE_SESSION);
    let second_import = harness.run(&[
        "import",
        "--understand-now",
        second_session.to_str().expect("utf8 path"),
    ]);
    let second_status = harness.run(&["status"]);
    let second_query = harness.run(&["query", "checkout latency"]);
    let second_goal_id = stable_uuid_from(
        &[&second_import, &second_query],
        &["reduce checkout latency", "G_checkout_latency"],
    )
    .map(IntentionId::from);
    let reopened = persisted_store(store_path.clone());
    let second_persisted = persisted_decisions_for_intention(reopened.as_ref(), second_goal_id);
    drop(reopened);
    let store_path_ref = store_path.as_deref();
    let mut expected_digest_summaries = vec![
        compact_expected_session(EXPECTED_CLEAN_CLAUDE_SESSION),
        compact_expected_session(EXPECTED_CLEAN_SECOND_CLAUDE_SESSION),
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
            extracted_from_edges: 14,
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
    let mut persisted_decisions = persisted
        .iter()
        .map(|(_, description, status)| (description.as_str(), *status))
        .collect::<Vec<_>>();
    persisted_decisions.sort_by(|left, right| left.0.cmp(right.0));
    let mut second_persisted_decisions = second_persisted
        .iter()
        .map(|(_, description, status)| (description.as_str(), *status))
        .collect::<Vec<_>>();
    second_persisted_decisions.sort_by(|left, right| left.0.cmp(right.0));
    let mut audit_persisted_decisions = audit_persisted
        .iter()
        .map(|(_, description, status)| (description.as_str(), *status))
        .collect::<Vec<_>>();
    audit_persisted_decisions.sort_by(|left, right| left.0.cmp(right.0));

    assert_eq!(
        (
            persisted_decisions,
            second_persisted_decisions,
            audit_persisted_decisions,
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
fn cli_import_streams_large_jsonl_in_bounded_chunks() {
    let harness = extraction_harness();
    let session = harness.data_root.path().join("large-session.jsonl");
    let line_count = 18;
    write_file(&session, &large_chunked_session(line_count));

    let before_calls = harness.providers.completion_count();
    let dry_run = harness.run(&[
        "import",
        "--understand-now",
        "--dry-run",
        "--chunk-tokens",
        "10000",
        session.to_str().expect("utf8 path"),
    ]);
    let import = harness.run(&[
        "import",
        "--understand-now",
        "--chunk-tokens",
        "10000",
        session.to_str().expect("utf8 path"),
    ]);
    let completion_calls = harness.providers.completion_count() - before_calls;
    let import_json = output_json(&import).expect("import json");
    let dry_run_json = output_json(&dry_run).expect("dry run json");
    let persisted = persisted_counts(store_path_from(&[&import]).as_deref());
    let source_chunks = source_chunks_after(&harness, before_calls);

    assert_eq!(dry_run.code, 0, "dry run failed: {}", combined(&dry_run));
    assert_eq!(import.code, 0, "import failed: {}", combined(&import));
    assert_eq!(
        (
            dry_run_json
                .get("resolved_chunk_window")
                .and_then(JsonValue::as_u64),
            import_json
                .get("resolved_chunk_window")
                .and_then(JsonValue::as_u64),
        ),
        (Some(10_000), Some(10_000))
    );
    assert!(
        dry_run_json
            .get("chunks")
            .and_then(JsonValue::as_u64)
            .is_some_and(|chunks| chunks > 1),
        "dry-run must stream the large JSONL into multiple bounded chunks: {}",
        dry_run.stdout
    );
    assert_eq!(
        (
            completion_calls,
            source_chunks.len(),
            source_chunks
                .iter()
                .all(|chunk| chunk.len() <= 10_000 && !chunk.is_empty()),
            compact_expected_session(&source_chunks.join("\n")),
        ),
        (
            source_chunks.len(),
            source_chunks.len(),
            true,
            compact_expected_session(&large_chunked_clean_session(line_count)),
        ),
        "large import must call the extractor once per bounded source chunk and reconstruct the imported transcript"
    );
    assert!(
        import_json
            .get("message")
            .and_then(JsonValue::as_str)
            .is_some_and(|message| message.starts_with("captured import in ")
                && message.contains("one condensed memory")),
        "chunked import should report one condensed memory: {}",
        import.stdout
    );
    assert!(
        matches!(memory_counts(&import), Some((goals, decisions, observations)) if goals >= 18 && decisions >= 18 && observations >= 18),
        "chunked import must aggregate extracted counts: {}",
        import.stdout
    );
    assert!(
        persisted.digests == 1
            && persisted.intentions >= 18
            && persisted.decisions >= 18
            && persisted.observations >= 18,
        "chunked import must persist one session memory with facts from multiple chunks: {persisted:?}"
    );
}

#[test]
fn cli_import_default_window_is_capped_for_real_provider_output_budget() {
    let harness = extraction_harness();
    let session = harness
        .data_root
        .path()
        .join("default-window-session.jsonl");
    write_file(&session, &large_chunked_session(18));

    let before_calls = harness.providers.completion_count();
    let dry_run = harness.run(&[
        "import",
        "--understand-now",
        "--dry-run",
        session.to_str().expect("utf8 path"),
    ]);
    let estimate = harness.run(&[
        "import",
        "--understand-now",
        "--estimate",
        session.to_str().expect("utf8 path"),
    ]);
    let dry_run_json = output_json(&dry_run).expect("dry-run json");
    let estimate_json = output_json(&estimate).expect("estimate json");

    assert_eq!(dry_run.code, 0, "dry run failed: {}", combined(&dry_run));
    assert_eq!(estimate.code, 0, "estimate failed: {}", combined(&estimate));
    assert_eq!(
        harness.providers.completion_count(),
        before_calls,
        "dry-run and estimate must not call the provider"
    );
    assert_eq!(
        (
            dry_run_json
                .get("resolved_chunk_window")
                .and_then(JsonValue::as_u64),
            estimate_json
                .pointer("/estimate/resolved_chunk_window")
                .and_then(JsonValue::as_u64),
        ),
        (Some(16_000), Some(16_000)),
        "the default import window must stay below the hosted-model output budget; \
         explicit --chunk-tokens can still opt into larger windows"
    );
    assert!(
        dry_run_json
            .get("chunks")
            .and_then(JsonValue::as_u64)
            .is_some_and(|chunks| chunks > 1),
        "default import should split a large JSONL into multiple bounded chunks: {}",
        dry_run.stdout
    );
}

#[test]
fn cli_import_rolls_plain_long_session_into_one_digest_without_label_markers() {
    let harness = extraction_harness();
    let session = harness.data_root.path().join("plain-long-session.jsonl");
    let (jsonl, _clean) = plain_sentinel_session(16);
    write_file(&session, &jsonl);

    let before_calls = harness.providers.completion_count();
    let import = harness.run(&[
        "import",
        "--understand-now",
        "--chunk-tokens=2500",
        session.to_str().expect("utf8 path"),
    ]);
    let import_json = output_json(&import).expect("import json");
    let source_chunks = source_chunks_after(&harness, before_calls);
    let store_path = store_path_from(&[&import]);
    let digest_summaries = persisted_text_values(store_path.as_deref(), "digests", "summary");

    assert_eq!(import.code, 0, "import failed: {}", combined(&import));
    assert_eq!(
        (
            import_json
                .get("import_chunks")
                .and_then(JsonValue::as_u64)
                .is_some_and(|chunks| chunks > 1),
            import_json
                .get("resolved_chunk_window")
                .and_then(JsonValue::as_u64),
            source_chunks
                .iter()
                .all(|chunk| chunk.len() <= 2500 && !chunk.is_empty()),
            persisted_counts(store_path.as_deref()).digests,
            digest_summaries.len(),
            digest_summaries
                .first()
                .is_some_and(|summary| summary.contains("SENTINEL_OPEN")
                    && summary.contains("SENTINEL_CLOSE")),
        ),
        (true, Some(2500), true, 1, 1, true,)
    );
}

#[test]
fn cli_import_rejects_malformed_chunk_options_loudly() {
    let harness = extraction_harness();
    let session = harness.data_root.path().join("session.jsonl");
    write_file(&session, SECOND_CLAUDE_SESSION);

    let bad_tokens = harness.run(&[
        "import",
        "--understand-now",
        "--chunk-tokens",
        "abc",
        session.to_str().expect("utf8 path"),
    ]);
    let bad_fraction = harness.run(&[
        "import",
        "--understand-now",
        "--context-fraction=1.5",
        session.to_str().expect("utf8 path"),
    ]);

    assert_eq!(
        (
            bad_tokens.code,
            output_json(&bad_tokens).and_then(|json| json
                .get("condition")
                .and_then(JsonValue::as_str)
                .map(str::to_string)),
            bad_fraction.code,
            output_json(&bad_fraction).and_then(|json| json
                .get("condition")
                .and_then(JsonValue::as_str)
                .map(str::to_string)),
        ),
        (
            2,
            Some("InvalidImportChunking".to_string()),
            2,
            Some("InvalidImportChunking".to_string()),
        )
    );
}

#[test]
fn cli_import_without_model_is_verbatim_no_derived_graph() {
    let harness = CliHarness::new();
    let session = harness.data_root.path().join("session.jsonl");
    write_file(&session, CLAUDE_SESSION);

    let import = harness.run(&[
        "import",
        "--understand-now",
        session.to_str().expect("utf8 path"),
    ]);
    let status = harness.run(&["status"]);
    let store_path = store_path_from(&[&import, &status]);
    let store_path_ref = store_path.as_deref();

    assert_eq!(
        (
            import.code,
            memory_counts(&status),
            persisted_counts(store_path_ref),
            persisted_text_values(store_path_ref, "digests", "summary"),
            persisted_text_values(store_path_ref, "intentions", "description"),
            persisted_text_values(store_path_ref, "decisions", "description"),
        ),
        (
            0,
            Some((0, 0, 0)),
            PersistedCounts {
                contexts: 1,
                digests: 1,
                intentions: 0,
                decisions: 0,
                observations: 0,
                extracted_from_edges: 0,
            },
            vec![EXPECTED_CLEAN_CLAUDE_SESSION.trim_end().to_string()],
            Vec::<String>::new(),
            Vec::<String>::new(),
        )
    );
    let text = combined(&import);
    assert!(text.contains("model-free capture"));
    assert!(text.contains("no derived graph"));
}

#[test]
fn cli_import_auth_replacement_uses_extractor_not_record_shorthand() {
    let harness = extraction_harness();
    let session = harness.data_root.path().join("auth-replacement.jsonl");
    write_file(&session, AUTH_REPLACEMENT_CLAUDE_SESSION);

    let import = harness.run(&[
        "import",
        "--understand-now",
        session.to_str().expect("utf8 path"),
    ]);
    let status = harness.run(&["status"]);
    let store_path = store_path_from(&[&import, &status]);
    let store_path_ref = store_path.as_deref();

    assert_eq!(
        (
            import.code,
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

#[test]
fn cli_import_rejects_bulk_single_file_only() {
    let harness = CliHarness::new();
    let bulk_dir = tempfile::tempdir().expect("bulk dir");
    let before = harness.run(&["status"]);
    let output = harness.run(&[
        "import",
        "--understand-now",
        bulk_dir.path().to_str().expect("utf8 dir"),
    ]);
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

    let import_harness = extraction_harness();
    let import_session = import_harness.data_root.path().join("session.jsonl");
    write_file(&import_session, CLAUDE_SESSION);
    let import = import_harness.run(&[
        "import",
        "--understand-now",
        import_session.to_str().expect("utf8 path"),
    ]);
    let import_status = import_harness.run(&["status"]);
    let import_store_path = store_path_from(&[&import, &import_status]);
    let import_store_path_ref = import_store_path.as_deref();
    let import_digest_summaries =
        persisted_text_values(import_store_path_ref, "digests", "summary");

    let before_record = harness.run(&["status"]);
    let record = harness.run(&[
        "record",
        "--understand-now",
        session.to_str().expect("utf8 path"),
    ]);
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
                extracted_from_edges: 10,
            },
            vec![compact_expected_session(EXPECTED_CLEAN_CLAUDE_SESSION)],
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
    let output = harness.run(&["import", "--understand-now", "./does-not-exist.jsonl"]);
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
    let output = harness.run(&[
        "import",
        "--understand-now",
        path.to_str().expect("utf8 path"),
    ]);
    let after = harness.run(&["status"]);

    assert_ne!(output.code, 0);
    assert!(combined(&output).contains("UnsupportedImportFormat"));
    assert_eq!(combined(&before), combined(&after));
}
