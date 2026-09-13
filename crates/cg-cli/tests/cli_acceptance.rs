mod support;

use std::fs;
use std::process::Command;

use cg_catalog::{
    CacheStatus, Capability, Catalog, CatalogEntry, License, ModelConnectionCatalog, ModelKind,
    PrincipalAwareConfig, Source,
};
use cg_cli::{
    Classification, ProviderOverride, classify_provider_reply, construct_claude_invocation,
    extract_reply, load_provider_specs, register_model_connection_checks_with_root,
    try_load_provider_specs, validate_embedded_provider_specs,
};
use cg_cli_conventions::{CheckRegistry, run_doctor_checks, run_selftest_checks};
use context_graph::{ContextId, Principal};
use serde_json::{Value, json};
use tempfile::TempDir;

use support::{
    CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider, base64_basic,
    catalog_disk_contains, cg_bin, file_inventory, parse_single_json, parse_single_json_object,
    repo_root, run_cg, run_variadic_fake, unique_label, write_file,
};

const INGESTION_EXPECTATION: &str = "automatic capture is ready for cg record and cg import";
const CLAUDE_SUCCESS: &str = include_str!("fixtures/provider_envelopes/claude_success.json");
const CLAUDE_401: &str = include_str!("fixtures/provider_envelopes/claude_401.json");
const CLAUDE_402: &str = include_str!("fixtures/provider_envelopes/claude_402.json");
const CLAUDE_429: &str = include_str!("fixtures/provider_envelopes/claude_429.json");
const CODEX_LAST_MESSAGE: &str = include_str!("fixtures/provider_envelopes/codex_last_message.txt");
const AGY_PLAIN: &str = include_str!("fixtures/provider_envelopes/agy_plain.txt");
const PROBELESS_HOST_ACCOUNT: &str = "account not exposed by host CLI";
const SELFTEST_ECHO_PROMPT: &str =
    "Reply with exactly this text and nothing else: cg-host-session-proof";

fn developer() -> Principal {
    Principal::Human("developer@example.test".to_string())
}

fn seed_connection_model(root: &std::path::Path, connection_name: &str, model_id: &str) {
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(principal.clone(), "FIXTURE_MODEL_KEY", "fixture-secret");
    let mut catalog = ModelConnectionCatalog::open(root);
    catalog.connect_env_var(
        connection_name,
        "FIXTURE_MODEL_KEY",
        principal.clone(),
        &mut config,
    );
    catalog.record_consent(connection_name, "fixture account");
    catalog.register_model(model_id, ModelKind::LlmClient, connection_name);
    catalog.bind_context_model(ContextId::new_v7(), model_id);
}

fn license_entry(model_id: &str, instructions: &str, manual_url: &str) -> CatalogEntry {
    CatalogEntry {
        id: model_id.to_string(),
        kind: ModelKind::LlmClient,
        capability: Capability {
            dimension: None,
            output_shape: Some("understanding".to_string()),
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "LicenseRef-Manual".to_string(),
            url: manual_url.to_string(),
            requires_manual_acceptance: true,
        },
        source: Source::LocalPath {
            dir: std::path::PathBuf::from("/manual/install/path"),
        },
        byte_size: 0,
        cache_status: CacheStatus::NotCached,
        local_path: None,
        install_instructions: Some(instructions.to_string()),
        audit_footnote: None,
        acceptance_recorded: false,
        manual_install_url: Some(manual_url.to_string()),
        artifact_filenames: Vec::new(),
        per_file_sha256: Vec::new(),
    }
}

fn assert_condition(output: &CommandOutput, expected: &str) {
    let parsed = parse_single_json_object(&output.stdout);
    assert!(
        parsed.is_ok(),
        "stdout must be one JSON object carrying condition `{expected}`, got stdout={:?}, stderr={:?}",
        output.stdout,
        output.stderr
    );
    let object = parsed.unwrap_or_default();
    assert_eq!(
        object.get("condition").and_then(Value::as_str),
        Some(expected),
        "typed failure envelope must carry the expected condition"
    );
}

fn completion_count(output_harness: &CliHarness) -> usize {
    output_harness.providers.completion_count()
}

fn write_agy_provider_override(root: &std::path::Path, required_help_flags: &[&str]) {
    let required_help_flags = required_help_flags
        .iter()
        .map(|flag| format!("\"{flag}\""))
        .collect::<Vec<_>>()
        .join(", ");
    write_file(
        root.join("cg/providers.toml"),
        &format!(
            r#"[providers.agy]
binary = "agy"
prompt_delivery = "argv"
completion_args = ["--print"]
env_remove = []
env_preserve = []
reply_framing = "text"
reply_path = "stdout"
timeout_ms = 360000
consent_required = true
required_help_flags = [{required_help_flags}]
last_verified = "test override: agy pure-completion drive not embedded"
"#
        ),
    );
}

fn assert_host_session_catalog_row(
    root: &std::path::Path,
    connection_name: &str,
    provider: &str,
    account: &str,
    consent_recorded: bool,
) {
    let catalog = ModelConnectionCatalog::open(root);
    let connection = catalog.read_connection(connection_name);
    assert!(
        connection.is_some(),
        "shared catalog must contain connection `{connection_name}`"
    );
    let connection = connection.unwrap();
    assert_eq!(connection.name, connection_name);
    assert_eq!(connection.exposed_secret, None);
    assert_eq!(
        connection.consent_recorded, consent_recorded,
        "shared catalog consent flag must match the connect path"
    );
    if consent_recorded {
        assert_eq!(
            connection.spend_owner, account,
            "shared catalog spend owner must equal the probed account label"
        );
    }
    assert_eq!(
        connection.credential_reference.env_var, "",
        "host-session references must not be encoded as an env-var key"
    );
    let source = serde_json::to_value(&connection.credential_reference.source)
        .expect("serialize credential source")
        .to_string();
    let source_lower = source.to_ascii_lowercase();
    assert!(
        source_lower.contains("host") && source_lower.contains(provider),
        "shared catalog credential source must be the selected host-session provider, got {source}"
    );
    assert!(
        !source.contains("EnvVar") && !source.contains("StoredSecret"),
        "host-session connection must not masquerade as legacy credential source: {source}"
    );
}

fn json_string_field(
    object: &serde_json::Map<String, Value>,
    field: &str,
    context: &str,
) -> String {
    let value = object.get(field).and_then(Value::as_str);
    assert!(
        value.is_some(),
        "{context} must carry string field `{field}`, got {object:?}"
    );
    value.unwrap_or_default().to_string()
}

fn assert_bucket(output: &CommandOutput, expected: &str) {
    let parsed = parse_single_json_object(&output.stdout);
    assert!(
        parsed.is_ok(),
        "selftest failure must emit exactly one JSON object, got {:?}",
        output.stdout
    );
    let object = parsed.unwrap_or_default();
    let bucket = object.get("bucket").and_then(Value::as_str);
    assert_eq!(
        bucket,
        Some(expected),
        "selftest bucket must be the closed-set value `{expected}`"
    );
}

fn assert_provider_spec_invalid(
    output: &CommandOutput,
    provider: &str,
    field: &str,
    value_fragment: &str,
    accepted_fragment: &str,
) {
    assert_eq!(
        output.code, 1,
        "invalid provider spec must exit as config failure"
    );
    let parsed = parse_single_json_object(&output.stdout);
    assert!(
        parsed.is_ok(),
        "provider-spec failure must be exactly one JSON object, got stdout={:?}, stderr={:?}",
        output.stdout,
        output.stderr
    );
    let object = parsed.unwrap_or_default();
    assert_eq!(
        object.get("condition").and_then(Value::as_str),
        Some("provider_spec_invalid")
    );
    assert_eq!(object.get("bucket").and_then(Value::as_str), Some("config"));
    assert_eq!(
        object.get("provider").and_then(Value::as_str),
        Some(provider)
    );
    assert_eq!(object.get("field").and_then(Value::as_str), Some(field));
    assert!(
        object
            .get("value")
            .and_then(Value::as_str)
            .is_some_and(|value| value.contains(value_fragment)),
        "failure must name offending value `{value_fragment}`, got {object:?}"
    );
    let accepted = object
        .get("accepted")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<Vec<_>>();
    assert!(
        accepted
            .iter()
            .any(|value| value.contains(accepted_fragment)),
        "failure must name accepted set containing `{accepted_fragment}`, got {accepted:?}"
    );
    assert!(
        object
            .get("path")
            .and_then(Value::as_str)
            .is_some_and(|path| path.ends_with("cg/providers.toml")),
        "override-layer errors must name the providers.toml path, got {object:?}"
    );
}

fn assert_closed_bucket(output: &CommandOutput) {
    let parsed = parse_single_json_object(&output.stdout);
    assert!(
        parsed.is_ok(),
        "selftest failure must emit exactly one JSON object, got {:?}",
        output.stdout
    );
    let object = parsed.unwrap_or_default();
    let bucket = object.get("bucket").and_then(Value::as_str);
    assert!(
        matches!(
            bucket,
            Some("ok")
                | Some("auth")
                | Some("rate_limit")
                | Some("billing")
                | Some("timeout")
                | Some("format")
                | Some("unknown")
        ),
        "selftest bucket must be in the closed reporting set, got {bucket:?}"
    );
}

fn candidate_array(value: &Value) -> Vec<Value> {
    let candidates = value.get("candidates").and_then(Value::as_array);
    assert!(
        candidates.is_some(),
        "candidate envelope must carry a machine-readable candidates array"
    );
    candidates.cloned().unwrap_or_default()
}

fn host_candidate_providers(value: &Value) -> Vec<String> {
    candidate_array(value)
        .into_iter()
        .filter(|candidate| candidate.to_string().contains("host-session"))
        .map(|candidate| {
            candidate
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

fn sandboxed_install(prefix: &std::path::Path, home: &std::path::Path) -> CommandOutput {
    let script = repo_root().join("scripts/install.sh");
    let output = Command::new(script)
        .arg("--prefix")
        .arg(prefix)
        .env("HOME", home)
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("run install script");
    CommandOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

#[test]
fn cli_help_root() {
    let harness = CliHarness::new();
    let help = harness.run(&["--help"]);
    assert_eq!(help.code, 0);
    assert!(
        help.stdout.contains("model")
            && help.stdout.contains("status")
            && help.stdout.contains("selftest")
            && help.stdout.contains("disconnect <agent>")
            && help.stdout.contains("connect <agent> --remove-all"),
        "root help must list the shipped verb groups; stdout={:?}",
        help.stdout
    );
    let model_pos = help.stdout.find("model auth login");
    let status_pos = help.stdout.find("status");
    assert!(
        matches!((model_pos, status_pos), (Some(model), Some(status)) if model < status),
        "connecting a model must be presented before later verbs"
    );
    let bare = harness.run(&[]);
    assert!(
        bare.stdout.contains("model auth login"),
        "bare cg must show help text, got {:?}",
        bare.stdout
    );
    // STALE-CONTRACT FIX (2026-07-03): this assertion predated the daily-loop work, which made
    // `cg connect claude-code` the real injection-hook installer (exit 0, hooks written into
    // the harness-isolated HOME). That work documented this failure as a known baseline instead
    // of updating it; updated here to the shipped contract.
    let install = harness.run(&["connect", "claude-code"]);
    assert_eq!(
        install.code, 0,
        "connect claude-code installs the injection hooks (daily-loop shipped contract)"
    );
}

#[test]
fn invalid_format_values_are_typed_usage_errors() {
    let harness = CliHarness::new();
    for args in [
        &["status", "--format", "yaml"][..],
        &["status", "--format=yaml"][..],
    ] {
        let output = harness.run(args);
        assert_eq!(
            output.code, 2,
            "invalid --format must exit with usage code 2"
        );
        assert!(
            output.stderr.contains("invalid --format value"),
            "invalid --format must be explained on stderr, got {:?}",
            output.stderr
        );
    }
    let missing_id = harness.run(&["model", "show", "--format", "text"]);
    assert_eq!(
        missing_id.code, 2,
        "model show --format text without an id must be usage, not model `text` lookup"
    );
    assert!(
        missing_id.stderr.contains("missing model id"),
        "missing model id must be explained on stderr, got {:?}",
        missing_id.stderr
    );
}

#[test]
fn cli_version() {
    let output = CliHarness::new().run(&["--version"]);
    assert_eq!(output.code, 0);
    assert!(
        output.stdout.contains("cg") && output.stdout.chars().any(|ch| ch == '.'),
        "`cg --version` must print binary identity plus semantic version, got {:?}",
        output.stdout
    );
}

#[test]
fn cli_exit_code_usage_error() {
    let harness = CliHarness::new();
    let before = file_inventory(harness.data_root.path());
    let output = harness.run(&["frobnicate"]);
    assert_eq!(output.code, 2, "unknown verbs must exit with usage code 2");
    assert!(output.stdout.is_empty(), "usage errors keep stdout empty");
    assert!(
        output.stderr.to_lowercase().contains("error") && output.stderr.contains("Usage"),
        "stderr must carry error and usage text, got {:?}",
        output.stderr
    );
    assert_eq!(
        file_inventory(harness.data_root.path()),
        before,
        "usage errors must not persist state"
    );
}

#[test]
fn cli_exit_code_not_found() {
    let harness = CliHarness::new();
    let text = harness.run(&["model", "show", "no-such-model", "--format", "text"]);
    assert_eq!(text.code, 3);
    assert!(
        text.stdout.is_empty(),
        "text-mode not-found keeps stdout empty"
    );
    assert!(
        text.stderr.contains("no-such-model") && text.stderr.contains("not found"),
        "stderr must name the missing id"
    );
    let json = harness.run(&["model", "show", "no-such-model"]);
    assert_eq!(json.code, 3);
    assert_condition(&json, "not_found");
}

#[test]
fn cli_stdout_is_json_by_default_for_json_flag() {
    let harness = CliHarness::new();
    seed_connection_model(harness.data_root.path(), "primary", "reasoning-fixture");
    let output = harness.run(&["model", "list", "--format", "json", "--log", "info"]);
    assert_eq!(output.code, 0);
    let parsed = parse_single_json(&output.stdout);
    assert!(
        parsed.is_ok(),
        "stdout must be exactly one JSON value, got {:?}",
        output.stdout
    );
    assert!(
        parsed
            .unwrap_or(Value::Null)
            .to_string()
            .contains("reasoning-fixture"),
        "JSON output must contain the seeded model id"
    );
    assert!(
        output.stderr.contains("INFO") || output.stderr.contains("info"),
        "log lines must go to stderr under --log info"
    );
}

#[test]
fn cli_stdout_json_without_flag() {
    let harness = CliHarness::new();
    let json_default = harness.run(&["model", "list"]);
    assert!(
        parse_single_json(&json_default.stdout).is_ok(),
        "JSON must be the default output"
    );
    let text = harness.run(&["model", "list", "--format", "text"]);
    assert!(
        parse_single_json(&text.stdout).is_err(),
        "--format text must opt into prose, not JSON"
    );
}

#[test]
fn cli_exit_codes_stable_and_distinct_for_typed_failures() {
    let harness = CliHarness::new();
    let first = harness.run(&["selftest"]);
    let second = harness.run(&["selftest"]);
    assert_eq!(first.code, 4, "no connection selftest exits 4");
    assert_eq!(second.code, 4, "no connection selftest exit code is stable");
    assert_condition(&first, "no_host_session");
    assert_condition(&second, "no_host_session");

    let label = unique_label("dev");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Fail,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let fail_one = harness.run(&["selftest"]);
    let fail_two = harness.run(&["selftest"]);
    assert_eq!(fail_one.code, 6, "live call failures exit 6");
    assert_eq!(fail_two.code, 6, "live call failure exit code is stable");
    for code in [first.code, fail_one.code] {
        assert!(
            ![0, 2, 3].contains(&code),
            "typed failure code {code} must be distinct from success/usage/not-found"
        );
    }
    assert_ne!(first.code, fail_one.code);
}

#[test]
fn cli_model_list() {
    let harness = CliHarness::new();
    seed_connection_model(harness.data_root.path(), "primary", "remote-reasoning");
    let local_dir = harness.data_root.path().join("local-reasoning");
    fs::create_dir_all(&local_dir).expect("local model dir");
    let mut catalog = ModelConnectionCatalog::open(harness.data_root.path());
    catalog.register_local_model("local-reasoning", ModelKind::LlmClient, &local_dir);

    let output = harness.run(&["model", "list"]);
    assert_eq!(output.code, 0, "model list must exit successfully");
    let parsed = parse_single_json(&output.stdout);
    assert!(parsed.is_ok(), "model list must be JSON by default");
    let text = parsed.unwrap_or(Value::Null).to_string();
    assert!(text.contains("remote-reasoning"));
    assert!(text.contains("local-reasoning"));
    assert!(text.contains("\"connected\":true"));
    assert!(text.contains("\"connected\":false"));
    assert!(
        !text.contains("discovered-marketplace"),
        "model list must contain only registered entries"
    );
}

#[test]
fn cli_model_show() {
    let harness = CliHarness::new();
    seed_connection_model(harness.data_root.path(), "primary", "licensed-reasoning");
    let output = harness.run(&["model", "show", "licensed-reasoning"]);
    assert_eq!(output.code, 0, "model show must exit successfully");
    let parsed = parse_single_json(&output.stdout);
    assert!(parsed.is_ok(), "model show must be JSON");
    let text = parsed.unwrap_or(Value::Null).to_string();
    assert!(text.contains("licensed-reasoning"));
    assert!(text.contains("LlmClient"));
    assert!(text.contains("primary"));
    assert!(text.contains("Proprietary"));
    assert!(text.contains("provider.example.invalid/license"));
}

#[test]
fn cli_model_show_license_gated_renders_manual_instructions() {
    let harness = CliHarness::new();
    let instructions = unique_label("install-manually");
    let manual_url = "https://provider.example.test/manual-license";
    Catalog::open_with_entries_against_root(
        harness.data_root.path(),
        "connection-substrate",
        vec![license_entry("manual-model", &instructions, manual_url)],
    )
    .expect("seed catalog entry");
    let before = file_inventory(harness.data_root.path());
    let output = harness.run(&["model", "show", "manual-model"]);
    assert!(
        output.stdout.contains(&instructions) && output.stdout.contains(manual_url),
        "show output must render exact manual license instructions"
    );
    assert_eq!(
        file_inventory(harness.data_root.path()),
        before,
        "showing license-gated metadata must not fetch or mutate state"
    );
}

#[test]
fn cli_status_connection_slice_shows_name_quota_kind_of_spend_and_expectation() {
    let harness = CliHarness::new();
    let label = unique_label("dev");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let connect = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let connection_name = parse_single_json(&connect.stdout)
        .ok()
        .and_then(|value| {
            value
                .get("connection")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "claude-host-session".to_string());
    let status = harness.run(&["model", "status"]);
    assert_eq!(status.code, 0, "connected status must exit successfully");
    let parsed = parse_single_json_object(&status.stdout);
    assert!(
        parsed.is_ok(),
        "status default output must be exactly one JSON object"
    );
    let object = parsed.unwrap_or_default();
    assert_eq!(
        object.get("connection_name").and_then(Value::as_str),
        Some(connection_name.as_str())
    );
    assert_eq!(
        object.get("spend_owner").and_then(Value::as_str),
        Some(label.as_str())
    );
    assert!(
        json_string_field(&object, "store_path", "status JSON")
            .contains(harness.data_root.path().to_string_lossy().as_ref()),
        "status JSON must carry the active data store path"
    );
    let spend_kind = json_string_field(&object, "spend_kind", "status JSON");
    assert!(
        spend_kind.contains("quota") || spend_kind.contains("subscription"),
        "status JSON must name the kind of spend"
    );
    assert_eq!(
        object.get("ingestion_expectation").and_then(Value::as_str),
        Some(INGESTION_EXPECTATION)
    );
    assert_eq!(
        object.get("connection_works"),
        Some(&json!(false)),
        "status must not claim the connection works before selftest proves a real reply"
    );
    assert_eq!(
        object.get("proof_status").and_then(Value::as_str),
        Some("not_run"),
        "status must distinguish stored consent from a completed proof call"
    );
    assert!(
        status.stdout.contains(&connection_name)
            && status.stdout.contains(&label)
            && (status.stdout.contains("quota") || status.stdout.contains("subscription"))
            && status.stdout.contains(INGESTION_EXPECTATION),
        "status must show connection, spend owner, spend kind, and ingestion expectation"
    );
    assert!(
        status
            .stdout
            .contains(harness.data_root.path().to_string_lossy().as_ref()),
        "status must name the data store path"
    );
    let text = harness.run(&["model", "status", "--format", "text"]);
    assert!(
        parse_single_json(&text.stdout).is_err(),
        "status --format text must opt into prose"
    );
    assert!(
        text.stdout.contains(&connection_name)
            && text.stdout.contains(&label)
            && text.stdout.contains(INGESTION_EXPECTATION),
        "status text must carry the same trust facts as JSON"
    );
}

#[test]
fn cli_status_without_connection_says_so_and_points_at_connect() {
    let output = CliHarness::new().run(&["model", "status"]);
    // `cg model status` reports a missing connection with its own settled exit code 4
    // (`model_byok_status_exit_code`), where the retired top-level status exited 0.
    assert_eq!(output.code, 4);
    let parsed = parse_single_json_object(&output.stdout);
    assert!(
        parsed.is_ok(),
        "status default output must be exactly one JSON object when disconnected"
    );
    let object = parsed.unwrap_or_default();
    assert_eq!(object.get("connection_active"), Some(&json!(false)));
    assert_eq!(object.get("connection_name"), Some(&Value::Null));
    assert!(
        json_string_field(&object, "next_step", "disconnected status")
            .contains("cg model auth login"),
        "disconnected status JSON must name the connect verb"
    );
    assert!(
        json_string_field(&object, "message", "disconnected status")
            .contains("no model connection"),
        "disconnected status JSON must plainly name the inactive state"
    );
    assert!(
        output.stdout.contains("no model connection")
            && output.stdout.contains("cg model auth login"),
        "status without a connection must say so and point at connect"
    );
}

#[test]
fn model_auth_login_offers_present_host_session_first_by_name() {
    let harness = CliHarness::new();
    let label = unique_label("dev");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness
        .providers
        .set_auth_probe_stderr(Provider::Claude, "warning: config file is deprecated");
    let output = harness.run(&["model", "auth", "login", "--format", "json"]);
    assert_eq!(
        output.code, 0,
        "single-host candidate listing must exit successfully"
    );
    let parsed = parse_single_json(&output.stdout);
    assert!(parsed.is_ok(), "candidate listing must be JSON");
    let value = parsed.unwrap_or(Value::Null);
    let candidates = candidate_array(&value);
    let first = candidates.first().unwrap_or(&Value::Null);
    assert_eq!(
        first.get("kind").and_then(Value::as_str),
        Some("host-session")
    );
    assert_eq!(
        first.get("provider").and_then(Value::as_str),
        Some("claude")
    );
    assert_eq!(
        first.get("account").and_then(Value::as_str),
        Some(label.as_str())
    );
    assert!(
        first.to_string().contains("claude") && first.to_string().contains(&label),
        "host-session candidate named with provider and account must appear first"
    );
    let listing = value.to_string();
    let host_pos = listing.find("host-session");
    let env_pos = listing.find("env-var");
    assert!(
        matches!((host_pos, env_pos), (Some(host), Some(env)) if host < env),
        "env-var/stored-key options must appear below the host session"
    );
    assert_eq!(
        completion_count(&harness),
        0,
        "single-host candidate listing must not spend a completion before consent"
    );
    assert!(
        harness.providers.auth_status_probe_count() >= 1,
        "single-host listing must use no-spend auth-status probing"
    );
}

#[test]
fn model_auth_login_omits_provider_whose_cli_cannot_be_driven() {
    let harness = CliHarness::new();
    let claude_label = unique_label("claude");
    harness.providers.install(
        Provider::Claude,
        &claude_label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.install(
        Provider::Codex,
        &unique_label("codex"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness
        .providers
        .set_auth_probe_drivable(Provider::Claude, false);
    let first = harness.run(&["model", "auth", "login", "--format", "json"]);
    assert_eq!(
        first.code, 0,
        "drivable-filter listing must exit successfully"
    );
    let first_json = parse_single_json(&first.stdout);
    assert!(first_json.is_ok(), "candidate listing must be JSON");
    let first_providers = host_candidate_providers(&first_json.unwrap_or(Value::Null));
    assert_eq!(
        first_providers,
        vec!["codex".to_string()],
        "probe-less provider remains drivable while a failed probed provider is omitted"
    );
    assert!(
        first.stdout.contains("codex") && !first.stdout.contains("claude"),
        "failed Claude probe must be omitted while probe-less Codex is listed"
    );
    assert_eq!(
        completion_count(&harness),
        0,
        "candidate listing must not spend a completion probe before consent"
    );
    assert_eq!(
        harness
            .providers
            .auth_status_probe_count_for(Provider::Codex),
        0,
        "probe-less Codex must not be driven through auth status"
    );
    assert_eq!(
        harness
            .providers
            .auth_status_probe_count_for(Provider::Claude),
        1,
        "Claude remains gated by the declared auth-status probe"
    );
}

#[test]
fn model_auth_login_treats_probe_less_provider_as_drivable_without_auth_status() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Codex,
        &unique_label("codex"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness
        .providers
        .set_auth_probe_unsupported(Provider::Codex);

    let output = harness.run(&["model", "auth", "login", "--format", "json"]);
    assert_eq!(output.code, 0, "probe-less provider listing exits cleanly");
    let parsed = parse_single_json(&output.stdout);
    assert!(parsed.is_ok(), "probe-less listing must be JSON");
    let value = parsed.unwrap_or(Value::Null);
    assert_eq!(host_candidate_providers(&value), vec!["codex".to_string()]);
    assert!(
        output.stdout.contains(PROBELESS_HOST_ACCOUNT),
        "probe-less providers must use the explicit account-not-exposed label"
    );
    assert_eq!(
        harness
            .providers
            .auth_status_probe_count_for(Provider::Codex),
        0,
        "codex must not run the invalid auth status probe"
    );
}

#[test]
fn model_auth_login_explicit_absent_agy_fails_like_any_missing_provider() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Agy,
        &unique_label("agy"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.install(
        Provider::Codex,
        &unique_label("codex"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );

    let output = harness.run(&["model", "auth", "login", "--connection", "agy", "--consent"]);

    assert_eq!(output.code, 4);
    assert_condition(&output, "no_host_session");
    assert!(
        output.stdout.contains("provider spec is missing"),
        "absent agy must fail through the normal missing-provider path, got {:?}",
        output.stdout
    );
    assert_eq!(
        completion_count(&harness),
        0,
        "missing provider selection must not spend"
    );
}

#[test]
fn model_auth_login_bounds_hung_auth_probe_and_omits_provider() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("claude"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.install(
        Provider::Codex,
        &unique_label("codex"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.set_auth_probe_hangs(Provider::Claude);
    write_file(
        harness.data_root.path().join("cg/providers.toml"),
        r#"[providers.claude]
timeout_ms = 700
"#,
    );
    let started = std::time::Instant::now();
    let output = harness.run(&["model", "auth", "login", "--format", "json"]);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "hung auth probe must be bounded by the provider timeout"
    );
    assert_eq!(output.code, 0);
    let parsed = parse_single_json(&output.stdout);
    assert!(parsed.is_ok(), "bounded-probe listing must be JSON");
    assert_eq!(
        host_candidate_providers(&parsed.unwrap_or(Value::Null)),
        vec!["codex".to_string()],
        "timed-out probed provider is not drivable; probe-less provider remains available"
    );
    assert!(
        harness
            .providers
            .auth_status_probe_count_for(Provider::Claude)
            >= 1,
        "Claude hung through the auth probe path, not a completion call"
    );
    assert_eq!(completion_count(&harness), 0);

    let blocked = CliHarness::new();
    blocked.providers.install(
        Provider::Claude,
        &unique_label("claude"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    blocked.providers.set_auth_probe_hangs(Provider::Claude);
    write_file(
        blocked.data_root.path().join("cg/providers.toml"),
        r#"[providers.claude]
timeout_ms = 700
"#,
    );
    let output = blocked.run(&["model", "auth", "login", "--connection", "claude"]);
    assert_eq!(output.code, 4);
    assert_condition(&output, "no_host_session");
    assert!(
        output.stdout.contains("claude") && output.stdout.contains("auth probe timed out"),
        "not-drivable probe timeout must surface a reason instead of a generic missing-session envelope"
    );
}

#[test]
fn model_auth_login_two_host_logins_require_explicit_choice() {
    let harness = CliHarness::new();
    let claude_label = unique_label("claude");
    let codex_label = unique_label("codex");
    harness.providers.install(
        Provider::Claude,
        &claude_label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.install(
        Provider::Codex,
        &codex_label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let output = harness.run(&["model", "auth", "login", "--consent"]);
    assert_eq!(
        output.code, 2,
        "multiple host logins require explicit selection"
    );
    assert_condition(&output, "selection_required");
    let parsed = parse_single_json(&output.stdout);
    assert!(
        parsed.is_ok(),
        "selection_required output must be exactly one JSON value"
    );
    let provider_order = host_candidate_providers(&parsed.unwrap_or(Value::Null));
    assert_eq!(
        provider_order,
        vec!["claude".to_string(), "codex".to_string()],
        "selection_required candidates must be deterministic and provider-sorted"
    );
    for _ in 0..3 {
        let repeated = harness.run(&["model", "auth", "login", "--consent"]);
        let repeated_json = parse_single_json(&repeated.stdout);
        assert!(
            repeated_json.is_ok(),
            "repeated selection_required output must stay machine-readable"
        );
        assert_eq!(
            host_candidate_providers(&repeated_json.unwrap_or(Value::Null)),
            provider_order,
            "selection_required candidate order must not depend on map iteration"
        );
    }
    assert!(
        output.stdout.contains("claude") && output.stdout.contains("codex"),
        "selection_required envelope must include both candidates"
    );
    assert_eq!(
        ModelConnectionCatalog::open(harness.data_root.path()).connection_count(),
        0,
        "ambiguous connect must not persist a selection"
    );
    assert_eq!(
        completion_count(&harness),
        0,
        "ambiguous connect must not spend before explicit provider selection"
    );
    assert_eq!(
        harness
            .providers
            .auth_status_probe_count_for(Provider::Codex),
        0,
        "ambiguous connect must not run codex auth status because the spec declares no probe"
    );
    assert!(
        harness
            .providers
            .auth_status_probe_count_for(Provider::Claude)
            >= 1,
        "ambiguous connect must still use Claude's declared no-spend auth-status probe"
    );
    let selected = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "codex",
        "--consent",
    ]);
    assert_eq!(selected.code, 0);
    assert!(
        ModelConnectionCatalog::open(harness.data_root.path()).connection_count() == 1,
        "explicit selection must persist exactly one connection"
    );
    assert_host_session_catalog_row(
        harness.data_root.path(),
        "codex",
        "codex",
        PROBELESS_HOST_ACCOUNT,
        true,
    );
    let status = harness.run(&["model", "status"]);
    assert!(
        status.stdout.contains("codex") && status.stdout.contains(PROBELESS_HOST_ACCOUNT),
        "status must expose the selected provider and account-exposure state"
    );
    let proof = harness.run(&["selftest"]);
    assert_eq!(proof.code, 0);
    assert_eq!(
        harness.providers.last_completion_provider().as_deref(),
        Some("codex"),
        "proof must use the explicitly selected provider"
    );
    assert!(
        proof.stdout.contains(
            &harness
                .providers
                .expected_reply(Provider::Codex)
                .unwrap_or_default()
        ),
        "proof reply must come from the selected provider"
    );
}

#[test]
fn model_auth_login_consent_names_account_and_spend_before_anything_spends() {
    let harness = CliHarness::new();
    let label = unique_label("dev");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let observed = harness.run_writing_stdin_after_output(
        &["model", "auth", "login"],
        "Type yes to connect with consent",
        "yes\n",
    );
    assert!(
        observed.observed_before_exit,
        "interactive login must render and flush the candidate/consent prompt before reading stdin"
    );
    let output = observed.output;
    assert!(
        output.stdout.contains(&label) && output.stdout.contains("quota"),
        "consent prompt transcript must name account and spend"
    );
    assert!(
        output.stderr.contains("Available host-session candidate")
            && output.stderr.contains(&label)
            && output.stderr.contains("Type yes"),
        "interactive prompt must name the candidate and consent action"
    );
    assert_eq!(
        completion_count(&harness),
        0,
        "connect must not spend completion calls before consent"
    );
    let catalog = ModelConnectionCatalog::open(harness.data_root.path());
    assert!(
        catalog.connection_count() > 0,
        "consented connect must persist a connection record"
    );
    assert_host_session_catalog_row(harness.data_root.path(), "claude", "claude", &label, true);
}

#[test]
fn model_auth_login_declined_consent_leaves_no_consent_and_spends_nothing() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let output = harness.run_with_stdin(&["model", "auth", "login"], "no\n");
    assert!(
        output.stdout.contains("consent was not granted")
            && output.stdout.contains("cg model auth login"),
        "declined consent must tell the user how to grant later"
    );
    let catalog = ModelConnectionCatalog::open(harness.data_root.path());
    assert!(
        catalog.connection_count() > 0,
        "declined consent still records the connection reference"
    );
    assert_host_session_catalog_row(
        harness.data_root.path(),
        "claude",
        "claude",
        &harness
            .providers
            .account_label(Provider::Claude)
            .unwrap_or_default(),
        false,
    );
    let selftest = harness.run(&["selftest"]);
    assert_eq!(selftest.code, 4);
    assert_condition(&selftest, "consent_not_recorded");
    assert_eq!(completion_count(&harness), 0);
}

#[test]
fn model_auth_login_non_interactive_consent_reaches_same_enforced_record() {
    let harness = CliHarness::new();
    let label = unique_label("dev");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let output = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert_eq!(output.code, 0);
    let catalog = ModelConnectionCatalog::open(harness.data_root.path());
    assert!(
        catalog.connection_count() > 0,
        "non-interactive connect with consent must persist the enforced record"
    );
    assert_host_session_catalog_row(harness.data_root.path(), "claude", "claude", &label, true);
    let selftest = harness.run(&["selftest"]);
    assert_eq!(selftest.code, 0);
    assert_eq!(completion_count(&harness), 1);
    assert!(
        selftest.stdout.contains(
            &harness
                .providers
                .expected_reply(Provider::Claude)
                .unwrap_or_default()
        ),
        "selftest reply must equal the fixture-secret-derived value for {label}"
    );
}

#[test]
fn model_auth_login_non_interactive_without_consent_flag_records_no_consent() {
    let harness = CliHarness::new();
    let label = unique_label("dev");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let output = harness.run(&["model", "auth", "login", "--connection", "claude"]);
    assert!(
        output.stdout.contains("grant consent"),
        "non-interactive connect without --consent must name how to grant later"
    );
    assert_host_session_catalog_row(harness.data_root.path(), "claude", "claude", &label, false);
    let selftest = harness.run(&["selftest"]);
    assert_eq!(selftest.code, 4);
    assert_condition(&selftest, "consent_not_recorded");
    assert_eq!(completion_count(&harness), 0);
}

#[test]
fn connect_success_names_proof_command_and_ingestion_expectation() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let output = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert!(
        output.stdout.contains("cg selftest") && output.stdout.contains(INGESTION_EXPECTATION),
        "connect success must name the proof command and ingestion expectation"
    );
    assert!(
        output.stdout.contains("claude"),
        "connect success must name the chosen connection"
    );
}

#[test]
fn model_connect_carries_no_identity_surface() {
    let harness = CliHarness::new();
    let help = harness.run(&["model", "auth", "login", "--help"]);
    assert!(
        !help.stdout.contains("principal")
            && !help.stdout.contains("identity")
            && !help.stdout.contains("api-key"),
        "model connection help must not expose identity/api-key flags"
    );
    let listing = harness.run(&["model", "auth", "login", "--format", "json"]);
    assert!(
        !listing.stdout.contains("principal") && !listing.stdout.contains("identity"),
        "candidate envelope must not ask who the user is"
    );
    let auth = harness.run(&["auth"]);
    assert_eq!(
        auth.code, 2,
        "top-level identity auth verb does not ship here"
    );
}

#[test]
fn connect_persists_reference_not_credential_across_fresh_process() {
    let harness = CliHarness::new();
    let label = unique_label("dev");
    let secret_marker = unique_label("host-secret");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness
        .providers
        .write_secret(Provider::Claude, &secret_marker);
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert_host_session_catalog_row(harness.data_root.path(), "claude", "claude", &label, true);
    let status = run_cg(
        &["model", "status"],
        harness.data_root.path(),
        &harness.providers.path_env(),
        None,
    );
    assert!(
        status.stdout.contains("claude"),
        "fresh status process must see the durable connection"
    );
    assert!(!catalog_disk_contains(
        harness.data_root.path(),
        &secret_marker
    ));
    assert!(!catalog_disk_contains(
        harness.data_root.path(),
        &base64_basic(&secret_marker)
    ));
}

#[test]
fn standing_consent_survives_fresh_process_without_reprompt() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let before = completion_count(&harness);
    let output = run_cg(
        &["selftest"],
        harness.data_root.path(),
        &harness.providers.path_env(),
        None,
    );
    assert_eq!(output.code, 0);
    assert_eq!(
        completion_count(&harness),
        before + 1,
        "fresh process selftest must spend exactly one live call"
    );
    assert!(
        !output.stdout.contains("Do you consent"),
        "standing consent must not reprompt"
    );
}

#[test]
fn selftest_names_spend_and_connection_at_invocation() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let text = harness.run(&["selftest", "--format", "text"]);
    let announcement = text.stdout.find("one live call");
    let reply = text.stdout.find("proof-");
    assert!(
        matches!((announcement, reply), (Some(a), Some(r)) if a < r),
        "selftest must announce spend and connection before rendering the reply"
    );
    let json = harness.run(&["selftest"]);
    assert!(
        json.stdout.contains("connection") && json.stdout.contains("claude"),
        "JSON selftest must carry the connection name"
    );
}

#[test]
fn selftest_pass_shows_reply_from_driven_cli_and_discard_line() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let secret = unique_label("secret");
    harness.providers.write_secret(Provider::Claude, &secret);
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let output = harness.run(&["selftest"]);
    assert!(output.stdout.contains("PASS"));
    assert_eq!(
        harness.providers.last_completion_prompt().as_deref(),
        Some(SELFTEST_ECHO_PROMPT),
        "selftest prompt must be self-contained and must not imply an external token exists"
    );
    let expected = harness.providers.expected_reply(Provider::Claude);
    assert!(
        expected.is_some(),
        "a passing proof must leave a completion record with the full prompt"
    );
    let expected = expected.unwrap_or_default();
    assert!(
        output.stdout.contains(&expected),
        "displayed reply must equal the secret-derived subprocess reply"
    );
    assert!(output.stdout.contains("shown and discarded"));
    assert!(output.stdout.contains(INGESTION_EXPECTATION));
    assert_eq!(completion_count(&harness), 1);
}

#[test]
fn selftest_delivers_prompt_by_argv_when_provider_spec_declares_argv_mode() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Agy,
        &unique_label("agy"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    write_agy_provider_override(harness.data_root.path(), &["--print"]);
    harness
        .providers
        .write_secret(Provider::Agy, "agy-argv-secret");
    let connected = harness.run(&["model", "auth", "login", "--connection", "agy", "--consent"]);
    assert_eq!(
        connected.code, 0,
        "probe-less agy override connection must persist, got stdout={:?} stderr={:?}",
        connected.stdout, connected.stderr
    );

    let output = harness.run(&["selftest"]);

    assert_eq!(output.code, 0, "agy argv-mode selftest must pass");
    assert!(output.stdout.contains("PASS"));
    assert_eq!(
        completion_count(&harness),
        1,
        "agy proof must spend exactly one completion"
    );
    let record = harness
        .providers
        .completion_records()
        .pop()
        .expect("agy completion record");
    let argv = record
        .get("argv")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<Vec<_>>();
    assert_eq!(
        argv.first().map(String::as_str),
        Some("--print"),
        "agy invocation must use the declared print flag"
    );
    let prompt_from_argv = argv.get(1).cloned().unwrap_or_default();
    assert_eq!(
        prompt_from_argv, SELFTEST_ECHO_PROMPT,
        "selftest prompt must be a self-contained echo instruction, got {argv:?}"
    );
    assert_eq!(
        record.get("prompt").and_then(Value::as_str),
        Some(prompt_from_argv.as_str()),
        "fake agy must compute the reply from argv-delivered prompt, not stdin"
    );
    let expected = harness.providers.expected_reply(Provider::Agy);
    assert!(
        expected.is_some_and(|reply| output.stdout.contains(&reply)),
        "shown proof must come from the argv-delivered prompt"
    );
}

#[test]
fn selftest_fails_when_binary_and_login_present_but_live_call_fails() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    harness
        .providers
        .set_behavior(Provider::Claude, CompletionBehavior::Fail);
    let output = harness.run(&["selftest"]);
    assert_eq!(output.code, 6);
    assert_condition(&output, "live_call_failed");
    assert_bucket(&output, "unknown");
    assert!(
        !output.stdout.contains("PASS") && output.stdout.contains("failed"),
        "failed calls must not render connected/PASS"
    );
    assert_eq!(completion_count(&harness), 1, "no retry after failed call");
}

#[test]
fn selftest_spends_exactly_one_live_call() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let _ = harness.run(&["selftest"]);
    assert_eq!(completion_count(&harness), 1);
    let _ = harness.run(&["selftest"]);
    assert_eq!(completion_count(&harness), 2);
}

#[test]
fn selftest_shows_progress_while_the_driven_cli_runs() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::SleepThenSucceed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let observed = harness.run_observing_provider_before_exit(
        &["selftest", "--format", "text"],
        "waiting on provider",
        Provider::Claude,
    );
    let output = observed.output;
    assert!(
        observed.observed_before_exit,
        "progress frame must be observed before the cg process exits"
    );
    assert!(
        observed.observed_while_provider_running,
        "progress frame must be observed after the provider starts and before its exit marker"
    );
    assert!(
        output.stderr.contains("waiting on provider"),
        "progress frame must be visible on stderr during the provider wait"
    );
    let combined = format!("{}{}", output.stdout, output.stderr);
    let progress = combined.find("waiting on provider");
    let result = combined.find("PASS");
    assert!(
        matches!((progress, result), (Some(progress), Some(result)) if progress < result),
        "progress frame must appear during the provider wait before PASS"
    );
}

#[test]
fn selftest_json_carries_reply_connection_and_passfail() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness
        .providers
        .write_secret(Provider::Claude, "json-secret");
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let output = harness.run(&["selftest"]);
    let parsed = parse_single_json_object(&output.stdout);
    assert!(parsed.is_ok(), "selftest JSON must be exactly one object");
    let object = parsed.unwrap_or_default();
    assert_eq!(object.get("status").and_then(Value::as_str), Some("pass"));
    assert_eq!(
        object.get("connection").and_then(Value::as_str),
        Some("claude")
    );
    assert_eq!(
        object.get("reply").and_then(Value::as_str),
        harness
            .providers
            .expected_reply(Provider::Claude)
            .as_deref()
    );

    harness
        .providers
        .set_behavior(Provider::Claude, CompletionBehavior::Fail);
    let failed = harness.run(&["selftest"]);
    assert_condition(&failed, "live_call_failed");
    assert_closed_bucket(&failed);
}

#[test]
fn selftest_classifies_billing_and_rate_limit_buckets_end_to_end() {
    let billing = CliHarness::new();
    billing.providers.install(
        Provider::Claude,
        &unique_label("billing"),
        LoginState::Valid,
        CompletionBehavior::BillingRequired,
    );
    let _ = billing.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let billing_output = billing.run(&["selftest"]);
    assert_eq!(billing_output.code, 6);
    assert_condition(&billing_output, "live_call_failed");
    assert_bucket(&billing_output, "billing");
    assert!(
        billing_output.stdout.contains("billing") || billing_output.stdout.contains("quota"),
        "billing failures must keep actionable quota/billing language"
    );

    let limited = CliHarness::new();
    limited.providers.install(
        Provider::Claude,
        &unique_label("limited"),
        LoginState::Valid,
        CompletionBehavior::RateLimit,
    );
    let _ = limited.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let limited_output = limited.run(&["selftest"]);
    assert_eq!(limited_output.code, 6);
    assert_condition(&limited_output, "live_call_failed");
    assert_bucket(&limited_output, "rate_limit");
    assert!(
        limited_output.stdout.contains("rate") || limited_output.stdout.contains("quota"),
        "rate-limit failures must keep actionable quota/rate language"
    );
}

#[test]
fn selftest_persists_nothing_the_model_said() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness
        .providers
        .write_secret(Provider::Claude, "barrier-secret");
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let before = file_inventory(harness.data_root.path());
    let output = harness.run(&["selftest"]);
    let expected = harness.providers.expected_reply(Provider::Claude);
    assert!(expected.is_some(), "proof must include a completion prompt");
    let expected = expected.unwrap_or_default();
    assert!(
        output.stdout.contains(&expected),
        "proof reply must be the real subprocess value"
    );
    assert_eq!(
        file_inventory(harness.data_root.path()),
        before,
        "selftest must not persist model replies or other state"
    );
    assert!(!catalog_disk_contains(harness.data_root.path(), &expected));
}

#[test]
fn no_host_session_message_guides_to_model_connect() {
    let harness = CliHarness::new();
    let connect = harness.run(&["model", "auth", "login"]);
    assert_eq!(connect.code, 4);
    assert_condition(&connect, "no_host_session");
    let selftest = harness.run(&["selftest"]);
    assert_eq!(selftest.code, 4);
    assert_condition(&selftest, "no_host_session");
    for output in [connect, selftest] {
        assert!(
            !output.stdout.to_lowercase().contains("local model"),
            "no-host-session guidance must not recommend the removed local-model path: {}",
            output.stdout
        );
        assert!(
            !output.stdout.contains("model use"),
            "no-host-session guidance must not recommend `cg model use`: {}",
            output.stdout
        );
        assert!(
            output.stdout.contains("cg model connect"),
            "no-host-session guidance must point at hosted `cg model connect`: {}",
            output.stdout
        );
        let first_line = output.stdout.lines().next().unwrap_or_default();
        assert!(
            !first_line.to_lowercase().contains("api key"),
            "API key must not be the headline"
        );
    }
}

#[test]
fn connect_with_expired_host_login_says_relogin_not_no_host_session() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Expired,
        CompletionBehavior::Succeed,
    );
    let output = harness.run(&["model", "auth", "login", "--format", "json"]);
    assert_eq!(
        output.code, 6,
        "expired host login connect must use the typed provider/auth failure exit"
    );
    assert_condition(&output, "host_login_expired");
    assert!(
        output.stdout.contains("re-login with your agent")
            && !output.stdout.contains("no_host_session"),
        "expired host login must be actionable relogin guidance"
    );
}

#[test]
fn connect_with_unsafe_host_settings_reports_billing_risk_not_no_host_session() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    write_file(
        harness.data_root.path().join(".claude/settings.json"),
        r#"{"env":{"ANTHROPIC_API_KEY":"metered-key"}}"#,
    );
    let output = harness.run(&["model", "auth", "login", "--connection", "claude"]);
    assert_eq!(
        output.code, 6,
        "unsafe settings must fail as provider/billing risk"
    );
    assert_condition(&output, "live_call_failed");
    assert_bucket(&output, "billing");
    assert!(
        output.stdout.contains("ANTHROPIC_API_KEY") && !output.stdout.contains("no_host_session"),
        "unsafe settings must name the protected key instead of hiding as no host session"
    );
}

#[test]
fn stale_host_login_says_relogin_with_your_agent() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    harness
        .providers
        .set_login_state(Provider::Claude, LoginState::Expired);
    let output = harness.run(&["selftest"]);
    assert_eq!(output.code, 6);
    assert_condition(&output, "host_login_expired");
    assert_bucket(&output, "auth");
    assert!(output.stdout.contains("re-login with your agent"));
    assert!(
        !output.stdout.contains("expired-marker"),
        "raw subprocess marker must not be the headline"
    );
    assert!(completion_count(&harness) <= 1);
}

#[test]
fn missing_host_tool_distinct_from_stale_login() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    harness.providers.remove_provider(Provider::Claude);
    let output = harness.run(&["selftest"]);
    assert_eq!(output.code, 6);
    assert_condition(&output, "host_tool_missing");
    assert!(output.stdout.contains("tool is gone"));
    let status = harness.run(&["model", "status"]);
    assert!(status.stdout.contains("host_tool_missing"));
    assert_eq!(completion_count(&harness), 0);
    assert!(!output.stdout.contains("host_login_expired"));
}

#[test]
fn selftest_bounds_a_hung_provider_cli_with_typed_failure() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Hang,
    );
    write_file(
        harness.data_root.path().join("cg/providers.toml"),
        r#"[providers.claude]
timeout_ms = 700
"#,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let output = harness.run(&["selftest"]);
    assert_eq!(output.code, 6);
    assert_condition(&output, "live_call_failed");
    assert_bucket(&output, "timeout");
    assert!(output.stdout.contains("did not complete in time"));
}

#[test]
fn selftest_fails_when_driven_cli_returns_empty_reply() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Empty,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let output = harness.run(&["selftest"]);
    assert_eq!(output.code, 6);
    assert_condition(&output, "live_call_failed");
    assert_bucket(&output, "format");
    assert!(
        !output.stdout.contains("PASS"),
        "empty exit-0 reply is not proof"
    );
}

#[test]
fn corrupted_connections_file_fails_loud_and_typed() {
    let harness = CliHarness::new();
    seed_connection_model(harness.data_root.path(), "primary", "reasoning");
    let state_file = harness.data_root.path().join("cg/model-connections.toml");
    write_file(&state_file, "not valid toml {{{");
    for args in [&["model", "status"][..], &["selftest"][..]] {
        let output = harness.run(args);
        assert_eq!(output.code, 1);
        assert_condition(&output, "connections_file_unreadable");
        assert!(
            output.stdout.contains("model-connections.toml") && !output.stderr.contains("panicked"),
            "corruption must be typed and must name the file"
        );
    }
    assert_eq!(
        fs::read_to_string(state_file).unwrap_or_default(),
        "not valid toml {{{",
        "garbage state must not be silently overwritten"
    );
}

#[test]
fn forward_unknown_connection_records_survive_known_connection_persist() {
    let harness = CliHarness::new();
    let state_file = harness.data_root.path().join("cg/model-connections.toml");
    write_file(
        &state_file,
        r#"write_sequence = 7
active_connection_name = "future"

[[connections]]
name = "future"
consent_recorded = true
spend_owner = "future@example.test"

[connections.credential_reference]
env_var = ""

[connections.credential_reference.source.FutureHostSession]
provider = "future"
account = "future@example.test"
"#,
    );
    let catalog = ModelConnectionCatalog::open(harness.data_root.path());
    assert!(
        catalog.unreadable_connection_error("future").is_some(),
        "unknown future connection source must be tolerated per-record instead of corrupting the whole file"
    );
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let output = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert_eq!(output.code, 0);
    let after = fs::read_to_string(&state_file).expect("state file after known persist");
    assert!(
        after.contains("FutureHostSession")
            && after.contains("name = \"future\"")
            && after.contains("name = \"claude\""),
        "known writes must preserve forward-compatible unknown records and add the new known record, got {after}"
    );
}

#[test]
fn reconnect_existing_connection_preserves_standing_consent() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let output = harness.run(&["model", "auth", "login", "--connection", "claude"]);
    assert_eq!(output.code, 0);
    let catalog = ModelConnectionCatalog::open(harness.data_root.path());
    assert_eq!(catalog.connection_count(), 1);
    let selftest = harness.run(&["selftest"]);
    assert_eq!(
        selftest.code, 0,
        "standing consent must survive reconnect without --consent"
    );
}

#[test]
fn cli_connection_resolves_through_shared_catalog_api() {
    let harness = CliHarness::new();
    let label = unique_label("dev");
    harness.providers.install(
        Provider::Claude,
        &label,
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let _ = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    let catalog = ModelConnectionCatalog::open(harness.data_root.path());
    assert!(
        catalog.connection_count() > 0,
        "binary-created connection must be visible through ModelConnectionCatalog"
    );
    assert_host_session_catalog_row(harness.data_root.path(), "claude", "claude", &label, true);
    let status = harness.run(&["model", "status"]);
    assert!(
        status.stdout.contains(&label),
        "catalog spend owner and cg status must agree"
    );
}

#[test]
fn install_writes_no_agent_hook_and_changes_no_agent_config() {
    let sandbox = TempDir::new().expect("install sandbox");
    let home = sandbox.path().join("home");
    let prefix = sandbox.path().join("prefix");
    let config = home.join("agent-config/claude-code/settings.json");
    write_file(&config, "{\n  \"hooks\": []\n}\n");
    let before = fs::read_to_string(&config).expect("read config before install");
    let output = sandboxed_install(&prefix, &home);
    assert_eq!(output.code, 0, "install script must complete in sandbox");
    let after = fs::read_to_string(&config).expect("read config after install");
    assert_eq!(
        after, before,
        "install must not inject an agent hook before observe ships"
    );
    assert!(
        !catalog_disk_contains(&home, "cg observe --agent claude-code"),
        "sandbox must not contain an injected cg observe hook"
    );
}

#[test]
fn install_delivers_compiled_binary_not_shell_shim() {
    let sandbox = TempDir::new().expect("install sandbox");
    let home = sandbox.path().join("home");
    let prefix = sandbox.path().join("prefix");
    write_file(
        home.join("agent-config/claude-code/settings.json"),
        "{\n  \"hooks\": []\n}\n",
    );
    let output = sandboxed_install(&prefix, &home);
    assert_eq!(output.code, 0, "install script must complete in sandbox");
    let installed = prefix.join("bin/cg");
    let version = Command::new(&installed)
        .arg("--version")
        .env("HOME", &home)
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("run installed cg");
    assert!(version.status.success());
    let stdout = String::from_utf8_lossy(&version.stdout);
    assert!(stdout.contains("cg") && stdout.contains('.'));
    let bytes = fs::read(&installed).expect("installed cg bytes");
    assert!(
        !bytes.starts_with(b"#!"),
        "installed cg must be the compiled binary, not a shell shim"
    );
}

#[test]
fn cg_doctor_runs_only_free_checks_names_drift_and_spends_nothing() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.install(
        Provider::Codex,
        &unique_label("codex"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.install(
        Provider::Agy,
        &unique_label("agy"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    write_file(
        harness
            .providers
            .fixture_dir
            .join("claude")
            .join("help_flags"),
        "--output-format\n",
    );
    write_file(
        harness.providers.fixture_dir.join("claude").join("version"),
        "claude 2.2.0-drift",
    );
    let output = harness.run(&["doctor"]);
    assert_eq!(output.code, 1, "doctor drift findings exit 1");
    assert!(output.stdout.contains("--mcp-config"));
    assert!(output.stdout.contains("2.2.0-drift"));
    assert!(
        output
            .stdout
            .contains("WARNING: model/claude/flags-in-help")
    );
    assert!(output.stdout.contains("OK: model/codex/flags-in-help"));
    assert!(
        !output.stdout.contains("model/agy/flags-in-help"),
        "doctor must not enumerate a provider withdrawn from embedded specs"
    );
    assert!(
        !output
            .stdout
            .contains("codex fixture-provider 2026.6.4 help is missing"),
        "doctor must check codex exec --help rather than root help"
    );
    assert!(!output.stdout.contains("replacement flag"));
    let json_output = harness.run(&["doctor", "--json"]);
    let parsed = parse_single_json_object(&json_output.stdout);
    assert!(parsed.is_ok(), "doctor --json must emit one JSON object");
    let object = parsed.unwrap_or_default();
    assert_eq!(object.get("ok"), Some(&json!(false)));
    let checks_run = object
        .get("checks_run")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        checks_run
            .iter()
            .any(|check| check.as_str() == Some("model/claude/flags-in-help")),
        "doctor JSON must name the free drift check it ran"
    );
    let findings = object
        .get("findings")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        !findings.is_empty(),
        "doctor JSON must carry structured findings"
    );
    let finding = findings.first().unwrap_or(&Value::Null);
    assert_eq!(
        finding.get("check_id").and_then(Value::as_str),
        Some("model/claude/flags-in-help")
    );
    assert!(
        matches!(
            finding.get("severity").and_then(Value::as_str),
            Some("info") | Some("warning") | Some("error")
        ),
        "doctor severity must be in the closed set"
    );
    for field in ["message", "target", "fix_hint"] {
        assert!(
            finding
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty()),
            "doctor finding must carry non-empty `{field}`"
        );
    }
    assert!(
        finding.to_string().contains("--mcp-config") && finding.to_string().contains("claude"),
        "doctor finding must name the provider drift"
    );
    assert_eq!(completion_count(&harness), 0, "doctor never spends");

    let stderr_help = CliHarness::new();
    stderr_help.providers.install(
        Provider::Agy,
        &unique_label("agy"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    write_file(
        stderr_help
            .providers
            .fixture_dir
            .join("agy")
            .join("help_flags"),
        "--print\n",
    );
    write_agy_provider_override(
        stderr_help.data_root.path(),
        &["--cg-test-only-missing-flag"],
    );

    let output = stderr_help.run(&["doctor"]);

    assert_eq!(output.code, 1);
    assert!(
        output.stdout.contains(
            "agy fixture-provider 2026.6.4 help is missing required host-session flag --cg-test-only-missing-flag"
        ),
        "doctor must capture version text from stderr when a stderr-help provider has a genuine missing flag"
    );
    assert_eq!(completion_count(&stderr_help), 0, "doctor never spends");
}

#[test]
fn cg_doctor_bounds_hung_subcommand_help_check() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Codex,
        &unique_label("codex"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    harness.providers.set_help_probe_hangs(Provider::Codex);
    write_file(
        harness.data_root.path().join("cg/providers.toml"),
        r#"[providers.codex]
timeout_ms = 700
"#,
    );
    let started = std::time::Instant::now();
    let output = harness.run(&["doctor"]);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "doctor help drift checks must be bounded by the provider metadata timeout"
    );
    assert_eq!(output.code, 1);
    assert!(output.stdout.contains("model/codex/flags-in-help"));
    assert!(output.stdout.contains("help check timed out"));
    assert_eq!(completion_count(&harness), 0, "doctor never spends");
}

#[test]
fn cg_doctor_empty_required_help_flags_still_checks_binary_and_version() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Agy,
        &unique_label("agy"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    write_agy_provider_override(harness.data_root.path(), &[]);

    let output = harness.run(&["doctor"]);

    assert!(
        output.stdout.contains("OK: model/agy/flags-in-help"),
        "empty flags may render OK only after the provider binary/version path ran"
    );
    let agy_probe_records = harness
        .providers
        .probe_records()
        .into_iter()
        .filter(|record| record.get("provider").and_then(Value::as_str) == Some("agy"))
        .collect::<Vec<_>>();
    assert!(
        agy_probe_records.iter().any(|record| {
            record
                .get("argv")
                .and_then(Value::as_array)
                .is_some_and(|argv| argv == &vec![json!("--version")])
        }),
        "doctor must resolve agy and run --version even when no help flags are required"
    );
    assert!(
        !agy_probe_records.iter().any(|record| {
            record
                .get("argv")
                .and_then(Value::as_array)
                .is_some_and(|argv| argv == &vec![json!("--help")])
        }),
        "empty required_help_flags skips only flag-presence assertions"
    );
    assert_eq!(completion_count(&harness), 0, "doctor never spends");

    let missing = CliHarness::new();
    write_agy_provider_override(missing.data_root.path(), &[]);
    let output = missing.run(&["doctor"]);
    assert!(
        output
            .stdout
            .contains("agy provider tool was not found on PATH"),
        "missing provider with empty flags must warn instead of returning vacuous OK"
    );
    assert!(
        !output.stdout.contains("OK: model/agy/flags-in-help"),
        "doctor must never print OK for an unresolved provider"
    );

    let failed_version = CliHarness::new();
    failed_version.providers.install(
        Provider::Agy,
        &unique_label("agy"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    failed_version
        .providers
        .set_version_probe_fails(Provider::Agy);
    write_agy_provider_override(failed_version.data_root.path(), &[]);
    let output = failed_version.run(&["doctor"]);
    assert_eq!(output.code, 1);
    assert!(
        output
            .stdout
            .contains("agy version check exited unsuccessfully"),
        "failed --version must be reported before empty help flags can produce OK"
    );
    assert!(
        !output.stdout.contains("OK: model/agy/flags-in-help"),
        "doctor must not render OK when --version exits unsuccessfully"
    );
}

#[test]
fn provider_spec_loads_embedded_and_user_override_layers() {
    let dir = TempDir::new().expect("provider spec tempdir");
    assert!(
        validate_embedded_provider_specs().is_ok(),
        "embedded provider specs must validate before any user command can load them"
    );
    let specs = load_provider_specs(dir.path(), &[]);
    for provider in ["claude", "codex"] {
        assert!(
            specs.providers.contains_key(provider),
            "embedded provider spec must include {provider}"
        );
    }
    assert!(
        !specs.providers.contains_key("agy"),
        "agy must not ship as an embedded provider until a pure-completion drive is verified"
    );
    let claude = specs.providers.get("claude");
    assert!(
        claude.is_some_and(|spec| {
            spec.binary == "claude"
                && spec.prompt_delivery.as_str() == "stdin"
                && spec
                    .env_remove
                    .iter()
                    .any(|name| name == "ANTHROPIC_API_KEY")
                && spec.reply_framing.as_str() == "json"
                && spec.consent_required
        }),
        "claude spec must declare binary, stdin prompt, env removal, JSON framing, and consent"
    );
    let codex = specs.providers.get("codex").expect("codex spec");
    assert!(
        codex.auth_probe_args.is_empty()
            && codex.timeout_ms > 2500
            && codex
                .required_help_flags
                .contains(&"--output-last-message".to_string()),
        "codex spec must be probe-less, generous-timeout, and checked through its completion subcommand help"
    );
    write_agy_provider_override(dir.path(), &["--print"]);
    let with_agy_override = load_provider_specs(dir.path(), &[]);
    let agy = with_agy_override
        .providers
        .get("agy")
        .expect("agy override spec");
    assert!(
        agy.auth_probe_args.is_empty()
            && agy.timeout_ms > 300_000
            && agy.prompt_delivery.as_str() == "argv"
            && agy.completion_args == vec!["--print".to_string()]
            && agy.required_help_flags == vec!["--print".to_string()]
            && agy
                .last_verified
                .contains("pure-completion drive not embedded"),
        "agy may return through a full user override, not through the embedded catalog"
    );

    let overrides = [ProviderOverride {
        provider: "claude".to_string(),
        timeout_ms: Some(42_000),
        remove_env: Some(Vec::new()),
        consent_required: Some(false),
    }];
    let layered = load_provider_specs(dir.path(), &overrides);
    let layered_claude = layered.providers.get("claude");
    assert!(
        layered_claude.is_some_and(|spec| {
            spec.timeout_ms == 42_000
                && spec
                    .env_remove
                    .iter()
                    .any(|name| name == "ANTHROPIC_API_KEY")
                && spec.consent_required
        }),
        "overrides may change capability description but not env-removal or consent ownership"
    );
}

#[test]
fn provider_spec_rejects_uninterpretable_override_values_before_any_provider_spawn() {
    for (toml, field, value, accepted) in [
        (
            r#"[providers.claude]
prompt_delivery = "telepathy"
"#,
            "prompt_delivery",
            "telepathy",
            "stdin",
        ),
        (
            r#"[providers.claude]
reply_framing = "markdown"
"#,
            "reply_framing",
            "markdown",
            "json",
        ),
        (
            r#"[providers.claude]
completion_args = ["-p", "{prompt}"]
"#,
            "completion_args",
            "{prompt}",
            "{output_file}",
        ),
        (
            r#"[providers.claude.success]
path = "/is_error"
"#,
            "success",
            "no recognized clause",
            "equals",
        ),
        (
            r#"[providers.claude]
prompt_delviery = "stdin"
"#,
            "prompt_delviery",
            "prompt_delviery",
            "known provider spec fields",
        ),
    ] {
        let harness = CliHarness::new();
        harness.providers.install(
            Provider::Claude,
            &unique_label("claude"),
            LoginState::Valid,
            CompletionBehavior::Succeed,
        );
        write_file(harness.data_root.path().join("cg/providers.toml"), toml);

        let output = harness.run(&["model", "auth", "login", "--format", "json"]);

        assert_provider_spec_invalid(&output, "claude", field, value, accepted);
        assert_eq!(
            harness.providers.probe_count(),
            0,
            "invalid `{field}` must be rejected before doctor/auth/help probes"
        );
        assert_eq!(
            completion_count(&harness),
            0,
            "invalid `{field}` must be rejected before any completion spend"
        );
        assert!(
            try_load_provider_specs(harness.data_root.path(), &[]).is_err(),
            "library loader must reject the same invalid override instead of falling back to embedded specs"
        );
    }
}

#[test]
fn provider_spec_rejects_incomplete_new_provider_before_any_provider_spawn() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Agy,
        &unique_label("agy"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    write_file(
        harness.data_root.path().join("cg/providers.toml"),
        r#"[providers.shadow]
binary = "agy"
"#,
    );

    let output = harness.run(&["model", "auth", "login", "--format", "json"]);

    assert_provider_spec_invalid(
        &output,
        "shadow",
        "prompt_delivery",
        "missing required",
        "new providers must declare",
    );
    assert_eq!(
        harness.providers.probe_count(),
        0,
        "incomplete new provider spec must fail before auth/help probes"
    );
    assert_eq!(
        completion_count(&harness),
        0,
        "incomplete new provider spec must fail before any completion spend"
    );
    assert!(
        try_load_provider_specs(harness.data_root.path(), &[]).is_err(),
        "library loader must not create a drivable provider from implicit defaults"
    );
}

#[test]
fn provider_spec_unreadable_override_path_fails_loud_before_any_provider_spawn() {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("claude"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    fs::create_dir_all(harness.data_root.path().join("cg/providers.toml"))
        .expect("create unreadable provider override path");

    let output = harness.run(&["model", "auth", "login", "--format", "json"]);

    assert_provider_spec_invalid(
        &output,
        "<override>",
        "providers.toml",
        "directory",
        "readable TOML",
    );
    assert_eq!(
        harness.providers.probe_count(),
        0,
        "unreadable override path must be rejected before auth/help probes"
    );
    assert_eq!(
        completion_count(&harness),
        0,
        "unreadable override path must be rejected before any completion spend"
    );
    assert!(
        try_load_provider_specs(harness.data_root.path(), &[]).is_err(),
        "library loader must not fall back to embedded specs when the override path is present but unreadable"
    );
}

#[test]
fn claude_invocation_uses_equals_form_and_stdin_prompt() {
    let prompt = "prove the host session path";
    let invocation = construct_claude_invocation(prompt);
    assert_eq!(
        invocation.argv,
        vec![
            "-p".to_string(),
            "--output-format=json".to_string(),
            "--strict-mcp-config".to_string(),
            "--mcp-config={\"mcpServers\":{}}".to_string(),
            "--tools=\"\"".to_string(),
        ],
        "claude invocation must pin the exact host-session argv"
    );
    assert_eq!(
        invocation.stdin, prompt,
        "prompt must be delivered on stdin"
    );
    assert!(
        !invocation.argv.iter().any(|arg| arg == prompt),
        "prompt must not be a trailing positional argument"
    );

    let argv = invocation
        .argv
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let positive = run_variadic_fake(&argv, prompt);
    assert_eq!(positive.code, 0);
    assert!(
        positive.stdout.contains("proof-"),
        "equals-form/stdin prompt should parse as a real completion"
    );
    let negative = run_variadic_fake(&["-p", "--mcp-config", prompt], "");
    assert_ne!(negative.code, 0);
    assert!(negative.stderr.contains("MCP config not found"));

    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let connected = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert_eq!(connected.code, 0);
    let output = Command::new(cg_bin())
        .args(["selftest"])
        .env("CG_DATA_ROOT", harness.data_root.path())
        .env("PATH", harness.providers.path_env())
        .env("HOME", harness.data_root.path())
        .env("ANTHROPIC_AUTH_TOKEN", "junk-auth-token")
        .env("ANTHROPIC_API_KEY", "junk-api-key")
        .env("CLAUDE_CODE_USE_BEDROCK", "1")
        .env("CLAUDE_CODE_USE_VERTEX", "1")
        .env("CLAUDE_CODE_USE_FOUNDRY", "1")
        .env("ANTHROPIC_BASE_URL", "https://metered.example.invalid")
        .env("CLAUDE_CODE_OAUTH_TOKEN", "subscription-oauth-token")
        .env("OPENAI_API_KEY", "junk-openai")
        .env("CODEX_API_KEY", "junk-codex")
        .output()
        .expect("run cg selftest with junk env");
    assert!(
        output.status.success(),
        "selftest should pass through fake provider"
    );
    let records = harness.providers.completion_records();
    let env = records
        .last()
        .and_then(|record| record.get("env"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for name in [
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
    ] {
        assert!(
            !env.contains_key(name),
            "fake provider must not observe protected env var {name}"
        );
    }
    assert_eq!(
        env.get("CLAUDE_CODE_OAUTH_TOKEN").and_then(Value::as_str),
        Some("subscription-oauth-token"),
        "host-session drive must preserve the subscription OAuth credential path"
    );
}

#[test]
fn reply_extraction_engine_matches_recorded_envelopes() {
    let dir = TempDir::new().expect("reply tempdir");
    let codex_file = dir.path().join("last-message.txt");
    std::fs::write(&codex_file, CODEX_LAST_MESSAGE).expect("codex fixture");

    let claude = extract_reply("claude", "json", CLAUDE_SUCCESS, None);
    assert_eq!(
        claude.as_deref(),
        Ok(
            "Host session proof received. I can answer through the logged-in Claude CLI; turn capture should stay off until ingestion ships."
        ),
        "claude JSON framing must extract /result"
    );
    let codex = extract_reply("codex", "file", "", Some(&codex_file));
    assert_eq!(
        codex.as_deref(),
        Ok(
            "Host session proof received. I can answer through this logged-in Codex CLI; no turn capture should begin until ingestion ships.\n"
        ),
        "codex file framing must read --output-last-message"
    );
    let agy = extract_reply("agy", "text", AGY_PLAIN, None);
    assert_eq!(
        agy.as_deref(),
        Ok(
            "Host session proof received. I can answer through the local Agy CLI; capture remains disabled until ingestion ships.\n"
        ),
        "text framing returns the whole stdout"
    );
    let missing = extract_reply("claude", "json", "{\"type\":\"result\"}", None);
    assert!(
        matches!(
            missing,
            Err(cg_cli::ReplyError::MissingPath { ref path, ref framing })
                if path == "/result" && framing == "json"
        ),
        "missing-path errors must name /result and the framing"
    );
    let error_with_result = extract_reply(
        "claude",
        "json",
        r#"{"type":"result","subtype":"success","is_error":true,"api_error_status":402,"result":"billing required"}"#,
        None,
    );
    assert!(
        matches!(
            error_with_result,
            Err(cg_cli::ReplyError::Unsuccessful { .. })
        ),
        "json success predicates must reject provider error envelopes even when /result exists"
    );
}

#[test]
fn classify_rules_map_401_to_host_login_expired_via_is_error_not_subtype() {
    let classified = classify_provider_reply("claude", CLAUDE_401);
    assert!(
        matches!(
            classified,
            Classification::HostLoginExpired { ref message }
                if message.contains("Invalid API key")
        ),
        "401 envelopes with is_error=true must map to host_login_expired even when subtype=success"
    );
    assert_ne!(
        classified,
        Classification::Success,
        "subtype=success alone must not classify an error envelope as success"
    );

    let billing = classify_provider_reply("claude", CLAUDE_402);
    assert!(
        matches!(
            billing,
            Classification::BillingRequired { ref message }
                if message.contains("billing") || message.contains("quota")
        ),
        "402 envelopes must map to the billing bucket instead of unknown"
    );

    let limited = classify_provider_reply("claude", CLAUDE_429);
    assert!(
        matches!(
            limited,
            Classification::RateLimited { ref message }
                if message.contains("rate") || message.contains("quota")
        ),
        "429 envelopes must map to the rate_limit bucket instead of unknown"
    );

    let other = classify_provider_reply(
        "claude",
        r#"{"type":"result","subtype":"success","is_error":true,"result":"provider down"}"#,
    );
    assert!(
        matches!(other, Classification::LiveCallFailed { ref message } if message == "provider down"),
        "non-401 is_error envelopes map to live_call_failed with /result message"
    );
}

#[test]
fn cg_doctor_and_selftest_runners_mount_from_shared_conventions_layer() {
    let dir = TempDir::new().expect("doctor registry override root");
    write_file(
        dir.path().join("cg/providers.toml"),
        r#"[providers.claude]
required_help_flags = ["--cg-test-only-missing-flag"]
"#,
    );
    let mut registry = CheckRegistry::default();
    register_model_connection_checks_with_root(&mut registry, dir.path());
    assert!(
        registry
            .checks()
            .iter()
            .any(|check| check.check_id == "model/claude/flags-in-help" && !check.spending),
        "model-connection checks must register through the host-handed conventions registrar"
    );
    assert!(
        registry.checks().iter().any(|check| check.spending),
        "shared registry must preserve the free-vs-spending marker"
    );
    let doctor = run_doctor_checks(&registry);
    assert!(
        doctor
            .checks_run
            .iter()
            .any(|id| id == "model/claude/flags-in-help"),
        "shared doctor runner must execute the registered model-provider drift check"
    );
    assert!(
        doctor
            .findings
            .iter()
            .any(|finding| finding.check_id == "model/claude/flags-in-help"),
        "drift check must produce a stable finding through the shared registry"
    );
    let selftest = run_selftest_checks(&registry);
    assert!(
        !selftest.ok,
        "shared selftest runner must fail closed unless the host binary wires a live proof"
    );
    assert!(
        selftest.checks_run.iter().any(|id| id.contains("model")),
        "selftest runner must mount from the same conventions registry"
    );
    assert!(
        !selftest
            .checks_run
            .iter()
            .any(|id| id == "model/claude/flags-in-help"),
        "selftest runner must not execute free doctor-only checks"
    );

    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    write_file(
        harness
            .providers
            .fixture_dir
            .join("claude")
            .join("help_flags"),
        "--output-format\n",
    );
    let output = harness.run(&["doctor", "--json"]);
    let parsed = parse_single_json_object(&output.stdout).unwrap_or_default();
    let checks_run = parsed
        .get("checks_run")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        checks_run
            .iter()
            .any(|id| id.as_str() == Some("model/claude/flags-in-help")),
        "cg doctor binary output must come from the same registered conventions runner"
    );
}
