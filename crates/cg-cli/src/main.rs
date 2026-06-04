use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cg_catalog::{Catalog, CredentialSource, ModelConnectionCatalog, Source};
use cg_cli::{Classification, construct_claude_invocation, extract_reply, load_provider_specs};
use serde::Deserialize;
use serde_json::{Value, json};

const INGESTION_EXPECTATION: &str = "automatic capture switches on when ingestion ships";
const CONNECTION_SCOPE: &str = "connection-substrate";
const PROVIDERS: [&str; 3] = ["claude", "codex", "agy"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Text,
}

#[derive(Clone, Debug)]
struct Candidate {
    provider: String,
    account: String,
    account_exposed: bool,
}

#[derive(Clone, Debug)]
enum ProbeResult {
    Valid {
        account: String,
        account_exposed: bool,
    },
    Expired,
    NotDrivable,
    Missing,
}

#[derive(Clone, Debug)]
struct AuthAccount {
    label: String,
    exposed: bool,
}

#[derive(Debug)]
struct ProviderRun {
    status: Option<ExitStatus>,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

#[derive(Debug, Deserialize)]
struct RawCatalogRegistrations {
    entries: Vec<cg_catalog::CatalogEntry>,
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let code = run(args);
    std::process::exit(code);
}

fn run(args: Vec<String>) -> i32 {
    match args.first().map(String::as_str) {
        None | Some("--help") | Some("-h") => {
            print_help();
            0
        }
        Some("--version") | Some("-V") => {
            println!("cg {}", env!("CARGO_PKG_VERSION"));
            0
        }
        Some("model") => handle_model(&args[1..]),
        Some("status") => handle_status(&args[1..]),
        Some("selftest") => handle_selftest(&args[1..]),
        Some("doctor") => handle_doctor(&args[1..]),
        Some("disconnect") => handle_disconnect(&args[1..]),
        Some("connect") => handle_connect_cleanup(&args[1..]),
        Some("auth") => usage_error("top-level auth is not available; use `cg model auth login`"),
        Some(other) => usage_error(&format!("unknown command `{other}`")),
    }
}

fn print_help() {
    println!(
        "Usage: cg <command>\n\nCommands:\n  model auth login    Connect cg to an existing model host session\n  model list          List registered models\n  model show <id>     Show model metadata\n  status              Show model connection status\n  selftest            Spend one consented proof call\n  doctor              Run free model-provider diagnostics"
    );
}

fn usage_error(message: &str) -> i32 {
    eprintln!("error: {message}\nUsage: cg <command>");
    2
}

fn handle_model(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("list") => handle_model_list(&args[1..]),
        Some("show") => handle_model_show(&args[1..]),
        Some("auth") if args.get(1).map(String::as_str) == Some("login") => {
            handle_model_auth_login(&args[2..])
        }
        Some("auth") => usage_error("expected `cg model auth login`"),
        _ => usage_error("expected `cg model list`, `cg model show`, or `cg model auth login`"),
    }
}

fn handle_model_list(args: &[String]) -> i32 {
    let format = output_format(args);
    if has_log_info(args) {
        eprintln!("INFO listing registered model connection catalog entries");
    }
    let catalog = ModelConnectionCatalog::open(data_root());
    if let Some(error) = catalog.load_error() {
        return unreadable_connections_file(error);
    }
    let models = catalog.list_models();
    if format == OutputFormat::Text {
        for model in models {
            let connected = model
                .connection_name
                .as_deref()
                .and_then(|name| catalog.read_connection(name))
                .is_some();
            println!("{}\t{:?}\tconnected={}", model.id, model.kind, connected);
        }
        return 0;
    }
    let rows = models
        .into_iter()
        .map(|model| {
            let connected = model
                .connection_name
                .as_deref()
                .and_then(|name| catalog.read_connection(name))
                .is_some();
            json!({
                "id": model.id,
                "kind": format!("{:?}", model.kind),
                "connection": model.connection_name,
                "connected": connected,
            })
        })
        .collect::<Vec<_>>();
    print_json(&json!(rows));
    0
}

fn handle_model_show(args: &[String]) -> i32 {
    let Some(id) = positional_arg(args) else {
        return usage_error("missing model id");
    };
    let format = output_format(args);
    let root = data_root();
    let catalog = ModelConnectionCatalog::open(&root);
    if let Some(error) = catalog.load_error() {
        return unreadable_connections_file(error);
    }
    let entry = raw_registration_entry(&root, &id).or_else(|| {
        Catalog::open_with_scope_against_root(&root, CONNECTION_SCOPE)
            .ok()
            .and_then(|catalog| catalog.show(&id).ok())
    });
    let Some(model) = catalog.read_model(&id).or_else(|| {
        entry.as_ref().map(|entry| cg_catalog::ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: match &entry.source {
                Source::NamedConnection { connection_name } => Some(connection_name.clone()),
                _ => None,
            },
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: entry.local_path.clone(),
            hosted_endpoint: None,
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        })
    }) else {
        if format == OutputFormat::Text {
            eprintln!("model `{id}` not found");
        } else {
            print_json(&json!({
                "condition": "not_found",
                "message": format!("model `{id}` not found"),
            }));
        }
        return 3;
    };
    if format == OutputFormat::Text {
        println!("id: {}", model.id);
        println!("kind: {:?}", model.kind);
        if let Some(connection) = &model.connection_name {
            println!("connection: {connection}");
        }
        if let Some(entry) = entry {
            println!("license: {}", entry.license.spdx);
            println!("license_url: {}", entry.license.url);
            if let Some(instructions) = entry.install_instructions {
                println!("install_instructions: {instructions}");
            }
            if let Some(url) = entry.manual_install_url {
                println!("manual_install_url: {url}");
            }
        }
        return 0;
    }
    let license = entry.as_ref().map(|entry| {
        json!({
            "spdx": entry.license.spdx,
            "url": entry.license.url,
            "requires_manual_acceptance": entry.license.requires_manual_acceptance,
            "install_instructions": entry.install_instructions,
            "manual_install_url": entry.manual_install_url,
        })
    });
    print_json(&json!({
        "id": model.id,
        "kind": format!("{:?}", model.kind),
        "connection": model.connection_name,
        "license": license,
    }));
    0
}

fn raw_registration_entry(root: &Path, id: &str) -> Option<cg_catalog::CatalogEntry> {
    let path = root.join("cg/catalog-registrations.toml");
    let text = fs::read_to_string(path).ok()?;
    let registrations = toml::from_str::<RawCatalogRegistrations>(&text).ok()?;
    registrations
        .entries
        .into_iter()
        .find(|entry| entry.id == id)
}

fn handle_status(args: &[String]) -> i32 {
    let format = output_format(args);
    let root = data_root();
    let catalog = ModelConnectionCatalog::open(&root);
    if let Some(error) = catalog.load_error() {
        return unreadable_connections_file(error);
    }
    let Some(connection) = selected_connection(&catalog) else {
        let message = "no model connection is active; run cg model auth login. local model installation is available before env-var key setup.";
        if format == OutputFormat::Text {
            println!("{message}");
        } else {
            print_json(&json!({
                "connection_active": false,
                "connection_name": Value::Null,
                "next_step": "cg model auth login",
                "message": message,
            }));
        }
        return 0;
    };
    let (provider, account) = provider_account(&connection);
    let tool_condition = if find_binary(&provider).is_some() {
        Value::Null
    } else {
        json!("host_tool_missing")
    };
    let spend_owner = if connection.spend_owner.is_empty() {
        account.clone()
    } else {
        connection.spend_owner.clone()
    };
    let consent_message = if connection.consent_recorded {
        "stored consent exists; run cg selftest for a live proof"
    } else {
        "connection reference exists but consent is not recorded; run cg model auth login --connection <provider> --consent"
    };
    if format == OutputFormat::Text {
        println!("connection: {}", connection.name);
        println!("provider: {provider}");
        println!("spend_owner: {spend_owner}");
        println!("consent_recorded: {}", connection.consent_recorded);
        println!("spend_kind: subscription quota");
        println!("store_path: {}", root.display());
        println!("{INGESTION_EXPECTATION}");
        if tool_condition != Value::Null {
            println!("host_tool_missing: provider tool is gone");
        }
        return 0;
    }
    print_json(&json!({
        "connection_active": true,
        "connection_name": connection.name,
        "provider": provider,
        "spend_owner": spend_owner,
        "store_path": root.display().to_string(),
        "spend_kind": "subscription quota",
        "consent_recorded": connection.consent_recorded,
        "ingestion_expectation": INGESTION_EXPECTATION,
        "connection_works": false,
        "proof_status": "not_run",
        "tool_condition": tool_condition,
        "message": if tool_condition == json!("host_tool_missing") {
            "host_tool_missing: provider tool is gone"
        } else {
            consent_message
        },
    }));
    0
}

fn handle_model_auth_login(args: &[String]) -> i32 {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!(
            "Usage: cg model auth login [--connection PROVIDER] [--consent] [--format json|text]\n\nConnect cg to an already logged-in agent CLI host session."
        );
        return 0;
    }

    let selected = flag_value(args, "--connection");
    let consent_flag = has_flag(args, "--consent");
    let candidates = discover_candidates();

    if candidates.is_empty() {
        let expired = PROVIDERS
            .into_iter()
            .any(|provider| matches!(probe_provider(provider), ProbeResult::Expired));
        if expired {
            return failure(
                6,
                "host_login_expired",
                "auth",
                "Host login expired; re-login with your agent before connecting cg.",
            );
        }
        return no_host_session_failure();
    }

    if let Some(provider) = selected {
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.provider == provider)
            .cloned()
        else {
            let result = probe_provider(&provider);
            return match result {
                ProbeResult::Expired => failure(
                    6,
                    "host_login_expired",
                    "auth",
                    "Host login expired; re-login with your agent before connecting cg.",
                ),
                _ => no_host_session_failure(),
            };
        };
        return persist_connection(candidate, consent_flag, None);
    }

    if candidates.len() > 1 {
        return selection_required(&candidates);
    }

    let candidate = candidates[0].clone();
    if consent_flag {
        return persist_connection(candidate, true, None);
    }
    let mut stdin_text = String::new();
    let _ = std::io::stdin().read_to_string(&mut stdin_text);
    match stdin_text.trim().to_ascii_lowercase().as_str() {
        "yes" | "y" => persist_connection(candidate, true, Some(true)),
        "no" | "n" => persist_connection(candidate, false, Some(false)),
        _ => {
            print_json(&json!({
                "candidates": candidate_envelope(&candidates),
                "message": "Available host-session candidate found. Run cg model auth login --connection <provider> --consent to connect; local model remains available before env-var key setup.",
            }));
            0
        }
    }
}

fn persist_connection(candidate: Candidate, consent: bool, interactive: Option<bool>) -> i32 {
    let mut catalog = ModelConnectionCatalog::open(data_root());
    if let Some(error) = catalog.load_error().map(str::to_string) {
        return unreadable_connections_file(&error);
    }
    let existing_consent = catalog
        .read_connection(&candidate.provider)
        .is_some_and(|connection| same_host_session_connection(&connection, &candidate));
    catalog.connect_host_session_with_account_exposure(
        &candidate.provider,
        &candidate.provider,
        &candidate.account,
        candidate.account_exposed,
    );
    if consent || existing_consent {
        catalog.record_consent(&candidate.provider, &candidate.account);
    }
    let consent_recorded = consent || existing_consent;
    let message = if consent_recorded {
        format!(
            "Connected {provider} for account {account}; spend uses subscription quota. Run cg selftest for proof. {INGESTION_EXPECTATION}",
            provider = candidate.provider,
            account = candidate.account
        )
    } else if interactive == Some(false) {
        format!(
            "consent was not granted for {provider}; run cg model auth login --connection {provider} --consent to grant later.",
            provider = candidate.provider
        )
    } else {
        format!(
            "Connection reference recorded for {provider}; grant consent with cg model auth login --connection {provider} --consent before cg selftest.",
            provider = candidate.provider
        )
    };
    print_json(&json!({
        "ok": true,
        "connection": candidate.provider,
        "provider": candidate.provider,
        "account": candidate.account,
        "spend_kind": "subscription quota",
        "consent_recorded": consent_recorded,
        "ingestion_expectation": INGESTION_EXPECTATION,
        "message": message,
    }));
    0
}

fn selection_required(candidates: &[Candidate]) -> i32 {
    print_json(&json!({
        "condition": "selection_required",
        "message": "Multiple host sessions are available; pass --connection claude, codex, or another listed provider.",
        "candidates": candidate_envelope(candidates),
    }));
    2
}

fn discover_candidates() -> Vec<Candidate> {
    PROVIDERS
        .into_iter()
        .filter_map(|provider| match probe_provider(provider) {
            ProbeResult::Valid {
                account,
                account_exposed,
            } => Some(Candidate {
                provider: provider.to_string(),
                account,
                account_exposed,
            }),
            _ => None,
        })
        .collect()
}

fn auth_probe_args(provider: &str) -> Vec<Vec<&'static str>> {
    match provider {
        "codex" => vec![vec!["auth", "status"], vec!["login", "status"]],
        "claude" => vec![vec!["auth", "status"]],
        "agy" => vec![vec!["auth", "status"], vec!["login", "status"]],
        _ => vec![vec!["auth", "status"]],
    }
}

fn probe_provider(provider: &str) -> ProbeResult {
    let Some(binary) = find_binary(provider) else {
        return ProbeResult::Missing;
    };
    let mut saw_expired = false;
    for probe_args in auth_probe_args(provider) {
        let output = Command::new(&binary)
            .args(&probe_args)
            .env_remove("ANTHROPIC_API_KEY")
            .env_remove("OPENAI_API_KEY")
            .env_remove("CODEX_API_KEY")
            .stdin(Stdio::null())
            .output();
        let Ok(output) = output else {
            continue;
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}\n{stderr}");
        if auth_output_is_explicitly_not_drivable(&combined) {
            return ProbeResult::NotDrivable;
        }
        if auth_output_is_expired(&combined) {
            saw_expired = true;
            continue;
        }
        if output.status.success()
            && let Some(account) = parse_auth_account(&combined)
        {
            return ProbeResult::Valid {
                account: account.label,
                account_exposed: account.exposed,
            };
        }
    }
    if saw_expired {
        ProbeResult::Expired
    } else {
        ProbeResult::NotDrivable
    }
}

fn parse_auth_account(output: &str) -> Option<AuthAccount> {
    if auth_output_is_expired(output) {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(output) {
        if json_auth_succeeds(&value) {
            return json_account_field(&value)
                .map(|label| AuthAccount {
                    label,
                    exposed: true,
                })
                .or_else(|| Some(account_not_exposed()));
        }
        return None;
    }
    let mut saw_json_auth_line = false;
    for line in output.lines().rev() {
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            saw_json_auth_line = true;
            if json_auth_succeeds(&value) {
                return json_account_field(&value)
                    .map(|label| AuthAccount {
                        label,
                        exposed: true,
                    })
                    .or_else(|| Some(account_not_exposed()));
            }
        }
    }
    if saw_json_auth_line {
        return None;
    }
    let lower = output.to_ascii_lowercase();
    if lower.contains("logged in")
        || lower.contains("logged-in")
        || lower.contains("authenticated")
        || lower.contains("signed in")
    {
        return extract_email(output)
            .map(|label| AuthAccount {
                label,
                exposed: true,
            })
            .or_else(|| Some(account_not_exposed()));
    }
    None
}

fn json_auth_succeeds(value: &Value) -> bool {
    value.get("ok").and_then(Value::as_bool) == Some(true)
        || value.get("loggedIn").and_then(Value::as_bool) == Some(true)
        || value.get("logged_in").and_then(Value::as_bool) == Some(true)
        || value.get("authenticated").and_then(Value::as_bool) == Some(true)
        || value.get("status").and_then(Value::as_str) == Some("logged_in")
}

fn account_not_exposed() -> AuthAccount {
    AuthAccount {
        label: "account not exposed by host CLI".to_string(),
        exposed: false,
    }
}

fn json_account_field(value: &Value) -> Option<String> {
    value
        .get("account")
        .or_else(|| value.get("email"))
        .or_else(|| value.get("login"))
        .or_else(|| value.get("username"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            value
                .get("user")
                .and_then(|user| user.get("email").or_else(|| user.get("login")))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn extract_email(output: &str) -> Option<String> {
    output
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | ':' | ';'))
        .find(|token| token.contains('@') && token.contains('.'))
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '@' && ch != '.')
                .to_string()
        })
        .filter(|token| !token.is_empty())
}

fn auth_output_is_expired(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    if let Ok(value) = serde_json::from_str::<Value>(output)
        && json_auth_is_expired(&value)
    {
        return true;
    }
    if output.lines().any(|line| {
        serde_json::from_str::<Value>(line)
            .map(|value| json_auth_is_expired(&value))
            .unwrap_or(false)
    }) {
        return true;
    }
    lower.contains("expired")
        || lower.contains("invalid api key")
        || lower.contains("not logged in")
        || lower.contains("not logged-in")
        || lower.contains("not authenticated")
        || lower.contains("not_authenticated")
        || lower.contains("not signed in")
        || lower.contains("not signed-in")
        || lower.contains("unauthenticated")
        || lower.contains("signed out")
        || lower.contains("signed-out")
        || lower.contains("logged out")
        || lower.contains("logged-out")
        || lower.contains("login required")
        || lower.contains("log in required")
        || lower.contains("sign in required")
        || lower.contains("please log in")
        || lower.contains("unauthorized")
}

fn json_auth_is_expired(value: &Value) -> bool {
    value.get("ok").and_then(Value::as_bool) == Some(false)
        || value.get("loggedIn").and_then(Value::as_bool) == Some(false)
        || value.get("logged_in").and_then(Value::as_bool) == Some(false)
        || value.get("authenticated").and_then(Value::as_bool) == Some(false)
        || value
            .get("status")
            .and_then(Value::as_str)
            .map(|status| {
                matches!(
                    status,
                    "not_authenticated"
                        | "unauthenticated"
                        | "not_logged_in"
                        | "logged_out"
                        | "signed_out"
                )
            })
            .unwrap_or(false)
}

fn auth_output_is_explicitly_not_drivable(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    lower.contains("non_interactive_probe_failed")
}

fn candidate_envelope(candidates: &[Candidate]) -> Vec<Value> {
    let mut out = candidates
        .iter()
        .map(|candidate| {
            json!({
                "kind": "host-session",
                "provider": candidate.provider,
                "account": candidate.account,
                "spend_kind": "subscription quota",
            })
        })
        .collect::<Vec<_>>();
    out.push(json!({
        "kind": "local model",
        "label": "local model",
    }));
    out.push(json!({
        "kind": "env-var key",
        "label": "env-var key",
    }));
    out
}

fn same_host_session_connection(
    connection: &cg_catalog::ConnectionRead,
    candidate: &Candidate,
) -> bool {
    candidate.account_exposed
        && connection.consent_recorded
        && matches!(
            &connection.credential_reference.source,
            CredentialSource::HostSession {
                provider,
                account,
                account_exposed,
            } if *account_exposed
                && provider == &candidate.provider
                && account == &candidate.account
        )
}

fn selected_connection(catalog: &ModelConnectionCatalog) -> Option<cg_catalog::ConnectionRead> {
    catalog
        .active_connection_name()
        .and_then(|name| catalog.read_connection(&name))
        .or_else(|| {
            catalog
                .list_connections()
                .into_iter()
                .find(|connection| connection.consent_recorded)
        })
        .or_else(|| catalog.list_connections().into_iter().next())
}

fn handle_selftest(args: &[String]) -> i32 {
    let format = output_format(args);
    let root = data_root();
    let catalog = ModelConnectionCatalog::open(&root);
    if let Some(error) = catalog.load_error() {
        return unreadable_connections_file(error);
    }
    let Some(connection) = selected_connection(&catalog) else {
        return no_host_session_failure();
    };
    if !connection.consent_recorded {
        return failure(
            4,
            "consent_not_recorded",
            "auth",
            "Consent is not recorded; run cg model auth login --connection <provider> --consent before cg selftest.",
        );
    }
    let (provider, _account) = provider_account(&connection);
    let Some(binary) = find_binary(&provider) else {
        return failure(
            6,
            "host_tool_missing",
            "unknown",
            "The selected provider tool is gone; re-install it or run cg model auth login again.",
        );
    };
    let prompt = "cg host-session proof: answer with the deterministic proof token only.";
    let specs = load_provider_specs(&root, &[]);
    let spec = specs
        .providers
        .get(&provider)
        .or_else(|| specs.providers.get("agy"));
    let framing = spec
        .map(|spec| spec.reply_framing.as_str())
        .unwrap_or("text");
    let timeout_ms = spec.map(|spec| spec.timeout_ms).unwrap_or(10_000);
    let output_file_guard = if framing == "file" {
        Some(TempOutputFile::new(temp_output_path(&provider)))
    } else {
        None
    };
    let output_file = output_file_guard.as_ref().map(TempOutputFile::path);
    let run = run_provider_completion(
        &provider,
        &binary,
        prompt,
        output_file,
        timeout_ms,
        format == OutputFormat::Text,
    );
    if run.timed_out {
        return failure(
            6,
            "live_call_failed",
            "timeout",
            "The provider did not complete in time; cg did not retry the proof call.",
        );
    }
    let Some(status) = run.status else {
        return failure(
            6,
            "live_call_failed",
            "unknown",
            "provider call failed before a status was available",
        );
    };
    if !status.success() {
        let diagnostic = if run.stdout.trim().is_empty() {
            run.stderr.trim()
        } else {
            run.stdout.trim()
        };
        return classified_failure(&provider, diagnostic);
    }
    let reply = match extract_reply(&provider, framing, &run.stdout, output_file) {
        Ok(reply) if !reply.trim().is_empty() => reply,
        _ => {
            return failure(
                6,
                "live_call_failed",
                "format",
                "provider returned successfully but no proof reply was found",
            );
        }
    };
    if format == OutputFormat::Text {
        println!(
            "one live call through {provider} will spend subscription quota for connection {connection}\nPASS: {reply}\nReply shown and discarded. {INGESTION_EXPECTATION}",
            connection = connection.name
        );
    } else {
        print_json(&json!({
            "status": "pass",
            "connection": connection.name,
            "provider": provider,
            "reply": reply,
            "bucket": "ok",
            "message": format!("PASS: one live call through {provider} succeeded; reply shown and discarded. {INGESTION_EXPECTATION}"),
        }));
    }
    0
}

fn run_provider_completion(
    provider: &str,
    binary: &Path,
    prompt: &str,
    output_file: Option<&Path>,
    timeout_ms: u64,
    progress_to_stdout: bool,
) -> ProviderRun {
    let mut command = Command::new(binary);
    match provider {
        "claude" => {
            let invocation = construct_claude_invocation(prompt);
            command.args(invocation.argv);
        }
        "codex" => {
            command.arg("exec");
            if let Some(output_file) = output_file {
                command.arg(format!("--output-last-message={}", output_file.display()));
            }
        }
        "agy" => {
            command.arg("--print");
        }
        _ => {}
    }
    command
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let Ok(mut child) = command.spawn() else {
        return ProviderRun {
            status: None,
            stdout: String::new(),
            stderr: "provider spawn failed".to_string(),
            timed_out: false,
        };
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = stdout.map(read_stream);
    let stderr_reader = stderr.map(read_stream);
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(prompt.as_bytes());
    }
    thread::sleep(Duration::from_millis(100));
    eprintln!("waiting on provider {provider}");
    if progress_to_stdout {
        println!("waiting on provider {provider}");
        let _ = std::io::stdout().flush();
    }
    let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(500));
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    let _ = Command::new("pkill")
                        .arg("-TERM")
                        .arg("-P")
                        .arg(child.id().to_string())
                        .status();
                    let _ = child.kill();
                    break child.wait().ok();
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break None,
        }
    };
    let stdout = stdout_reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let stderr = stderr_reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    ProviderRun {
        status,
        stdout,
        stderr,
        timed_out,
    }
}

fn read_stream<R: Read + Send + 'static>(mut stream: R) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut out = String::new();
        let _ = stream.read_to_string(&mut out);
        out
    })
}

fn classified_failure(provider: &str, diagnostic: &str) -> i32 {
    match cg_cli::classify_provider_reply(provider, diagnostic) {
        Classification::Success => failure(
            6,
            "live_call_failed",
            "unknown",
            "provider failed without an actionable diagnostic",
        ),
        Classification::HostLoginExpired { .. } => failure(
            6,
            "host_login_expired",
            "auth",
            "Host session expired; re-login with your agent before running cg selftest.",
        ),
        Classification::BillingRequired { message } => failure(
            6,
            "live_call_failed",
            "billing",
            &format!("failed: {message}"),
        ),
        Classification::RateLimited { message } => failure(
            6,
            "live_call_failed",
            "rate_limit",
            &format!("failed: {message}"),
        ),
        Classification::LiveCallFailed { message } => failure(
            6,
            "live_call_failed",
            "unknown",
            &format!("failed: {message}"),
        ),
    }
}

fn handle_doctor(args: &[String]) -> i32 {
    let as_json = has_flag(args, "--json");
    let mut checks_run = vec!["model/claude/flags-in-help".to_string()];
    checks_run.sort();
    let mut findings = Vec::new();
    if let Some(binary) = find_binary("claude") {
        let help = Command::new(&binary)
            .arg("--help")
            .env_remove("ANTHROPIC_API_KEY")
            .output()
            .ok();
        let version = Command::new(&binary)
            .arg("--version")
            .env_remove("ANTHROPIC_API_KEY")
            .output()
            .ok();
        let help_text = help
            .as_ref()
            .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
            .unwrap_or_default();
        let version_text = version
            .as_ref()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown version".to_string());
        for required in ["--mcp-config", "--tools"] {
            if !help_text.contains(required) {
                findings.push(json!({
                    "check_id": "model/claude/flags-in-help",
                    "severity": "warning",
                    "message": format!("claude {version_text} help is missing required host-session flag {required}"),
                    "target": "claude",
                    "fix_hint": format!("upgrade claude so cg can pass {required} with equals-form invocation"),
                }));
            }
        }
    }
    let ok = findings.is_empty();
    if as_json {
        print_json(&json!({
            "ok": ok,
            "checks_run": checks_run,
            "findings": findings,
        }));
    } else if ok {
        println!("OK: model provider checks passed");
    } else {
        for finding in &findings {
            println!(
                "{}: {}",
                finding
                    .get("target")
                    .and_then(Value::as_str)
                    .unwrap_or("provider"),
                finding
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("provider drift")
            );
        }
    }
    if ok { 0 } else { 1 }
}

fn handle_disconnect(args: &[String]) -> i32 {
    let Some(agent) = args.first() else {
        return usage_error("missing agent for disconnect");
    };
    for path in agent_paths(agent, false) {
        clean_hooks(agent, &path);
    }
    0
}

fn handle_connect_cleanup(args: &[String]) -> i32 {
    let Some(agent) = args.first() else {
        return usage_error("missing agent for connect");
    };
    if args.get(1).map(String::as_str) == Some("--remove-all") {
        for path in agent_paths(agent, true) {
            clean_hooks(agent, &path);
        }
    }
    0
}

fn agent_paths(agent: &str, all: bool) -> Vec<PathBuf> {
    let home = home_dir();
    match agent {
        "claude-code" if all => {
            let mut out = Vec::new();
            collect_matching_files(
                &home,
                Path::new("agent-config/claude-code/settings.json"),
                &mut out,
            );
            out
        }
        "claude-code" => vec![home.join("agent-config/claude-code/settings.json")],
        "cursor" => vec![home.join(".cursor/hooks.json")],
        "gemini-cli" => vec![home.join(".gemini/settings.json")],
        _ => Vec::new(),
    }
}

fn collect_matching_files(root: &Path, suffix: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_matching_files(&path, suffix, out);
        } else if path.ends_with(suffix) {
            out.push(path);
        }
    }
}

fn clean_hooks(agent: &str, path: &Path) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let Ok(mut value) = serde_json::from_str::<Value>(&text) else {
        return;
    };
    let Some(hooks) = value.get_mut("hooks").and_then(Value::as_array_mut) else {
        return;
    };
    let begin = format!("# BEGIN cg-injected-hook-{agent}");
    let end = format!("# END cg-injected-hook-{agent}");
    let observe = format!("cg observe --agent {agent}");
    let mut out = Vec::new();
    let mut skipping = false;
    for hook in hooks.iter() {
        let text = hook.as_str().unwrap_or_default();
        if text.starts_with(&begin) {
            skipping = true;
            continue;
        }
        if skipping {
            if text.starts_with(&end) {
                skipping = false;
            }
            continue;
        }
        if text == observe {
            continue;
        }
        out.push(hook.clone());
    }
    *hooks = out;
    if let Ok(serialized) = serde_json::to_string_pretty(&value) {
        let _ = fs::write(path, format!("{serialized}\n"));
    }
}

fn no_host_session_failure() -> i32 {
    failure(
        4,
        "no_host_session",
        "auth",
        "No drivable host session was found. Use a local model first; env-var key setup is a later fallback.",
    )
}

fn unreadable_connections_file(error: &str) -> i32 {
    failure(
        1,
        "connections_file_unreadable",
        "unknown",
        &format!("model-connections.toml could not be read: {error}"),
    )
}

fn failure(code: i32, condition: &str, bucket: &str, message: &str) -> i32 {
    print_json(&json!({
        "condition": condition,
        "bucket": bucket,
        "message": message,
    }));
    code
}

fn provider_account(connection: &cg_catalog::ConnectionRead) -> (String, String) {
    match &connection.credential_reference.source {
        CredentialSource::HostSession {
            provider, account, ..
        } => (provider.clone(), account.clone()),
        CredentialSource::EnvVar { .. } | CredentialSource::StoredSecret { .. } => {
            (connection.name.clone(), connection.spend_owner.clone())
        }
    }
}

fn output_format(args: &[String]) -> OutputFormat {
    if flag_value(args, "--format").as_deref() == Some("text") {
        OutputFormat::Text
    } else {
        OutputFormat::Json
    }
}

fn has_log_info(args: &[String]) -> bool {
    flag_value(args, "--log")
        .as_deref()
        .is_some_and(|value| value.eq_ignore_ascii_case("info"))
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find_map(|window| {
        if window[0] == flag {
            Some(window[1].clone())
        } else {
            None
        }
    })
}

fn positional_arg(args: &[String]) -> Option<String> {
    args.iter()
        .find(|arg| !arg.starts_with('-'))
        .map(ToString::to_string)
}

fn print_json(value: &Value) {
    println!(
        "{}",
        serde_json::to_string(value).expect("JSON CLI envelope serializes")
    );
}

fn data_root() -> PathBuf {
    env::var_os("CG_DATA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".cg"))
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn find_binary(name: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(name);
    if candidate.components().count() > 1 && is_executable_file(&candidate) {
        return Some(candidate);
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|path| is_executable_file(path))
}

fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

fn temp_output_path(provider: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    env::temp_dir().join(format!(
        "cg-{provider}-last-message-{}-{nanos}.txt",
        std::process::id()
    ))
}

struct TempOutputFile {
    path: PathBuf,
}

impl TempOutputFile {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempOutputFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
