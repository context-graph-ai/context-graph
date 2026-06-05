use std::env;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use cg_catalog::{
    Catalog, CredentialSource, HostSessionProbe, ModelConnectionCatalog, ModelProofError,
    PrincipalAwareConfig, ProviderSpecLoadError, Source, UnderstandingRuntime, load_provider_specs,
    model_connection_system_principal, probe_host_session_provider, provider_tool_available,
    register_model_connection_checks_with_root, try_load_provider_specs,
};
use cg_cli::{CheckRegistry, run_doctor_checks};
use serde::Deserialize;
use serde_json::{Value, json};

const INGESTION_EXPECTATION: &str = "automatic capture switches on when ingestion ships";
const CONNECTION_SCOPE: &str = "connection-substrate";

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
    if let Some(message) = format_usage_error(&args) {
        return usage_error(&message);
    }
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
        "Usage: cg <command>\n\nCommands:\n  model auth login              Connect cg to an existing model host session\n  model list                    List registered models\n  model show <id>               Show model metadata\n  status                        Show model connection status\n  selftest                      Spend one consented proof call\n  doctor                        Run free model-provider diagnostics\n  disconnect <agent>            Remove cg hooks for one agent config\n  connect <agent> --remove-all  Remove cg hooks for one agent across configs\n\nExamples:\n  cg model auth login --connection claude --consent\n  cg status\n  cg selftest"
    );
}

fn usage_error(message: &str) -> i32 {
    eprintln!("error: {message}\nUsage: cg <command>");
    2
}

fn handle_model(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") => {
            println!(
                "Usage: cg model <command>\n\nCommands:\n  auth login    Connect cg to an existing model host session\n  list          List registered models\n  show <id>     Show model metadata\n\nExamples:\n  cg model auth login --connection claude --consent\n  cg model list\n  cg model show embeddinggemma-300m"
            );
            0
        }
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
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!(
            "Usage: cg model list [--format json|text]\n\nExamples:\n  cg model list\n  cg model list --format text"
        );
        return 0;
    }
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
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!(
            "Usage: cg model show <id> [--format json|text]\n\nExamples:\n  cg model show embeddinggemma-300m\n  cg model show nomic-embed-text-v1.5 --format text"
        );
        return 0;
    }
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
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!(
            "Usage: cg status [--format json|text]\n\nExamples:\n  cg status\n  cg status --format text"
        );
        return 0;
    }
    let format = output_format(args);
    let root = data_root();
    if let Err(err) = try_load_provider_specs(&root, &[]) {
        return provider_spec_failure(&err);
    }
    let catalog = ModelConnectionCatalog::open(&root);
    if let Some(error) = catalog.load_error() {
        return unreadable_connections_file(error);
    }
    if let Some(active) = catalog.active_connection_name()
        && let Some(error) = catalog.unreadable_connection_error(&active)
    {
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
    let tool_condition = if provider_tool_available(&root, &provider) {
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
            "Usage: cg model auth login [--connection PROVIDER] [--consent] [--format json|text]\n\nConnect cg to an already logged-in agent CLI host session.\n\nExamples:\n  cg model auth login\n  cg model auth login --connection claude --consent"
        );
        return 0;
    }

    let selected = flag_value(args, "--connection");
    let consent_flag = has_flag(args, "--consent");
    let root = data_root();
    if let Err(err) = try_load_provider_specs(&root, &[]) {
        return provider_spec_failure(&err);
    }
    let candidates = discover_candidates(&root);

    if candidates.is_empty() {
        let unsafe_settings =
            provider_names(&root).into_iter().find_map(
                |provider| match probe_host_session_provider(&root, &provider) {
                    HostSessionProbe::UnsafeSettings { message } => Some(message),
                    _ => None,
                },
            );
        if let Some(message) = unsafe_settings {
            return failure(6, "live_call_failed", "billing", &message);
        }
        let expired = provider_names(&root).into_iter().any(|provider| {
            matches!(
                probe_host_session_provider(&root, &provider),
                HostSessionProbe::Expired
            )
        });
        if expired {
            return failure(
                6,
                "host_login_expired",
                "auth",
                "Host login expired; re-login with your agent before connecting cg.",
            );
        }
        if let Some((provider, reason)) =
            provider_names(&root).into_iter().find_map(
                |provider| match probe_host_session_provider(&root, &provider) {
                    HostSessionProbe::NotDrivable { reason } => Some((provider, reason)),
                    _ => None,
                },
            )
        {
            return no_host_session_failure_with_reason(&provider, &reason);
        }
        return no_host_session_failure();
    }

    if let Some(provider) = selected {
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.provider == provider)
            .cloned()
        else {
            let result = probe_host_session_provider(&root, &provider);
            return match result {
                HostSessionProbe::Expired => failure(
                    6,
                    "host_login_expired",
                    "auth",
                    "Host login expired; re-login with your agent before connecting cg.",
                ),
                HostSessionProbe::UnsafeSettings { message } => {
                    failure(6, "live_call_failed", "billing", &message)
                }
                HostSessionProbe::NotDrivable { reason } => {
                    no_host_session_failure_with_reason(&provider, &reason)
                }
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
    render_consent_prompt(&candidates);
    let mut stdin_text = String::new();
    let _ = std::io::stdin().lock().read_line(&mut stdin_text);
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
    render_selection_prompt(candidates);
    print_json(&json!({
        "condition": "selection_required",
        "message": "Multiple host sessions are available; pass --connection claude, codex, or another listed provider.",
        "candidates": candidate_envelope(candidates),
    }));
    2
}

fn render_selection_prompt(candidates: &[Candidate]) {
    eprintln!("Multiple host sessions are available:");
    for candidate in candidates {
        eprintln!("- {} ({})", candidate.provider, candidate.account);
    }
    eprintln!("Run cg model auth login --connection <provider> --consent to connect one.");
    let _ = std::io::stderr().flush();
}

fn render_consent_prompt(candidates: &[Candidate]) {
    eprintln!("Available host-session candidate:");
    for candidate in candidates {
        eprintln!("- {} ({})", candidate.provider, candidate.account);
    }
    eprintln!(
        "Spend uses subscription quota. Type yes to connect with consent, or no to record the reference without consent."
    );
    let _ = std::io::stderr().flush();
}

fn discover_candidates(root: &Path) -> Vec<Candidate> {
    provider_names(root)
        .into_iter()
        .filter_map(
            |provider| match probe_host_session_provider(root, &provider) {
                HostSessionProbe::Valid {
                    account,
                    account_exposed,
                } => Some(Candidate {
                    provider,
                    account,
                    account_exposed,
                }),
                _ => None,
            },
        )
        .collect()
}

fn provider_names(root: &Path) -> Vec<String> {
    load_provider_specs(root, &[])
        .providers
        .keys()
        .cloned()
        .collect()
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
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!(
            "Usage: cg selftest [--format json|text]\n\nExamples:\n  cg selftest\n  cg selftest --format text"
        );
        return 0;
    }
    let format = output_format(args);
    let root = data_root();
    if let Err(err) = try_load_provider_specs(&root, &[]) {
        return provider_spec_failure(&err);
    }
    let catalog = ModelConnectionCatalog::open(&root);
    if let Some(error) = catalog.load_error() {
        return unreadable_connections_file(error);
    }
    if let Some(active) = catalog.active_connection_name()
        && let Some(error) = catalog.unreadable_connection_error(&active)
    {
        return unreadable_connections_file(error);
    }
    let Some(connection) = selected_connection(&catalog) else {
        return no_host_session_failure();
    };
    let prompt = "cg host-session proof: answer with the deterministic proof token only.";
    let connection_name = connection.name.clone();
    let mut runtime = UnderstandingRuntime::<cg_catalog::NoopLlmClient>::new(
        catalog,
        model_connection_system_principal(),
        PrincipalAwareConfig::default(),
    );
    let proof = match runtime.prove_host_session_connection(
        &connection_name,
        prompt,
        format == OutputFormat::Text,
    ) {
        Ok(proof) => proof,
        Err(err) => return proof_failure(err),
    };
    if format == OutputFormat::Text {
        println!(
            "one live call through {provider} will spend subscription quota for connection {connection}\nPASS: {reply}\nReply shown and discarded. {INGESTION_EXPECTATION}",
            provider = proof.provider,
            connection = proof.connection_name,
            reply = proof.reply
        );
    } else {
        print_json(&json!({
            "status": "pass",
            "connection": proof.connection_name,
            "provider": proof.provider,
            "reply": proof.reply,
            "bucket": "ok",
            "message": format!("PASS: one live call through {} succeeded; reply shown and discarded. {INGESTION_EXPECTATION}", proof.provider),
        }));
    }
    0
}

fn handle_doctor(args: &[String]) -> i32 {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!("Usage: cg doctor [--json]\n\nExamples:\n  cg doctor\n  cg doctor --json");
        return 0;
    }
    let as_json = has_flag(args, "--json");
    let root = data_root();
    if let Err(err) = try_load_provider_specs(&root, &[]) {
        return provider_spec_failure(&err);
    }
    let mut registry = CheckRegistry::default();
    register_model_connection_checks_with_root(&mut registry, root);
    let report = run_doctor_checks(&registry);
    if as_json {
        let value = serde_json::to_value(&report).expect("doctor report serializes");
        print_json(&value);
    } else {
        print_doctor_text(&report);
    }
    if report.ok { 0 } else { 1 }
}

fn print_doctor_text(report: &cg_cli::DoctorReport) {
    for check_id in &report.checks_run {
        let mut emitted = false;
        for finding in report
            .findings
            .iter()
            .filter(|finding| &finding.check_id == check_id)
        {
            emitted = true;
            let severity = format!("{:?}", finding.severity).to_ascii_uppercase();
            println!(
                "{severity}: {check_id}: {}: {}",
                finding.target.as_deref().unwrap_or("provider"),
                finding.message
            );
        }
        if !emitted {
            println!("OK: {check_id}");
        }
    }
}

fn handle_disconnect(args: &[String]) -> i32 {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!(
            "Usage: cg disconnect <agent>\n\nExamples:\n  cg disconnect claude-code\n  cg disconnect cursor"
        );
        return 0;
    }
    let Some(agent) = args.first() else {
        return usage_error("missing agent for disconnect");
    };
    for path in agent_paths(agent, false) {
        clean_hooks(agent, &path);
    }
    0
}

fn handle_connect_cleanup(args: &[String]) -> i32 {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        println!(
            "Usage: cg connect <agent> --remove-all\n\nExamples:\n  cg connect claude-code --remove-all"
        );
        return 0;
    }
    let Some(agent) = args.first() else {
        return usage_error("missing agent for connect");
    };
    if args.get(1).map(String::as_str) != Some("--remove-all") {
        return usage_error("cg connect cleanup requires --remove-all");
    }
    for path in agent_paths(agent, true) {
        clean_hooks(agent, &path);
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

fn no_host_session_failure_with_reason(provider: &str, reason: &str) -> i32 {
    failure(
        4,
        "no_host_session",
        "auth",
        &format!(
            "No drivable host session was found for {provider}: {reason}. Use a local model first; env-var key setup is a later fallback."
        ),
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

fn provider_spec_failure(error: &ProviderSpecLoadError) -> i32 {
    print_json(&json!({
        "condition": "provider_spec_invalid",
        "bucket": "config",
        "provider": error.provider,
        "field": error.field,
        "value": error.value,
        "accepted": error.accepted,
        "path": error.path.as_ref().map(|path| path.display().to_string()),
        "message": error.message(),
    }));
    1
}

fn failure(code: i32, condition: &str, bucket: &str, message: &str) -> i32 {
    print_json(&json!({
        "condition": condition,
        "bucket": bucket,
        "message": message,
    }));
    code
}

fn proof_failure(error: ModelProofError) -> i32 {
    let code = match &error {
        ModelProofError::NoHostSession | ModelProofError::ConsentNotRecorded => 4,
        ModelProofError::HostToolMissing { .. }
        | ModelProofError::HostLoginExpired { .. }
        | ModelProofError::LiveCallFailed { .. }
        | ModelProofError::ReplyFormat { .. }
        | ModelProofError::UnsafeSettings { .. } => 6,
    };
    failure(code, error.condition(), error.bucket(), &error.message())
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

fn format_usage_error(args: &[String]) -> Option<String> {
    for (index, arg) in args.iter().enumerate() {
        if let Some(value) = arg.strip_prefix("--format=")
            && value != "json"
            && value != "text"
        {
            return Some(format!(
                "invalid --format value `{value}`; expected `json` or `text`"
            ));
        }
        if arg == "--format" {
            let Some(value) = args.get(index + 1) else {
                return Some("--format requires `json` or `text`".to_string());
            };
            if value != "json" && value != "text" {
                return Some(format!(
                    "invalid --format value `{value}`; expected `json` or `text`"
                ));
            }
        }
    }
    None
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
    let prefix = format!("{flag}=");
    args.iter()
        .find_map(|arg| arg.strip_prefix(&prefix).map(ToString::to_string))
        .or_else(|| {
            args.windows(2).find_map(|window| {
                if window[0] == flag {
                    Some(window[1].clone())
                } else {
                    None
                }
            })
        })
}

fn positional_arg(args: &[String]) -> Option<String> {
    let mut skip_next = false;
    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if matches!(arg.as_str(), "--format" | "--connection" | "--log") {
            skip_next = true;
            continue;
        }
        if arg.starts_with("--format=")
            || arg.starts_with("--connection=")
            || arg.starts_with("--log=")
            || arg.starts_with('-')
        {
            continue;
        }
        return Some(arg.clone());
    }
    None
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
