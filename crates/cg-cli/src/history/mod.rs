//! The event-history command boundary: `cg import`, `cg find`, `cg show`, `cg status` and
//! `cg forget`.
//!
//! This module parses arguments, resolves paths and the controlling terminal, opens files, calls
//! the context-graph library and the Claude Code connector, and renders their answers. Every rule
//! about events, bindings, projection, ranking and erasure lives in those libraries; nothing here
//! navigates Claude JSON, writes SQL, renders passages or ranks results.

mod forget;
mod import;
mod read;
mod worker;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::embedding_space::{BuiltinSpaceRegistry, event_history_embedding_space};
use context_graph::{
    CgError, ConsumerReader, ContextBindingKey, ContextId, EmbedderConfig, EmbedderRegistration,
    EventPlacementPolicy, EventSearchOrder, EventSearchProfile, EventSearchSessionScope,
    EventSearchWorkPolicy, EventSearchWorkSpec, StoreConfig, event_history_reader_declaration,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::OutputFormat;

pub(crate) use forget::handle_forget;
pub(crate) use import::handle_import;
pub(crate) use read::{handle_find, handle_show, handle_status};
#[cfg(feature = "event-search-test-support")]
pub(crate) use worker::apply_store_open_retry_allowance_for_test;
pub(crate) use worker::{WORKER_COMMAND, handle_event_search_worker};

/// Render a pasteable POSIX-shell import retry from separate argv tokens. The target can be an
/// operator's actual source path or the documented receipt placeholder; either way it is never
/// combined with flags before quoting.
pub(crate) fn import_retry_command(
    target: &str,
    context: Option<&str>,
    mode: Option<&str>,
    work_policy: EventSearchWorkPolicy,
    embedding_workers: Option<NonZeroUsize>,
) -> String {
    let sessions = match work_policy.sessions {
        EventSearchSessionScope::All => "all".to_string(),
        EventSearchSessionScope::Recent(limit) => limit.get().to_string(),
    };
    let order = match work_policy.order {
        EventSearchOrder::Newest => "newest",
        EventSearchOrder::Oldest => "oldest",
    };
    let mut tokens = vec![
        "cg".to_string(),
        "import".to_string(),
        posix_shell_quote(target),
    ];
    if let Some(context) = context {
        tokens.push("--context".to_string());
        tokens.push(posix_shell_quote(context));
    }
    if let Some(mode) = mode {
        tokens.push("--search".to_string());
        tokens.push(mode.to_string());
    }
    tokens.push("--search-sessions".to_string());
    tokens.push(sessions);
    tokens.push("--search-order".to_string());
    tokens.push(order.to_string());
    if let Some(workers) = embedding_workers {
        tokens.push("--embedding-workers".to_string());
        tokens.push(workers.get().to_string());
    }
    tokens.join(" ")
}

/// Quote one literal POSIX-shell token without evaluating or otherwise interpreting it.
fn posix_shell_quote(token: &str) -> String {
    if !token.is_empty()
        && token.bytes().all(|byte| {
            matches!(byte,
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'+' | b',' | b'-'
                    | b'.' | b'/' | b':' | b'=' | b'@'
            )
        })
    {
        return token.to_string();
    }
    format!("'{}'", token.replace('\'', "'\"'\"'"))
}

/// Canonical metadata identity; Nomic's document/query wrapper uses this ID for both the space
/// and encoding. Constructing the request does not load or inspect a model artifact.
pub(crate) fn search_work_spec(profile: &EventSearchProfile) -> EventSearchWorkSpec {
    EventSearchWorkSpec {
        profile: profile.clone(),
        embedding_space_id: BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
        encoding_profile_id: BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
    }
}

/// The versioned namespace of workspace-to-Context bindings. The reference beside it is an opaque
/// digest of the resolved project root; no path or basename is stored.
const WORKSPACE_BINDING_NAMESPACE: &str = "cg.workspace/v1";

/// The advisory per-data-root lock only event-search projection workers take.
const EVENT_SEARCH_WORKER_LOCK: &str = "event-search-worker.lock";

/// The exact `store_not_found` next step every read command prints.
const STORE_NOT_FOUND_NEXT: &str = "create it with the intended write command; for project history run `cg import <project-directory>`";

// ---------------------------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------------------------

/// One command line split into positionals, value flags and switches. Every token is accounted
/// for: an unknown flag, a missing value or a repeated flag is a usage error naming the token.
pub(crate) struct Parsed {
    pub positionals: Vec<String>,
    pub values: BTreeMap<&'static str, String>,
    pub switches: BTreeSet<&'static str>,
}

impl Parsed {
    pub fn value(&self, flag: &str) -> Option<&str> {
        self.values.get(flag).map(String::as_str)
    }

    pub fn has(&self, flag: &str) -> bool {
        self.switches.contains(flag)
    }
}

pub(crate) fn parse(
    args: &[String],
    value_flags: &[&'static str],
    switch_flags: &[&'static str],
) -> Result<Parsed, String> {
    let mut parsed = Parsed {
        positionals: Vec::new(),
        values: BTreeMap::new(),
        switches: BTreeSet::new(),
    };
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        index += 1;
        if token == "--" {
            parsed.positionals.extend(args[index..].iter().cloned());
            break;
        }
        if !token.starts_with("--") || token == "-" {
            parsed.positionals.push(token.clone());
            continue;
        }
        let (name, inline) = match token.split_once('=') {
            Some((name, value)) => (name, Some(value.to_string())),
            None => (token.as_str(), None),
        };
        if let Some(flag) = value_flags.iter().find(|flag| **flag == name) {
            let value = match inline {
                Some(value) => value,
                None => match args.get(index) {
                    Some(value) if !value.starts_with("--") => {
                        index += 1;
                        value.clone()
                    }
                    _ => return Err(format!("{flag} requires a value")),
                },
            };
            if parsed.values.insert(flag, value).is_some() {
                return Err(format!("{flag} was given more than once"));
            }
        } else if let Some(flag) = switch_flags.iter().find(|flag| **flag == name) {
            if inline.is_some() {
                return Err(format!("{flag} takes no value"));
            }
            parsed.switches.insert(flag);
        } else {
            return Err(format!("unknown option `{name}`"));
        }
    }
    Ok(parsed)
}

// ---------------------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------------------

pub(crate) fn print_json(value: &Value) {
    super::print_json(value);
}

/// A usage error: nothing was done. JSON callers get one object on stdout; everyone gets the
/// message on stderr.
pub(crate) fn usage(format: OutputFormat, command: &str, message: &str) -> i32 {
    if format == OutputFormat::Json {
        print_json(&json!({
            "ok": false,
            "state": "usage_error",
            "message": message,
            "usage": command_usage(command),
        }));
    }
    eprintln!("error: {message}\nUsage: {}", command_usage(command));
    2
}

pub(crate) fn command_usage(command: &str) -> &'static str {
    match command {
        "import" => {
            "cg import <project-directory>|<session.jsonl> [--context NAME] [--search background|wait|defer] [--embedding-workers N] [--search-order newest|oldest] [--search-sessions N|all] [--session REF] [--dry-run] [--format json|text]"
        }
        "find" => {
            "cg find <question> [--context NAME[,NAME...] | --all-contexts | --tenant-id TENANT] [--limit N] [--format json|text]"
        }
        "show" => {
            "cg show <passage-id> [--context NAME[,NAME...] | --all-contexts | --tenant-id TENANT] [--format json|text]"
        }
        "status" => {
            "cg status [--context NAME[,NAME...] | --all-contexts | --tenant-id TENANT] [--format json|text]"
        }
        "forget" => {
            "cg forget --session <ref> [--context NAME] [--yes] [--format json|text]\n       cg forget --context NAME [--yes] [--format json|text]"
        }
        _ => "cg <command>",
    }
}

/// A typed refusal. JSON callers get the whole object on stdout; text callers get the message and
/// the next action on stderr.
pub(crate) fn refuse(format: OutputFormat, code: i32, mut value: Value, text: &str) -> i32 {
    if format == OutputFormat::Json {
        if let Some(object) = value.as_object_mut() {
            object.entry("ok").or_insert(Value::Bool(false));
        }
        print_json(&value);
    } else {
        eprintln!("{text}");
    }
    code
}

/// The exact `store_not_found` contract shared by `status`, `find`, `show` and `forget`.
pub(crate) fn store_not_found(format: OutputFormat, db_path: &Path, with_next: bool) -> i32 {
    let db_path = db_path.display().to_string();
    if format == OutputFormat::Text {
        eprintln!("error: Context Graph store not found: {db_path}\nnext: {STORE_NOT_FOUND_NEXT}");
        return 1;
    }
    let mut value = json!({
        "ok": false,
        "state": "store_not_found",
        "store": {"db_path": db_path},
        "retry": "create_store_with_write_command",
    });
    if with_next {
        value["next"] = json!("cg import <project-directory>");
    }
    print_json(&value);
    1
}

/// Map a library refusal that is not command-specific to its CLI state and exit code.
pub(crate) fn library_refusal(format: OutputFormat, error: &CgError) -> i32 {
    let (code, state) = match error {
        CgError::StoreLocked { .. } => (4, "store_busy"),
        CgError::StoreHeldByReaders { .. } => (1, "store_held_by_readers"),
        CgError::ReaderCoordinationUnavailable { .. } => (1, "reader_coordination_unavailable"),
        CgError::ReadActiveTimeExceeded { .. } => (1, "read_active_time_limit"),
        CgError::LegacySchemaDetected { .. } => (1, "schema_mismatch"),
        CgError::StoreUnreadable { .. } => (1, "store_unreadable"),
        CgError::StoreNeedsWritableRecovery { .. } => (1, "store_needs_recovery"),
        CgError::WriteDidNotFitTheMemoryLimit { .. } => (1, "write_memory_limit"),
        CgError::TenantPolicyBindingRequired { .. } => (1, "tenant_policy_blocked"),
        CgError::EventPlacementPolicyMismatch { .. } => (1, "tenant_policy_blocked"),
        _ => (1, "error"),
    };
    let message = match (state, error) {
        ("store_busy", CgError::StoreLocked { holder_pid, path }) => format!(
            "store busy / already in use: {} is held for writing by {}; retry after it exits",
            path.display(),
            holder_pid.map_or_else(
                || "a process that published no id".to_string(),
                |pid| format!("process {pid}")
            )
        ),
        ("schema_mismatch", _) => format!(
            "{error}. The old database is untouched; point CG_DATA_ROOT at a new empty directory to start a fresh store"
        ),
        ("store_needs_recovery", _) => format!(
            "{error}. Run `cg import <project-directory> --search defer` to recover the store"
        ),
        (_, CgError::Engine(_)) => error.to_string(),
        _ => error.to_string(),
    };
    // The condition name is the CLI's settled vocabulary for a store refusal, carried in both
    // formats so a script can branch on it rather than on the exit code.
    let condition = match error {
        CgError::StoreNeedsWritableRecovery { .. } => "StoreNeedsWritableRecovery",
        CgError::StoreLocked { .. } => "StoreBusy",
        CgError::StoreHeldByReaders { .. } => "StoreHeldByReaders",
        CgError::ReaderCoordinationUnavailable { .. } => "ReaderCoordinationUnavailable",
        CgError::ReadActiveTimeExceeded { .. } => "ReadActiveTimeExceeded",
        CgError::WriteDidNotFitTheMemoryLimit { .. } => "WriteDidNotFitTheMemoryLimit",
        CgError::Engine(_)
        | CgError::StoreUnreadable { .. }
        | CgError::LegacySchemaDetected { .. } => "MemoryError",
        _ => "Error",
    };
    refuse(
        format,
        code,
        json!({"ok": false, "state": state, "condition": condition, "message": message}),
        &format!("error ({condition}): {message}"),
    )
}

// ---------------------------------------------------------------------------------------------
// Store, reader and model
// ---------------------------------------------------------------------------------------------

pub(crate) fn db_path(root: &Path) -> PathBuf {
    super::memory_db_path(root)
}

/// The Store configuration every event-history writer uses. `placement` is the optional expected
/// policy from `<data-root>/cg/fabric.toml`; `None` preserves an existing Store's persisted policy.
pub(crate) fn store_config(
    root: &Path,
    placement: Option<EventPlacementPolicy>,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> StoreConfig {
    StoreConfig {
        db_path: db_path(root),
        extra_embedding_spaces: vec![event_history_embedding_space()],
        event_placement: placement,
        read_memory_limit_bytes,
        read_active_time_ms: crate::read_config::active_time_in_force(),
        write_memory_limit_bytes: crate::write_config::in_force().bytes,
        ..StoreConfig::default()
    }
}

/// The optional `event_placement` value from the existing fabric configuration.
/// This machine's cg configuration file and its parsed contents, or `None` when there is none.
pub(crate) fn cg_config(root: &Path) -> Result<Option<(PathBuf, toml::Value)>, String> {
    let path = root
        .join("cg")
        .join(context_graph::fabric::FABRIC_CONFIG_FILE);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("reading {}: {error}", path.display())),
    };
    let value: toml::Value =
        toml::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(Some((path, value)))
}

pub(crate) fn resolved_embedding_workers(
    root: &Path,
    cli_value: Option<&str>,
) -> Result<std::num::NonZeroUsize, String> {
    let cli_value = cli_value
        .map(|value| parse_embedding_workers(value, "--embedding-workers"))
        .transpose()?;
    let configured = configured_embedding_workers(root)?;
    if let Some(value) = cli_value.or(configured) {
        return Ok(value);
    }
    available_embedding_workers()
}

pub(crate) fn resolved_search_work_policy(
    root: &Path,
    cli_order: Option<&str>,
    cli_sessions: Option<&str>,
) -> Result<EventSearchWorkPolicy, String> {
    let configured = cg_config(root)?;
    let search = configured
        .as_ref()
        .and_then(|(_, value)| value.get("search"));
    let config_path = configured.as_ref().map(|(path, _)| path);
    let configured_order = match search.and_then(|search| search.get("order")) {
        None => EventSearchOrder::Newest,
        Some(toml::Value::String(value)) => parse_search_order(value, "search.order")
            .map_err(|message| format!("{}: {message}", config_path.unwrap().display()))?,
        Some(value) => {
            return Err(format!(
                "{}: search.order must be newest or oldest, found {value}",
                config_path.unwrap().display()
            ));
        }
    };
    let order = cli_order
        .map(|value| parse_search_order(value, "--search-order"))
        .transpose()?
        .unwrap_or(configured_order);
    let configured_sessions = match search.and_then(|search| search.get("sessions")) {
        None => EventSearchSessionScope::Recent(
            std::num::NonZeroUsize::new(10).expect("default session bound is positive"),
        ),
        Some(toml::Value::String(value)) if value == "all" => EventSearchSessionScope::All,
        Some(toml::Value::Integer(value)) => usize::try_from(*value)
            .ok()
            .and_then(std::num::NonZeroUsize::new)
            .map(EventSearchSessionScope::Recent)
            .ok_or_else(|| {
                format!(
                    "{}: search.sessions must be a positive whole number or all, found {value}",
                    config_path.unwrap().display()
                )
            })?,
        Some(value) => {
            return Err(format!(
                "{}: search.sessions must be a positive whole number or all, found {value}",
                config_path.unwrap().display()
            ));
        }
    };
    let sessions = cli_sessions
        .map(|value| parse_search_sessions(value, "--search-sessions"))
        .transpose()?
        .unwrap_or(configured_sessions);
    Ok(EventSearchWorkPolicy { order, sessions })
}

pub(crate) fn parse_search_order(value: &str, source: &str) -> Result<EventSearchOrder, String> {
    match value {
        "newest" => Ok(EventSearchOrder::Newest),
        "oldest" => Ok(EventSearchOrder::Oldest),
        _ => Err(format!(
            "{source} must be newest or oldest, found `{value}`"
        )),
    }
}

pub(crate) fn parse_search_sessions(
    value: &str,
    source: &str,
) -> Result<EventSearchSessionScope, String> {
    if value == "all" {
        return Ok(EventSearchSessionScope::All);
    }
    value
        .parse::<usize>()
        .ok()
        .and_then(std::num::NonZeroUsize::new)
        .map(EventSearchSessionScope::Recent)
        .ok_or_else(|| format!("{source} must be a positive whole number or all, found `{value}`"))
}

fn configured_embedding_workers(root: &Path) -> Result<Option<std::num::NonZeroUsize>, String> {
    let Some((path, value)) = cg_config(root)? else {
        return Ok(None);
    };
    let Some(configured) = value
        .get("search")
        .and_then(|search| search.get("embedding_workers"))
    else {
        return Ok(None);
    };
    match configured {
        toml::Value::Integer(value) => usize::try_from(*value)
            .ok()
            .and_then(std::num::NonZeroUsize::new)
            .map(Some)
            .ok_or_else(|| {
                format!(
                    "{}: search.embedding_workers must be a positive whole number, found {value}",
                    path.display()
                )
            }),
        other => Err(format!(
            "{}: search.embedding_workers must be a positive whole number, found {other}",
            path.display()
        )),
    }
}

pub(crate) fn parse_embedding_workers(
    value: &str,
    source: &str,
) -> Result<std::num::NonZeroUsize, String> {
    value
        .parse::<usize>()
        .ok()
        .and_then(std::num::NonZeroUsize::new)
        .ok_or_else(|| format!("{source} must be a positive whole number, found `{value}`"))
}

fn available_embedding_workers() -> Result<std::num::NonZeroUsize, String> {
    std::thread::available_parallelism()
        .map_err(|error| format!("available CPU parallelism could not be determined: {error}"))
}

#[cfg(test)]
mod embedding_worker_policy_red {
    use super::{resolved_embedding_workers, resolved_search_work_policy};
    use context_graph::{EventSearchOrder, EventSearchSessionScope};

    fn write_config(root: &std::path::Path, body: &str) {
        let directory = root.join("cg");
        std::fs::create_dir_all(&directory).expect("create isolated config directory");
        std::fs::write(directory.join("fabric.toml"), body)
            .expect("write isolated worker-count config");
    }

    #[test]
    fn omitted_worker_policy_resolves_to_rust_available_parallelism() {
        let root = tempfile::tempdir().expect("isolated config root");
        assert_eq!(
            resolved_embedding_workers(root.path(), None).expect("resolve default worker count"),
            std::thread::available_parallelism().expect("host parallelism is available")
        );
    }

    #[test]
    fn configured_worker_policy_is_used_when_the_cli_is_silent() {
        let root = tempfile::tempdir().expect("isolated config root");
        let mut observed = Vec::new();
        for cap in [1, 2] {
            write_config(
                root.path(),
                &format!("[search]\nembedding_workers = {cap}\n"),
            );
            observed.push(
                resolved_embedding_workers(root.path(), None)
                    .expect("configured positive cap resolves")
                    .get(),
            );
        }
        assert_eq!(observed, [1, 2], "configured caps were not delivered");
    }

    #[test]
    fn explicit_cli_worker_policy_wins_over_configuration_in_both_directions() {
        let root = tempfile::tempdir().expect("isolated config root");
        let mut observed = Vec::new();
        for (configured, cli) in [(1, "2"), (2, "1")] {
            write_config(
                root.path(),
                &format!("[search]\nembedding_workers = {configured}\n"),
            );
            observed.push(
                resolved_embedding_workers(root.path(), Some(cli))
                    .expect("positive CLI cap resolves")
                    .get(),
            );
        }
        assert_eq!(observed, [2, 1], "CLI caps did not override configuration");
    }

    #[test]
    fn resolver_refuses_zero_negative_and_non_number_values_from_each_input() {
        let root = tempfile::tempdir().expect("isolated config root");
        let cli_refusals = ["0", "-1", "many"]
            .map(|value| resolved_embedding_workers(root.path(), Some(value)).is_err());
        let mut config_refusals = Vec::new();
        for value in ["0", "-1", "\"many\""] {
            write_config(
                root.path(),
                &format!("[search]\nembedding_workers = {value}\n"),
            );
            config_refusals.push(resolved_embedding_workers(root.path(), None).is_err());
        }
        assert_eq!(cli_refusals, [true; 3], "invalid CLI caps were accepted");
        assert_eq!(
            config_refusals, [true; 3],
            "invalid configured caps were accepted"
        );
    }

    #[test]
    fn search_work_policy_defaults_to_newest_ten_and_cli_overrides_config() {
        let root = tempfile::tempdir().expect("isolated config root");
        let defaults = resolved_search_work_policy(root.path(), None, None)
            .expect("resolve default search policy");
        assert_eq!(defaults.order, EventSearchOrder::Newest);
        assert_eq!(
            defaults.sessions,
            EventSearchSessionScope::Recent(std::num::NonZeroUsize::new(10).unwrap())
        );

        write_config(root.path(), "[search]\norder = \"oldest\"\nsessions = 3\n");
        let configured = resolved_search_work_policy(root.path(), None, None)
            .expect("resolve configured search policy");
        assert_eq!(configured.order, EventSearchOrder::Oldest);
        assert_eq!(
            configured.sessions,
            EventSearchSessionScope::Recent(std::num::NonZeroUsize::new(3).unwrap())
        );
        let overridden = resolved_search_work_policy(root.path(), Some("newest"), Some("all"))
            .expect("CLI policy overrides configuration");
        assert_eq!(overridden.order, EventSearchOrder::Newest);
        assert_eq!(overridden.sessions, EventSearchSessionScope::All);
    }

    #[test]
    fn invalid_search_config_is_refused_even_when_cli_values_are_present() {
        let root = tempfile::tempdir().expect("isolated config root");
        for body in [
            "[search]\norder = \"sideways\"\nsessions = 10\n",
            "[search]\norder = \"newest\"\nsessions = 0\n",
            "[search]\norder = \"newest\"\nsessions = \"many\"\n",
        ] {
            write_config(root.path(), body);
            assert!(
                resolved_search_work_policy(root.path(), Some("oldest"), Some("all")).is_err(),
                "invalid configuration reached the write path: {body}"
            );
        }
    }
}

/// The passage profile new search generations use: the library default, with `target_tokens` and
/// `hard_tokens` overridden by an optional `[search.passages]` section of the cg configuration
/// file. An override must be positive integers with the hard limit at least the target.
pub(crate) fn configured_passage_profile(root: &Path) -> Result<EventSearchProfile, String> {
    let mut profile = EventSearchProfile::default();
    let capability =
        context_graph::embedder::nomic::NomicTextEmbedder::declared_event_search_capability();
    let Some((path, value)) = cg_config(root)? else {
        return profile
            .validate_for(capability)
            .map(|()| profile)
            .map_err(|error| format!("default event-search passage profile: {error}"));
    };
    let Some(passages) = value
        .get("search")
        .and_then(|search| search.get("passages"))
    else {
        return profile
            .validate_for(capability)
            .map(|()| profile)
            .map_err(|error| format!("{}: {error}", path.display()));
    };
    let tokens = |key: &str| -> Result<Option<std::num::NonZeroU32>, String> {
        match passages.get(key) {
            None => Ok(None),
            Some(toml::Value::Integer(value)) => u32::try_from(*value)
                .ok()
                .and_then(std::num::NonZeroU32::new)
                .map(Some)
                .ok_or_else(|| {
                    format!(
                        "{}: search.passages.{key} must be a positive whole number of tokens, found {value}",
                        path.display()
                    )
                }),
            Some(other) => Err(format!(
                "{}: search.passages.{key} must be a positive whole number of tokens, found {other}",
                path.display()
            )),
        }
    };
    if let Some(target) = tokens("target_tokens")? {
        profile.target_passage_tokens = target;
    }
    if let Some(hard) = tokens("hard_tokens")? {
        profile.hard_passage_tokens = hard;
    }
    profile.validate_for(capability).map_err(|error| match error {
        context_graph::CgError::EventSearchProfileInvalid { .. } => format!(
            "{}: search.passages.hard_tokens ({}) must be at least target_tokens ({})",
            path.display(),
            profile.hard_passage_tokens,
            profile.target_passage_tokens
        ),
        context_graph::CgError::EventSearchOverlapInvalid { .. } => format!(
            "{}: search.passages.hard_tokens ({}) must exceed the oversized-event overlap ({})",
            path.display(),
            profile.hard_passage_tokens,
            profile.oversized_event_overlap_tokens
        ),
        context_graph::CgError::EventSearchProfileExceedsCapability {
            requested_hard_passage_tokens,
            maximum_prepared_tokens,
        } => format!(
            "{}: search.passages.hard_tokens ({requested_hard_passage_tokens}) exceeds the bundled event-history model's declared prepared-token maximum ({maximum_prepared_tokens})",
            path.display()
        ),
        _ => format!("{}: {error}", path.display()),
    })?;
    Ok(profile)
}

pub(crate) fn configured_placement(root: &Path) -> Result<Option<EventPlacementPolicy>, String> {
    let Some((path, value)) = cg_config(root)? else {
        return Ok(None);
    };
    let section = value.get("fabric").unwrap_or(&value);
    match section.get("event_placement") {
        None => Ok(None),
        Some(toml::Value::String(policy)) if policy == "local_only" => {
            Ok(Some(EventPlacementPolicy::LocalOnly))
        }
        Some(toml::Value::String(policy)) if policy == "tenant_hub" => {
            Ok(Some(EventPlacementPolicy::TenantHub))
        }
        Some(other) => Err(format!(
            "{}: event_placement must be \"local_only\" or \"tenant_hub\", found {other}",
            path.display()
        )),
    }
}

pub(crate) fn open_reader(
    root: &Path,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> Result<ConsumerReader, CgError> {
    ConsumerReader::open_with_read_limits(
        db_path(root),
        event_history_reader_declaration(),
        read_memory_limit_bytes,
        crate::read_config::active_time_in_force(),
    )
}

fn nomic_model() -> Arc<NomicTextEmbedder> {
    Arc::new(NomicTextEmbedder::new(
        EmbedderConfig::default_nomic_offline(),
    ))
}

/// Whether the pinned local event-search artifact is installed. A non-network presence check; it
/// never loads or downloads weights.
pub(crate) fn search_artifact_available() -> bool {
    EventHistoryTextEmbedder::over(nomic_model()).is_ok()
}

/// The offline, deny-download event-search embedder registration.
pub(crate) fn search_registration() -> Result<EmbedderRegistration, CgError> {
    EventHistoryTextEmbedder::registration(nomic_model())
}

// ---------------------------------------------------------------------------------------------
// Workspace bindings and Context selection
// ---------------------------------------------------------------------------------------------

/// The opaque binding key for one resolved project root.
pub(crate) fn workspace_binding_key(resolved_root: &Path) -> ContextBindingKey {
    let mut digest = Sha256::new();
    digest.update(resolved_root.as_os_str().as_encoded_bytes());
    ContextBindingKey {
        namespace: WORKSPACE_BINDING_NAMESPACE.to_string(),
        opaque_reference: format!("{:x}", digest.finalize()),
    }
}

/// The nearest bound directory for `start` (itself, then each parent) and its Context.
pub(crate) fn nearest_binding(
    reader: &ConsumerReader,
    start: &Path,
) -> Result<Option<(ContextId, PathBuf)>, CgError> {
    let Ok(resolved) = fs::canonicalize(start) else {
        return Ok(None);
    };
    for directory in resolved.ancestors() {
        if let Some(context_id) =
            reader.resolve_context_binding(&workspace_binding_key(directory))?
        {
            return Ok(Some((context_id, directory.to_path_buf())));
        }
    }
    Ok(None)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectionSource {
    ProjectBinding,
    Explicit,
    AllAuthorized,
    TenantDefault,
    AnonymousDefault,
}

impl SelectionSource {
    pub fn as_str(self) -> &'static str {
        match self {
            SelectionSource::ProjectBinding => "project_binding",
            SelectionSource::Explicit => "explicit",
            SelectionSource::AllAuthorized => "all_authorized",
            SelectionSource::TenantDefault => "tenant_default",
            SelectionSource::AnonymousDefault => "anonymous_default",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SelectionSource::ProjectBinding => "project binding",
            SelectionSource::Explicit => "explicit",
            SelectionSource::AllAuthorized => "all authorized contexts",
            SelectionSource::TenantDefault => "tenant default",
            SelectionSource::AnonymousDefault => "default context",
        }
    }
}

pub(crate) struct Selection {
    pub source: SelectionSource,
    pub contexts: Vec<(ContextId, String)>,
    /// The bound project directory when the selection came from a workspace binding.
    pub bound_directory: Option<PathBuf>,
    /// The fallback Context name that was looked for but does not exist.
    pub missing_default: Option<String>,
}

impl Selection {
    pub fn json(&self) -> Value {
        let mut value = json!({
            "source": self.source.as_str(),
            "contexts": self
                .contexts
                .iter()
                .map(|(id, name)| json!({"id": id.to_string(), "name": name}))
                .collect::<Vec<_>>(),
        });
        if let Some(name) = &self.missing_default {
            value["missing_default_context"] = json!(name);
        }
        value
    }

    pub fn name_of(&self, id: ContextId) -> String {
        self.contexts
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, name)| name.clone())
            .unwrap_or_else(|| id.to_string())
    }
}

pub(crate) enum SelectionError {
    Usage(String),
    UnknownContext(Vec<String>),
    Library(CgError),
}

/// Split and validate `--context a,b`: trimmed, deduplicated, never empty.
pub(crate) fn context_names(raw: &str) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    for item in raw.split(',') {
        let name = item.trim();
        if name.is_empty() {
            return Err(format!("--context `{raw}` contains an empty Context name"));
        }
        if !names.iter().any(|existing: &String| existing == name) {
            names.push(name.to_string());
        }
    }
    Ok(names)
}

/// Resolve the read selection shared by `find`, `show`, `status` and `forget`.
pub(crate) fn resolve_selection(
    reader: &ConsumerReader,
    root: &Path,
    explicit: Option<&str>,
    all_contexts: bool,
    tenant: Option<&str>,
) -> Result<Selection, SelectionError> {
    if explicit.is_some() && all_contexts {
        return Err(SelectionError::Usage(
            "--context and --all-contexts are mutually exclusive".to_string(),
        ));
    }
    let visible = reader
        .graph()
        .list_contexts()
        .map_err(SelectionError::Library)?;
    if let Some(raw) = explicit {
        let names = context_names(raw).map_err(SelectionError::Usage)?;
        let mut contexts = Vec::new();
        let mut unknown = Vec::new();
        for name in names {
            match visible.iter().find(|context| context.name == name) {
                Some(context) => contexts.push((context.id, context.name.clone())),
                None => unknown.push(name),
            }
        }
        if !unknown.is_empty() {
            return Err(SelectionError::UnknownContext(unknown));
        }
        return Ok(Selection {
            source: SelectionSource::Explicit,
            contexts,
            bound_directory: None,
            missing_default: None,
        });
    }
    if all_contexts {
        return Ok(Selection {
            source: SelectionSource::AllAuthorized,
            contexts: visible
                .iter()
                .map(|context| (context.id, context.name.clone()))
                .collect(),
            bound_directory: None,
            missing_default: None,
        });
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if let Some((context_id, directory)) =
        nearest_binding(reader, &cwd).map_err(SelectionError::Library)?
    {
        let name = visible
            .iter()
            .find(|context| context.id == context_id)
            .map(|context| context.name.clone())
            .ok_or_else(|| {
                SelectionError::Library(CgError::NotFound {
                    id: Some(context_id.to_string()),
                    missing: Some("context"),
                })
            })?;
        return Ok(Selection {
            source: SelectionSource::ProjectBinding,
            contexts: vec![(context_id, name)],
            bound_directory: Some(directory),
            missing_default: None,
        });
    }
    // The tenant-named default: `--tenant-id` names it for this invocation, else the configured
    // fabric tenant.
    let tenant = tenant
        .map(str::to_string)
        .or_else(|| {
            context_graph::fabric::FabricConfig::load(root)
                .ok()
                .flatten()
                .and_then(|config| config.tenant)
        })
        .filter(|tenant| !tenant.trim().is_empty());
    let (source, name) = match tenant {
        Some(tenant) => (SelectionSource::TenantDefault, tenant),
        None => (SelectionSource::AnonymousDefault, "default".to_string()),
    };
    match visible.iter().find(|context| context.name == name) {
        Some(context) => Ok(Selection {
            source,
            contexts: vec![(context.id, context.name.clone())],
            bound_directory: None,
            missing_default: None,
        }),
        None => Ok(Selection {
            source,
            contexts: Vec::new(),
            bound_directory: None,
            missing_default: Some(name),
        }),
    }
}

pub(crate) fn selection_refusal(format: OutputFormat, command: &str, error: SelectionError) -> i32 {
    match error {
        SelectionError::Usage(message) => usage(format, command, &message),
        SelectionError::UnknownContext(names) => {
            let message = format!(
                "unknown Context {}; nothing was read. Run `cg status --all-contexts` to list the Contexts this store holds",
                names
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            refuse(
                format,
                2,
                json!({"ok": false, "state": "unknown_context", "unknown_contexts": names, "message": message}),
                &format!("error: {message}"),
            )
        }
        SelectionError::Library(error) => library_refusal(format, &error),
    }
}

// ---------------------------------------------------------------------------------------------
// The controlling terminal
// ---------------------------------------------------------------------------------------------

/// The controlling terminal, when this process has one. Prompts go here and only here: never to
/// stdout, and never in non-interactive execution.
pub(crate) struct Terminal {
    reader: BufReader<fs::File>,
    writer: fs::File,
}

impl Terminal {
    pub fn open() -> Option<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/tty")
            .ok()?;
        let writer = file.try_clone().ok()?;
        Some(Self {
            reader: BufReader::new(file),
            writer,
        })
    }

    pub fn say(&mut self, line: &str) {
        let _ = writeln!(self.writer, "{line}");
    }

    /// Write `prompt` and read one answer line, trimmed. End of input is an empty answer.
    pub fn ask(&mut self, prompt: &str) -> String {
        let _ = write!(self.writer, "{prompt}");
        let _ = self.writer.flush();
        let mut answer = String::new();
        let _ = self.reader.read_line(&mut answer);
        answer.trim().to_string()
    }

    /// A `[y/N]` question: only an explicit yes is yes.
    pub fn confirm(&mut self, prompt: &str) -> bool {
        matches!(self.ask(prompt).to_ascii_lowercase().as_str(), "y" | "yes")
    }
}

/// Render a count with thousands separators, as the human outputs do.
pub(crate) fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(character);
    }
    out
}
