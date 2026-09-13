//! `cg import <project-directory>|<session.jsonl>`.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cg_connector_claude_code::{
    ClaudeCodeJsonlAdapter, ClaudeHistoryLocator, effective_history_root,
};
use context_graph::{
    CgError, ContextId, ContextSelector, CreateContext, EventCoverageCounts,
    EventHistoryContextStatus, EventPlacementState, EventSourcePrefixCheck, ImportEventOptions,
    Store,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    OutputFormat, SelectionSource, Terminal, configured_placement, grouped, library_refusal,
    nearest_binding, open_reader, parse, print_json, refuse, search_artifact_available,
    store_config, usage, worker, workspace_binding_key,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Background,
    Wait,
    Defer,
}

impl SearchMode {
    fn as_str(self) -> &'static str {
        match self {
            SearchMode::Background => "background",
            SearchMode::Wait => "wait",
            SearchMode::Defer => "defer",
        }
    }
}

/// One selected source file.
struct Source {
    path: PathBuf,
    file_stem: OsString,
    source_locator_ref: String,
}

/// How the Context for this import is chosen.
enum ContextPlan {
    /// The directory (or a parent) is already bound.
    Bound { context_id: ContextId, name: String },
    /// Bind the project directory to this name, creating the Context if it does not exist.
    Bind {
        name: String,
        source: SelectionSource,
    },
    /// Exact-file import into one named Context; no binding is read or written.
    Named { name: String },
}

struct FailedSource {
    file: String,
    position: Value,
    reason: String,
}

pub(crate) fn handle_import(args: &[String]) -> i32 {
    let format = super::super::output_format(args);
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", IMPORT_HELP);
        return 0;
    }
    let parsed = match parse(
        args,
        &["--context", "--search", "--session", "--format"],
        &["--dry-run"],
    ) {
        Ok(parsed) => parsed,
        Err(message) => return usage(format, "import", &message),
    };
    let [target] = parsed.positionals.as_slice() else {
        return usage(
            format,
            "import",
            &match parsed.positionals.len() {
                0 => "import requires one project directory or one .jsonl file".to_string(),
                _ => format!(
                    "import accepts exactly one project directory or one .jsonl file; unexpected `{}`",
                    parsed.positionals[1]
                ),
            },
        );
    };
    let mode = match parsed.value("--search").unwrap_or("background") {
        "background" => SearchMode::Background,
        "wait" => SearchMode::Wait,
        "defer" => SearchMode::Defer,
        other => {
            return usage(
                format,
                "import",
                &format!("--search `{other}` is not one of background, wait or defer"),
            );
        }
    };
    let explicit = match parsed.value("--context") {
        Some(raw) => {
            let name = raw.trim();
            if name.is_empty() || name.contains(',') {
                return usage(
                    format,
                    "import",
                    &format!("--context `{raw}` must name exactly one Context"),
                );
            }
            Some(name.to_string())
        }
        None => None,
    };
    let input = Path::new(target);
    let metadata = match fs::metadata(input) {
        Ok(metadata) => metadata,
        Err(_) => {
            return usage(
                format,
                "import",
                &format!("`{target}` is not an existing project directory or .jsonl file"),
            );
        }
    };
    let root = super::super::data_root();
    let started = Instant::now();

    // Resolve the sources before any Context or binding decision.
    let (sources, helpers, project, plan) = if metadata.is_dir() {
        if parsed.value("--session").is_some() {
            return usage(
                format,
                "import",
                "--session applies only to an exact .jsonl file; a project directory imports each session under its own identity",
            );
        }
        let project = match fs::canonicalize(input) {
            Ok(project) => project,
            Err(error) => {
                return usage(
                    format,
                    "import",
                    &format!("cannot resolve `{target}`: {error}"),
                );
            }
        };
        let locator = ClaudeHistoryLocator::new(effective_history_root(
            &super::super::home_dir(),
            std::env::var_os("CLAUDE_CONFIG_DIR")
                .map(PathBuf::from)
                .as_deref(),
        ));
        let resolved = match locator.resolve_project(&project) {
            Ok(resolved) => resolved,
            Err(error) => {
                let message = format!("{error}; nothing was written");
                return refuse(
                    format,
                    1,
                    json!({"ok": false, "state": "source_unreadable", "message": message}),
                    &format!("error: {message}"),
                );
            }
        };
        if resolved.sources.is_empty() {
            let message = "no Claude Code sessions were found for this project; nothing was written. Claude Code files each session under the directory it was started in: import that directory, or name a session file with --context";
            return refuse(
                format,
                2,
                json!({
                    "ok": false,
                    "state": "project_history_not_found",
                    "sources": {"files": 0, "sessions": 0, "helper_transcripts_not_imported": resolved.helper_transcripts_not_imported},
                    "retry": Value::Null,
                    "message": message,
                }),
                &format!("error: {message}"),
            );
        }
        let sources = resolved
            .sources
            .into_iter()
            .map(|source| Source {
                path: source.path,
                file_stem: source.file_stem,
                source_locator_ref: source.source_locator_ref,
            })
            .collect::<Vec<_>>();
        let plan = match project_context_plan(&root, &project, explicit.as_deref(), target, format)
        {
            Ok(plan) => plan,
            Err(code) => return code,
        };
        (
            sources,
            resolved.helper_transcripts_not_imported,
            Some(project),
            plan,
        )
    } else {
        if input
            .extension()
            .is_none_or(|extension| extension != "jsonl")
        {
            return usage(
                format,
                "import",
                &format!("`{target}` is neither a project directory nor a .jsonl session file"),
            );
        }
        let Some(name) = explicit.clone() else {
            return usage(
                format,
                "import",
                "an exact .jsonl file requires --context NAME; it never uses or changes a project binding",
            );
        };
        let path = match fs::canonicalize(input) {
            Ok(path) => path,
            Err(error) => {
                return usage(
                    format,
                    "import",
                    &format!("cannot resolve `{target}`: {error}"),
                );
            }
        };
        let mut digest = Sha256::new();
        digest.update(path.as_os_str().as_encoded_bytes());
        let source = Source {
            file_stem: path.file_stem().map(OsString::from).unwrap_or_default(),
            source_locator_ref: format!("{:x}", digest.finalize()),
            path,
        };
        (vec![source], 0, None, ContextPlan::Named { name })
    };
    let session_override = parsed.value("--session").map(str::to_string);

    if parsed.has("--dry-run") {
        return dry_run(format, &root, &sources, helpers, &plan, mode);
    }

    let placement = match configured_placement(&root) {
        Ok(placement) => placement,
        Err(message) => {
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "config_error", "message": message}),
                &format!("error: {message}"),
            );
        }
    };
    let artifact_available = mode != SearchMode::Defer && search_artifact_available();
    if mode != SearchMode::Defer && !artifact_available && format == OutputFormat::Text {
        eprintln!(
            "note: the local Nomic search model is not installed; history will be captured but not made searchable. Run `cg model fetch --defaults`."
        );
    }

    if let Err(error) = fs::create_dir_all(&root) {
        let message = format!("cannot create the data root {}: {error}", root.display());
        return refuse(
            format,
            1,
            json!({"ok": false, "state": "error", "message": message}),
            &format!("error: {message}"),
        );
    }
    let store = match Store::open(store_config(&root, placement)) {
        Ok(store) => store,
        Err(error) => return library_refusal(format, &error),
    };
    match store.event_placement_state() {
        Ok(EventPlacementState::TenantHubUnbound) => {
            let message = "this store expects tenant-hub event placement and has no authenticated hub binding yet; no Context binding, event schema or event row was written. Finish tenant enrollment for this data root, then retry the import";
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "tenant_policy_blocked", "capture": {"status": "not_run"}, "search": {"status": "not_run"}, "retry": format!("cg import {target}"), "message": message}),
                &format!("error: {message}"),
            );
        }
        Ok(_) => {}
        Err(error) => return library_refusal(format, &error),
    }

    let (context_id, context_name, selection_source, binding_created) =
        match apply_context_plan(&store, &plan, project.as_deref()) {
            Ok(applied) => applied,
            Err(CgError::ContextBindingConflict { .. }) => {
                return binding_conflict(format, target);
            }
            Err(error) => return library_refusal(format, &error),
        };

    // Capture every selected source in order. A malformed or divergent file stops only itself.
    let mut totals = EventCoverageCounts::default();
    let mut inserted = 0_u64;
    let mut already_present = 0_u64;
    let mut refused = 0_u64;
    let mut failed_events = 0_u64;
    let mut batches = 0_u64;
    let mut pending_tails = 0_u64;
    let mut captured_sources = Vec::new();
    let mut failed = Vec::new();
    let mut sessions = std::collections::BTreeSet::new();
    for source in &sources {
        let file_name = source
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut file = match fs::File::open(&source.path) {
            Ok(file) => file,
            Err(error) => {
                failed.push(FailedSource {
                    file: file_name,
                    position: Value::Null,
                    reason: format!("cannot open the session file: {error}"),
                });
                continue;
            }
        };
        let mut adapter = ClaudeCodeJsonlAdapter::new(source.file_stem.clone());
        let options = ImportEventOptions {
            context_id,
            source_locator_ref: source.source_locator_ref.clone(),
            session_override: session_override.clone(),
            batch_limits: Default::default(),
        };
        match store.import_event_stream(&mut adapter, &mut file, options) {
            Ok(receipt) => {
                inserted += receipt.inserted;
                already_present += receipt.already_present;
                refused += receipt.refused;
                failed_events += receipt.failed;
                batches += receipt.batch_count;
                pending_tails += receipt.pending_tail_records;
                totals.complete += receipt.coverage.complete;
                totals.opaque += receipt.coverage.opaque;
                totals.payload_refused += receipt.coverage.payload_refused;
                totals.source_declared_missing += receipt.coverage.source_declared_missing;
                sessions.insert(receipt.source.session_ref.clone());
                captured_sources.push(json!({
                    "source_instance_ref": receipt.source.source_instance_ref,
                    "session_ref": receipt.source.session_ref,
                    "inserted": receipt.inserted,
                    "already_present": receipt.already_present,
                    "last_seq": receipt.event_watermark,
                    "pending_tail_records": receipt.pending_tail_records,
                }));
            }
            Err(error) => match &error {
                CgError::SourceRecordMalformed { position, .. } => failed.push(FailedSource {
                    file: file_name,
                    position: position.clone(),
                    reason: error.to_string(),
                }),
                CgError::SourceHistoryDiverged { .. }
                | CgError::SourceEventDiverged { .. }
                | CgError::SourcePayloadWouldBeTruncated { .. }
                | CgError::InvalidSourceCommitBoundary { .. } => failed.push(FailedSource {
                    file: file_name,
                    position: Value::Null,
                    reason: error.to_string(),
                }),
                _ => {
                    drop(store);
                    return library_refusal(format, &error);
                }
            },
        }
    }
    let capture_ms = started.elapsed().as_millis() as u64;

    // The pending snapshot is taken at capture commit.
    let snapshot = match store.event_history_status(ContextSelector::one(context_id)) {
        Ok(mut set) => set.contexts.remove(0),
        Err(error) => return library_refusal(format, &error),
    };
    let delivery = snapshot.delivery.clone();
    drop(store);

    // A receipt carries no source path: its retry names the input by kind.
    let retry_target = if project.is_some() {
        "<project-directory>".to_string()
    } else {
        format!("<session.jsonl> --context {}", context_name)
    };
    let receipt = ImportReceipt {
        target: retry_target,
        mode,
        context_id,
        context_name,
        selection_source,
        binding_created,
        db_path: super::db_path(&root),
        files: sources.len(),
        sessions: sessions.len(),
        helpers,
        inserted,
        already_present,
        refused,
        failed_events,
        batches,
        pending_tails,
        coverage: totals,
        captured_sources,
        failed,
        capture_ms,
        delivery: delivery
            .map(|delivery| {
                json!({
                    "state": serde_json::to_value(delivery.state).unwrap_or(Value::Null),
                    "eligible": delivery.eligible,
                    "accepted": delivery.accepted,
                    "equivalent": delivery.equivalent,
                    "pending": delivery.pending,
                    "refused": delivery.refused,
                })
            })
            .unwrap_or(Value::Null),
    };
    finish(format, &root, receipt, snapshot, artifact_available)
}

/// `--dry-run`: resolve the sources and the Context, verify every committed prefix through the
/// read-only reader, and report what a real run would capture. Nothing is written.
fn dry_run(
    format: OutputFormat,
    root: &Path,
    sources: &[Source],
    helpers: usize,
    plan: &ContextPlan,
    mode: SearchMode,
) -> i32 {
    let (name, selection_source) = match plan {
        ContextPlan::Bound { name, .. } => (name.clone(), SelectionSource::ProjectBinding),
        ContextPlan::Bind { name, source } => (name.clone(), *source),
        ContextPlan::Named { name } => (name.clone(), SelectionSource::Explicit),
    };
    let reader = match open_reader(root) {
        Ok(reader) => Some(reader),
        Err(CgError::StoreMissing { .. }) => None,
        Err(error) => return library_refusal(format, &error),
    };
    let context_id = match (plan, reader.as_ref()) {
        (ContextPlan::Bound { context_id, .. }, _) => Some(*context_id),
        (_, Some(reader)) => match reader.graph().list_contexts() {
            Ok(contexts) => contexts
                .into_iter()
                .find(|context| context.name == name)
                .map(|context| context.id),
            Err(error) => return library_refusal(format, &error),
        },
        (_, None) => None,
    };
    let declaration = cg_connector_claude_code::event_adapter_declaration();
    let mut planned = Vec::with_capacity(sources.len());
    let (mut new, mut unchanged, mut appended, mut diverged, mut bytes) = (0, 0, 0, 0, 0_u64);
    for source in sources {
        let file = source
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut handle = match fs::File::open(&source.path) {
            Ok(handle) => handle,
            Err(error) => {
                diverged += 1;
                planned.push(
                    json!({"file": file, "prefix": "unreadable", "reason": error.to_string()}),
                );
                continue;
            }
        };
        let check = match (reader.as_ref(), context_id) {
            (Some(reader), Some(context_id)) => reader.check_event_source_prefix(
                context_id,
                &declaration,
                &source.source_locator_ref,
                &mut handle,
            ),
            _ => handle
                .metadata()
                .map(|metadata| EventSourcePrefixCheck::New {
                    source_bytes: metadata.len(),
                })
                .map_err(|error| CgError::Engine(error.to_string())),
        };
        let check = match check {
            Ok(check) => check,
            Err(error) => return library_refusal(format, &error),
        };
        let entry = match &check {
            EventSourcePrefixCheck::New { source_bytes } => {
                new += 1;
                bytes += source_bytes;
                json!({"file": file, "prefix": "new", "bytes_to_capture": source_bytes})
            }
            EventSourcePrefixCheck::Unchanged { committed_bytes } => {
                unchanged += 1;
                json!({"file": file, "prefix": "unchanged", "committed_bytes": committed_bytes, "bytes_to_capture": 0})
            }
            EventSourcePrefixCheck::Appended {
                committed_bytes,
                pending_bytes,
            } => {
                appended += 1;
                bytes += pending_bytes;
                json!({"file": file, "prefix": "appended", "committed_bytes": committed_bytes, "bytes_to_capture": pending_bytes})
            }
            EventSourcePrefixCheck::Diverged {
                committed_bytes,
                reason,
            } => {
                diverged += 1;
                json!({"file": file, "prefix": "diverged", "committed_bytes": committed_bytes, "reason": reason})
            }
        };
        planned.push(entry);
    }
    drop(reader);
    if format == OutputFormat::Text {
        println!(
            "dry run: {} session files for \"{name}\" ({}): {new} new, {appended} grown, {unchanged} unchanged, {diverged} diverged; {} bytes would be captured. Nothing was written.",
            sources.len(),
            selection_source.label(),
            grouped(bytes),
        );
        return 0;
    }
    print_json(&json!({
        "ok": true,
        "state": "dry_run",
        "context": {"id": context_id.map(|id| id.to_string()), "name": name},
        "selection": {"source": selection_source.as_str()},
        "sources": {"files": sources.len(), "helper_transcripts_not_imported": helpers},
        "capture": {
            "status": "not_run",
            "new_sources": new,
            "unchanged_sources": unchanged,
            "appended_sources": appended,
            "diverged_sources": diverged,
            "bytes_to_capture": bytes,
            "sources": planned,
        },
        "search": {"mode": mode.as_str(), "status": "not_run"},
        "message": "dry run: sources resolved and committed prefixes verified; nothing was written",
    }));
    0
}

struct ImportReceipt {
    target: String,
    mode: SearchMode,
    context_id: ContextId,
    context_name: String,
    selection_source: SelectionSource,
    binding_created: bool,
    db_path: PathBuf,
    files: usize,
    sessions: usize,
    helpers: usize,
    inserted: u64,
    already_present: u64,
    refused: u64,
    failed_events: u64,
    batches: u64,
    pending_tails: u64,
    coverage: EventCoverageCounts,
    captured_sources: Vec<Value>,
    failed: Vec<FailedSource>,
    capture_ms: u64,
    delivery: Value,
}

struct SearchOutcome {
    state: &'static str,
    ok: bool,
    code: i32,
    search_status: &'static str,
    worker: &'static str,
    pending: u64,
    covered: u64,
    embedded: u64,
    reason: Option<String>,
    retry: Value,
    search_ms: u64,
}

fn finish(
    format: OutputFormat,
    root: &Path,
    receipt: ImportReceipt,
    snapshot: EventHistoryContextStatus,
    artifact_available: bool,
) -> i32 {
    let pending = snapshot.search.pending_events;
    let covered = snapshot.search.covered_events;
    let wait_retry = format!("cg import {} --search wait", receipt.target);
    let outcome = if !receipt.failed.is_empty() {
        SearchOutcome {
            state: "source_partial",
            ok: false,
            code: 1,
            search_status: if pending == 0 { "ready" } else { "pending" },
            worker: "not_started",
            pending,
            covered,
            embedded: 0,
            reason: None,
            retry: json!(format!("cg import {}", receipt.target)),
            search_ms: 0,
        }
    } else if pending == 0 {
        SearchOutcome {
            state: "complete",
            ok: true,
            code: 0,
            search_status: "ready",
            worker: "not_started",
            pending,
            covered,
            embedded: 0,
            reason: None,
            retry: Value::Null,
            search_ms: 0,
        }
    } else if receipt.mode == SearchMode::Defer {
        SearchOutcome {
            state: "captured_search_deferred",
            ok: true,
            code: 0,
            search_status: "pending",
            worker: "not_started",
            pending,
            covered,
            embedded: 0,
            reason: None,
            retry: json!(wait_retry),
            search_ms: 0,
        }
    } else if !artifact_available {
        SearchOutcome {
            state: "captured_search_behind",
            ok: false,
            code: 1,
            search_status: "blocked",
            worker: "not_started",
            pending,
            covered,
            embedded: 0,
            reason: Some("the local Nomic search model is not installed".to_string()),
            retry: json!(format!("cg model fetch --defaults && {wait_retry}")),
            search_ms: 0,
        }
    } else if receipt.mode == SearchMode::Background {
        let started = Instant::now();
        match worker::kick(root) {
            Ok(worker) => SearchOutcome {
                state: "captured_search_scheduled",
                ok: true,
                code: 0,
                search_status: "scheduled",
                worker,
                pending,
                covered,
                embedded: 0,
                reason: None,
                retry: Value::Null,
                search_ms: started.elapsed().as_millis() as u64,
            },
            Err(reason) => SearchOutcome {
                state: "captured_search_behind",
                ok: false,
                code: 1,
                search_status: "pending",
                worker: "spawn_failed",
                pending,
                covered,
                embedded: 0,
                reason: Some(reason),
                retry: json!(wait_retry),
                search_ms: started.elapsed().as_millis() as u64,
            },
        }
    } else {
        let started = Instant::now();
        match worker::run_in_foreground(root, receipt.context_id) {
            Ok(after) => SearchOutcome {
                state: "complete",
                ok: true,
                code: 0,
                search_status: "ready",
                worker: "foreground",
                pending: after.pending_events,
                covered: after.covered_events,
                embedded: after.covered_events.saturating_sub(covered),
                reason: None,
                retry: Value::Null,
                search_ms: started.elapsed().as_millis() as u64,
            },
            Err(reason) => SearchOutcome {
                state: "captured_search_behind",
                ok: false,
                code: 1,
                search_status: "failed",
                worker: "foreground",
                pending,
                covered,
                embedded: 0,
                reason: Some(reason),
                retry: json!(wait_retry),
                search_ms: started.elapsed().as_millis() as u64,
            },
        }
    };

    if format == OutputFormat::Text {
        render_text(&receipt, &outcome);
        return outcome.code;
    }
    let failed_sources = receipt
        .failed
        .iter()
        .map(|failed| json!({"file": failed.file, "position": failed.position, "reason": failed.reason}))
        .collect::<Vec<_>>();
    let mut value = json!({
        "ok": outcome.ok,
        "state": outcome.state,
        "context": {"id": receipt.context_id.to_string(), "name": receipt.context_name},
        "selection": {
            "source": receipt.selection_source.as_str(),
            "binding_created": receipt.binding_created,
        },
        "store": {"db_path": receipt.db_path.display().to_string()},
        "sources": {
            "files": receipt.files,
            "sessions": receipt.sessions,
            "helper_transcripts_not_imported": receipt.helpers,
        },
        "capture": {
            "status": if receipt.failed.is_empty() { "complete" } else { "partial" },
            "events": {
                "inserted": receipt.inserted,
                "already_present": receipt.already_present,
                "refused": receipt.refused,
                "failed": receipt.failed_events,
            },
            "coverage": {
                "complete": receipt.coverage.complete,
                "opaque": receipt.coverage.opaque,
                "payload_refused": receipt.coverage.payload_refused,
                "source_declared_missing": receipt.coverage.source_declared_missing,
            },
            "batches": receipt.batches,
            "pending_tail_records": receipt.pending_tails,
            "sources": receipt.captured_sources,
            "failed_sources": failed_sources,
            "local_embedding_calls": 0,
            "understanding_calls": 0,
            "hosted_provider_calls": 0,
            "elapsed_ms": receipt.capture_ms,
        },
        "search": {
            "mode": receipt.mode.as_str(),
            "status": outcome.search_status,
            "worker": outcome.worker,
            "covered_events": outcome.covered,
            "pending_events": outcome.pending,
            "local_embedding_calls": outcome.embedded,
            "elapsed_ms": outcome.search_ms,
        },
        "delivery": receipt.delivery,
        "retry": outcome.retry,
    });
    if let Some(reason) = &outcome.reason {
        value["search"]["reason"] = json!(reason);
    }
    print_json(&value);
    outcome.code
}

fn render_text(receipt: &ImportReceipt, outcome: &SearchOutcome) {
    let source = match receipt.selection_source {
        SelectionSource::ProjectBinding if receipt.binding_created => "project binding, new",
        other => other.label(),
    };
    println!("context: {} ({source})", receipt.context_name);
    println!(
        "sources: {} files / {} sessions{}",
        receipt.files,
        receipt.sessions,
        if receipt.helpers > 0 {
            format!(" ({} helper transcripts not imported)", receipt.helpers)
        } else {
            String::new()
        }
    );
    println!(
        "history captured: {} new events ({} already present)",
        grouped(receipt.inserted),
        grouped(receipt.already_present)
    );
    for failed in &receipt.failed {
        println!("failed source: {} — {}", failed.file, failed.reason);
    }
    let search = match outcome.search_status {
        "ready" => "semantic search: caught up".to_string(),
        "scheduled" => format!(
            "semantic search: scheduled in the background — {} events still need passage embeddings; `cg status` reports when it is ready",
            grouped(outcome.pending)
        ),
        "pending" => format!(
            "semantic search: behind — {} events still need passage embeddings",
            grouped(outcome.pending)
        ),
        "blocked" => format!(
            "semantic search: blocked — local Nomic artifact is not present; {} events still need passage embeddings",
            grouped(outcome.pending)
        ),
        _ => format!(
            "semantic search: behind — {} events still need passage embeddings{}",
            grouped(outcome.pending),
            outcome
                .reason
                .as_ref()
                .map(|reason| format!(" ({reason})"))
                .unwrap_or_default()
        ),
    };
    println!("{search}");
    if let Some(retry) = outcome.retry.as_str() {
        println!("next: {retry}");
    }
}

fn binding_conflict(format: OutputFormat, target: &str) -> i32 {
    let message = format!(
        "`{target}` is already linked to a different Context; no Context, binding, event or projection row changed. Import it without --context to use its linked Context"
    );
    refuse(
        format,
        2,
        json!({"ok": false, "state": "context_binding_conflict", "message": message, "retry": format!("cg import {target}")}),
        &format!("error: {message}"),
    )
}

/// Decide the Context for a project directory import without writing anything.
fn project_context_plan(
    root: &Path,
    project: &Path,
    explicit: Option<&str>,
    target: &str,
    format: OutputFormat,
) -> Result<ContextPlan, i32> {
    let mut existing_names = Vec::new();
    let mut bound = None;
    match open_reader(root) {
        Ok(reader) => {
            match nearest_binding(&reader, project) {
                Ok(Some((context_id, _))) => {
                    let name = reader
                        .graph()
                        .get_context(context_id)
                        .map_err(|error| library_refusal(format, &error))?
                        .map(|context| context.name)
                        .unwrap_or_else(|| context_id.to_string());
                    bound = Some((context_id, name));
                }
                Ok(None) => {}
                Err(error) => return Err(library_refusal(format, &error)),
            }
            existing_names = reader
                .graph()
                .list_contexts()
                .map_err(|error| library_refusal(format, &error))?
                .into_iter()
                .map(|context| context.name)
                .collect();
        }
        Err(CgError::StoreMissing { .. }) | Err(CgError::TenantPolicyBindingRequired { .. }) => {}
        Err(error) => return Err(library_refusal(format, &error)),
    }
    if let Some((context_id, name)) = bound {
        if let Some(explicit) = explicit
            && explicit != name
        {
            return Err(binding_conflict(format, target));
        }
        return Ok(ContextPlan::Bound { context_id, name });
    }
    if let Some(name) = explicit {
        return Ok(ContextPlan::Bind {
            name: name.to_string(),
            source: SelectionSource::Explicit,
        });
    }
    let suggested = project
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| "default".to_string());
    let required = |format: OutputFormat| {
        let retry = format!("cg import {target} --context {suggested}");
        let message = format!(
            "no Context is linked to this project and there is no terminal to ask on; nothing was written. Run `{retry}`"
        );
        refuse(
            format,
            2,
            json!({
                "ok": false,
                "state": "context_binding_required",
                "suggested_context": suggested,
                "retry": retry,
                "message": message,
            }),
            &format!("error: {message}"),
        )
    };
    let Some(mut terminal) = Terminal::open() else {
        return Err(required(format));
    };
    terminal.say("No context is linked to this project.");
    let answer = terminal.ask(&format!("Context name [{suggested}]: "));
    let name = if answer.is_empty() {
        suggested.clone()
    } else {
        answer
    };
    if name.contains(',') {
        return Err(usage(
            format,
            "import",
            &format!("Context name `{name}` must not contain a comma"),
        ));
    }
    if existing_names.contains(&name)
        && !terminal.confirm(&format!(
            "Context \"{name}\" already exists. Link this project to it? [y/N] "
        ))
    {
        return Err(required(format));
    }
    Ok(ContextPlan::Bind {
        name,
        source: SelectionSource::ProjectBinding,
    })
}

/// Apply the Context decision. A new Context and its binding commit together; an existing Context
/// is bound without being changed.
fn apply_context_plan(
    store: &Store,
    plan: &ContextPlan,
    project: Option<&Path>,
) -> Result<(ContextId, String, SelectionSource, bool), CgError> {
    let existing = |name: &str| -> Result<Option<ContextId>, CgError> {
        Ok(store
            .list_contexts()?
            .into_iter()
            .find(|context| context.name == name)
            .map(|context| context.id))
    };
    match plan {
        ContextPlan::Bound { context_id, name } => Ok((
            *context_id,
            name.clone(),
            SelectionSource::ProjectBinding,
            false,
        )),
        ContextPlan::Named { name } => {
            let id = match existing(name)? {
                Some(id) => id,
                None => {
                    store
                        .create_context(CreateContext {
                            name: name.clone(),
                            ..Default::default()
                        })?
                        .id
                }
            };
            Ok((id, name.clone(), SelectionSource::Explicit, false))
        }
        ContextPlan::Bind { name, source } => {
            let project = project.expect("a binding plan always has a project directory");
            let key = workspace_binding_key(project);
            let receipt = match existing(name)? {
                Some(id) => store.bind_context(key, id)?,
                None => match store.create_context_with_binding(
                    CreateContext {
                        name: name.clone(),
                        ..Default::default()
                    },
                    key.clone(),
                ) {
                    Ok(receipt) => receipt,
                    Err(CgError::ContextAlreadyExists { .. }) => {
                        let id = existing(name)?.ok_or(CgError::NotFound {
                            id: None,
                            missing: Some("context"),
                        })?;
                        store.bind_context(key, id)?
                    }
                    Err(error) => return Err(error),
                },
            };
            Ok((receipt.context_id, name.clone(), *source, receipt.changed))
        }
    }
}

const IMPORT_HELP: &str = "Usage: cg import <project-directory>|<session.jsonl> [--context NAME] [--search background|wait|defer] [--session REF] [--dry-run] [--format json|text]

Example:
  cd ~/projects/my-app && cg import .

Imports Claude Code conversation history into this machine's Context Graph store.

A project directory reaches the Claude Code sessions Claude recorded for exactly that directory
(its local history under ~/.claude/projects, or $CLAUDE_CONFIG_DIR/projects). The project tree is
never searched for .jsonl files, and sessions started in a sub-directory belong to that
sub-directory. The first import asks once, on the terminal, which Context the project belongs to
and remembers the answer; later imports, finds, shows and statuses from the project or any
sub-directory use it. Without a terminal, pass --context NAME.

A single .jsonl file imports exactly that session into the Context named by --context and never
links the current directory. --session REF overrides that file's session reference.

Capture is durable and model-free. Search is then made ready according to --search:
  background (default)  schedule the local search index and return; `cg status` reports progress
  wait                  build the search index before returning
  defer                 capture only; start no search work

background and wait need the local Nomic model (`cg model fetch --defaults`); nothing is
downloaded by import. JSON is the default output; --format text prints a human summary.
--dry-run resolves the sources and the Context and writes nothing.

Exit: 0 complete / scheduled / deferred; 1 captured_search_behind, source_partial,
schema_mismatch, tenant_policy_blocked; 2 project_history_not_found, context_binding_required,
context_binding_conflict, usage error; 4 store_busy.";
