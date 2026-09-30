//! `cg import <project-directory>|<session.jsonl>`.

use std::ffi::OsString;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cg_connector_claude_code::{
    ClaudeCodeJsonlAdapter, ClaudeHistoryLocator, effective_history_root,
};
use context_graph::{
    CgError, ContextId, ContextSelector, CreateContext, EventCoverageCounts, EventPlacementState,
    EventSearchWorkAssessment, EventSearchWorkPolicy, EventSourceAdapter, EventSourcePrefixCheck,
    ImportEventOptions, Store,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    OutputFormat, SelectionSource, Terminal, configured_passage_profile, configured_placement,
    grouped, import_retry_command, library_refusal, nearest_binding, open_reader, parse,
    parse_embedding_workers, parse_search_order, parse_search_sessions, print_json, refuse,
    resolved_embedding_workers, resolved_search_work_policy, search_artifact_available,
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

#[derive(Clone, Copy)]
struct EmbeddingWorkerRequest {
    resolved: NonZeroUsize,
    explicit: Option<NonZeroUsize>,
}

struct ProjectContextOptions {
    read_only: bool,
    format: OutputFormat,
    read_memory_limit_bytes: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
    embedding_workers: EmbeddingWorkerRequest,
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

pub(crate) fn handle_import(
    args: &[String],
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> i32 {
    let format = super::super::output_format(args);
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", IMPORT_HELP);
        return 0;
    }
    let parsed = match parse(
        args,
        &[
            "--context",
            "--search",
            "--embedding-workers",
            "--search-order",
            "--search-sessions",
            "--session",
            "--format",
        ],
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
    let explicit_embedding_workers = match parsed.value("--embedding-workers") {
        Some(value) => match parse_embedding_workers(value, "--embedding-workers") {
            Ok(value) => Some(value),
            Err(message) => return usage(format, "import", &message),
        },
        None => None,
    };
    if let Some(value) = parsed.value("--search-order")
        && let Err(message) = parse_search_order(value, "--search-order")
    {
        return usage(format, "import", &message);
    }
    if let Some(value) = parsed.value("--search-sessions")
        && let Err(message) = parse_search_sessions(value, "--search-sessions")
    {
        return usage(format, "import", &message);
    }
    let embedding_workers =
        match resolved_embedding_workers(&root, parsed.value("--embedding-workers")) {
            Ok(value) => value,
            Err(message) => {
                return refuse(
                    format,
                    1,
                    json!({"ok": false, "state": "config_error", "message": message}),
                    &format!("error: {message}"),
                );
            }
        };
    let embedding_worker_request = EmbeddingWorkerRequest {
        resolved: embedding_workers,
        explicit: explicit_embedding_workers,
    };
    let work_policy = match resolved_search_work_policy(
        &root,
        parsed.value("--search-order"),
        parsed.value("--search-sessions"),
    ) {
        Ok(policy) => policy,
        Err(message) => {
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "config_error", "message": message}),
                &format!("error: {message}"),
            );
        }
    };
    // Placement and passage profile are pure configuration admission. Keep them beside the
    // worker-cap check so even a real import refuses before project planning can recover a store.
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
    let desired_profile = match configured_passage_profile(&root) {
        Ok(profile) => profile,
        Err(message) => {
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "config_error", "message": message}),
                &format!("error: {message}"),
            );
        }
    };
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
        let plan = match project_context_plan(
            &root,
            &project,
            explicit.as_deref(),
            target,
            ProjectContextOptions {
                read_only: parsed.has("--dry-run"),
                format,
                read_memory_limit_bytes,
                work_policy,
                embedding_workers: embedding_worker_request,
            },
        ) {
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
        return dry_run(
            format,
            &root,
            &sources,
            helpers,
            &plan,
            mode,
            read_memory_limit_bytes,
        );
    }

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
    let store = match Store::open(store_config(&root, placement, read_memory_limit_bytes)) {
        Ok(store) => store,
        Err(error) => return library_refusal(format, &error),
    };
    match store.event_placement_state() {
        Ok(EventPlacementState::TenantHubUnbound) => {
            let message = "this store expects tenant-hub event placement and has no authenticated hub binding yet; no Context binding, event schema or event row was written. Finish tenant enrollment for this data root, then retry the import";
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "tenant_policy_blocked", "capture": {"status": "not_run"}, "search": {"status": "not_run"}, "retry": retry_import(target, explicit.as_deref(), None, work_policy, embedding_worker_request), "message": message}),
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
                return binding_conflict(format, target, work_policy, embedding_worker_request);
            }
            Err(error) => return library_refusal(format, &error),
        };

    // Capture receipts and later read retries must retain the selection this invocation resolved,
    // even when defer or a missing model means no projection runner writes it.
    if let Err(error) = store.persist_event_search_work_policy(context_id, work_policy) {
        return library_refusal(format, &error);
    }

    let captured_before = match store.event_history_status(ContextSelector::one(context_id)) {
        Ok(status) => status.contexts[0].captured_events,
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
                CgError::WriteDidNotFitTheMemoryLimit {
                    requested_bytes,
                    limit_bytes,
                } => {
                    let captured_after =
                        match store.event_history_status(ContextSelector::one(context_id)) {
                            Ok(status) => status.contexts[0].captured_events,
                            Err(error) => return library_refusal(format, &error),
                        };
                    let current_committed_bytes = match store.check_event_source_prefix(
                        context_id,
                        adapter.declaration(),
                        &source.source_locator_ref,
                        &mut file,
                    ) {
                        Ok(EventSourcePrefixCheck::Unchanged { committed_bytes })
                        | Ok(EventSourcePrefixCheck::Appended {
                            committed_bytes, ..
                        })
                        | Ok(EventSourcePrefixCheck::Diverged {
                            committed_bytes, ..
                        }) => committed_bytes,
                        Ok(EventSourcePrefixCheck::New { .. }) => 0,
                        Err(error) => return library_refusal(format, &error),
                    };
                    let position = if current_committed_bytes > 0 || captured_sources.is_empty() {
                        json!({"file": file_name, "committed_bytes": current_committed_bytes})
                    } else {
                        let previous = captured_sources.last().expect("nonempty checked above");
                        json!({
                            "source_instance_ref": previous["source_instance_ref"],
                            "last_seq": previous["last_seq"],
                        })
                    };
                    let Some(committed_this_import) = captured_after.checked_sub(captured_before)
                    else {
                        return library_refusal(
                            format,
                            &CgError::Engine(
                                "captured event count decreased during import".to_string(),
                            ),
                        );
                    };
                    let message = format!(
                        "cg import stopped because its next write needed another {} of working \
                         memory and could not fit under the {} writer memory limit in force. \
                         Earlier committed events remain in the store; the refused write changed \
                         nothing. To continue, {}, then retry the same import; cg resumes from \
                         committed progress.",
                        context_graph::write_budget::human_bytes(*requested_bytes as u64),
                        context_graph::write_budget::human_bytes(*limit_bytes as u64),
                        context_graph::write_budget::how_to_raise_the_writer_budget(),
                    );
                    drop(store);
                    return refuse(
                        format,
                        1,
                        json!({
                            "ok": false,
                            "state": "write_memory_limit",
                            "condition": "WriteDidNotFitTheMemoryLimit",
                            "message": message,
                            "requested_bytes": requested_bytes,
                            "limit_bytes": limit_bytes,
                            "context": {"id": context_id.to_string(), "name": context_name},
                            "capture": {
                                "status": "partial",
                                "events": {
                                    "inserted": committed_this_import,
                                    "already_present": already_present,
                                    "refused": refused,
                                    "failed": failed_events,
                                },
                                "last_committed_position": position,
                                "sources": captured_sources,
                            },
                            "search": {"status": "not_run"},
                        }),
                        &format!("error: {message}"),
                    );
                }
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
    let work = match store.event_search_work_with_policy(
        ContextSelector::one(context_id),
        &super::search_work_spec(&desired_profile),
        work_policy,
    ) {
        Ok(mut assessments) => assessments.remove(0),
        Err(error) => return library_refusal(format, &error),
    };
    let background_request = if mode == SearchMode::Background
        && work.work_required
        && artifact_available
        && failed.is_empty()
    {
        match store.select_event_search_work(context_id, work_policy) {
            Ok(selected) => Some(worker::accept_background_work(
                &root,
                &store,
                context_id,
                embedding_worker_request.resolved,
                work_policy,
                &selected,
                &super::search_work_spec(&desired_profile),
            )),
            Err(error) => return library_refusal(format, &error),
        }
    } else {
        None
    };
    let observed_memory_refusals =
        if failed.is_empty() && !work.work_required && mode != SearchMode::Defer {
            match worker::completed_memory_refusals(
                &root,
                &store,
                context_id,
                work_policy,
                &desired_profile,
            ) {
                Ok(observed) => observed,
                Err(error) => return library_refusal(format, &error),
            }
        } else {
            Vec::new()
        };
    drop(store);
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    worker::pause_before_import_acknowledgement();

    // A receipt carries no source path: its retry names the input by kind.
    let retry_target = if project.is_some() {
        "<project-directory>".to_string()
    } else {
        "<session.jsonl>".to_string()
    };
    let receipt = ImportReceipt {
        target: retry_target,
        retry_context: project.is_none().then(|| context_name.clone()),
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
        background_request,
        observed_memory_refusals,
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
    finish(
        format,
        (&root, read_memory_limit_bytes),
        receipt,
        work,
        &desired_profile,
        artifact_available,
        embedding_worker_request,
        work_policy,
    )
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
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> i32 {
    let (name, selection_source) = match plan {
        ContextPlan::Bound { name, .. } => (name.clone(), SelectionSource::ProjectBinding),
        ContextPlan::Bind { name, source } => (name.clone(), *source),
        ContextPlan::Named { name } => (name.clone(), SelectionSource::Explicit),
    };
    let reader = match open_reader(root, read_memory_limit_bytes) {
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
    retry_context: Option<String>,
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
    background_request: Option<Result<worker::PendingBackgroundRequest, String>>,
    delivery: Value,
    observed_memory_refusals: Vec<String>,
}

struct SearchOutcome {
    state: &'static str,
    ok: bool,
    code: i32,
    search_status: &'static str,
    worker: &'static str,
    pending: u64,
    covered: u64,
    outside_scope: u64,
    outside_scope_indexed: u64,
    no_searchable_text: u64,
    embedded: u64,
    reason: Option<String>,
    retry: Value,
    search_ms: u64,
}

#[allow(clippy::too_many_arguments)]
fn finish(
    format: OutputFormat,
    (root, read_memory_limit_bytes): (&Path, std::num::NonZeroUsize),
    receipt: ImportReceipt,
    work: EventSearchWorkAssessment,
    desired_profile: &context_graph::EventSearchProfile,
    artifact_available: bool,
    embedding_worker_request: EmbeddingWorkerRequest,
    work_policy: EventSearchWorkPolicy,
) -> i32 {
    if receipt.failed.is_empty()
        && !work.work_required
        && receipt.mode != SearchMode::Defer
        && let Err(error) = worker::acknowledge_completed_memory_refusal(
            root,
            receipt.context_id,
            &receipt.observed_memory_refusals,
            read_memory_limit_bytes,
        )
    {
        return library_refusal(format, &error);
    }
    let pending = work.pending_events;
    let covered = work.covered_events;
    let outside_scope = work.outside_scope_events;
    let outside_scope_indexed = work.outside_scope_indexed_events;
    let no_searchable_text = work.no_searchable_text_events;
    let wait_retry = retry_import(
        &receipt.target,
        receipt.retry_context.as_deref(),
        Some(SearchMode::Wait),
        work_policy,
        embedding_worker_request,
    );
    let outcome = if !receipt.failed.is_empty() {
        SearchOutcome {
            state: "source_partial",
            ok: false,
            code: 1,
            search_status: if work.work_required {
                "pending"
            } else {
                "ready"
            },
            worker: "not_started",
            pending,
            covered,
            outside_scope,
            outside_scope_indexed,
            no_searchable_text,
            embedded: 0,
            reason: None,
            retry: json!(retry_import(
                &receipt.target,
                receipt.retry_context.as_deref(),
                None,
                work_policy,
                embedding_worker_request,
            )),
            search_ms: 0,
        }
    } else if !work.work_required {
        SearchOutcome {
            state: "complete",
            ok: true,
            code: 0,
            search_status: "ready",
            worker: "not_started",
            pending,
            covered,
            outside_scope,
            outside_scope_indexed,
            no_searchable_text,
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
            outside_scope,
            outside_scope_indexed,
            no_searchable_text,
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
            outside_scope,
            outside_scope_indexed,
            no_searchable_text,
            embedded: 0,
            reason: Some("the local Nomic search model is not installed".to_string()),
            retry: json!(format!("cg model fetch --defaults && {wait_retry}")),
            search_ms: 0,
        }
    } else if receipt.mode == SearchMode::Background {
        let started = Instant::now();
        match receipt
            .background_request
            .clone()
            .expect("background pending work was accepted")
            .and_then(|request| {
                worker::kick_accepted_background_work(root, desired_profile, request)
            }) {
            Ok(worker) => SearchOutcome {
                state: "captured_search_scheduled",
                ok: true,
                code: 0,
                search_status: "scheduled",
                worker,
                pending,
                covered,
                outside_scope,
                outside_scope_indexed,
                no_searchable_text,
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
                outside_scope,
                outside_scope_indexed,
                no_searchable_text,
                embedded: 0,
                reason: Some(reason),
                retry: json!(wait_retry),
                search_ms: started.elapsed().as_millis() as u64,
            },
        }
    } else {
        let started = Instant::now();
        let after = worker::run_in_foreground(
            root,
            receipt.context_id,
            desired_profile,
            embedding_worker_request.resolved,
            work_policy,
            read_memory_limit_bytes,
        );
        let coverage = after.work.as_ref().unwrap_or(&work);
        match after.result {
            Ok(()) => SearchOutcome {
                state: "complete",
                ok: true,
                code: 0,
                search_status: "ready",
                worker: "foreground",
                pending: coverage.pending_events,
                covered: coverage.covered_events,
                outside_scope: coverage.outside_scope_events,
                outside_scope_indexed: coverage.outside_scope_indexed_events,
                no_searchable_text: coverage.no_searchable_text_events,
                embedded: after.embedding_invocations,
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
                pending: coverage.pending_events,
                covered: coverage.covered_events,
                outside_scope: coverage.outside_scope_events,
                outside_scope_indexed: coverage.outside_scope_indexed_events,
                no_searchable_text: coverage.no_searchable_text_events,
                embedded: after.embedding_invocations,
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
            "outside_scope_events": outcome.outside_scope,
            "outside_scope_indexed_events": outcome.outside_scope_indexed,
            "no_searchable_text_events": outcome.no_searchable_text,
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
    if outcome.outside_scope > 0 || outcome.no_searchable_text > 0 {
        println!(
            "search coverage: {} events indexed; {} outside this build scope ({} already indexed); {} have no searchable text",
            grouped(outcome.covered),
            grouped(outcome.outside_scope),
            grouped(outcome.outside_scope_indexed),
            grouped(outcome.no_searchable_text)
        );
    }
    if let Some(retry) = outcome.retry.as_str() {
        println!("next: {retry}");
    }
}

fn retry_import(
    target: &str,
    context: Option<&str>,
    mode: Option<SearchMode>,
    work_policy: EventSearchWorkPolicy,
    embedding_workers: EmbeddingWorkerRequest,
) -> String {
    import_retry_command(
        target,
        context,
        mode.map(SearchMode::as_str),
        work_policy,
        embedding_workers.explicit,
    )
}

fn binding_conflict(
    format: OutputFormat,
    target: &str,
    work_policy: EventSearchWorkPolicy,
    embedding_workers: EmbeddingWorkerRequest,
) -> i32 {
    let message = format!(
        "`{target}` is already linked to a different Context; no Context, binding, event or projection row changed. Import it without --context to use its linked Context"
    );
    refuse(
        format,
        2,
        json!({"ok": false, "state": "context_binding_conflict", "message": message, "retry": retry_import(target, None, None, work_policy, embedding_workers)}),
        &format!("error: {message}"),
    )
}

/// Decide the Context for a project directory import without writing anything.
fn project_context_plan(
    root: &Path,
    project: &Path,
    explicit: Option<&str>,
    target: &str,
    options: ProjectContextOptions,
) -> Result<ContextPlan, i32> {
    let ProjectContextOptions {
        read_only,
        format,
        read_memory_limit_bytes,
        work_policy,
        embedding_workers,
    } = options;
    let mut existing_names = Vec::new();
    let mut bound = None;
    // Planning reads read-only so an unbound, unattended import writes nothing. A store whose
    // last writer died refuses every read-only open until a writable open settles it. Ordinary
    // import recovers there; a dry-run must return that typed refusal without opening the store
    // writable.
    let reader = match open_reader(root, read_memory_limit_bytes) {
        Err(error @ CgError::StoreNeedsWritableRecovery { .. }) if read_only => {
            return Err(library_refusal(format, &error));
        }
        Err(CgError::StoreNeedsWritableRecovery { .. }) => {
            match Store::open(store_config(root, None, read_memory_limit_bytes)) {
                Ok(store) => drop(store),
                Err(error) => return Err(library_refusal(format, &error)),
            }
            open_reader(root, read_memory_limit_bytes)
        }
        other => other,
    };
    match reader {
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
            return Err(binding_conflict(
                format,
                target,
                work_policy,
                embedding_workers,
            ));
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
        let retry = retry_import(
            target,
            Some(&suggested),
            None,
            work_policy,
            embedding_workers,
        );
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

const IMPORT_HELP: &str = "Usage: cg import <project-directory>|<session.jsonl> [--context NAME] [--search background|wait|defer] [--embedding-workers N] [--search-order newest|oldest] [--search-sessions N|all] [--session REF] [--dry-run] [--format json|text]

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

Pass --embedding-workers N to bound simultaneous passage embedding calls for this request. N must
be positive; the flag overrides [search].embedding_workers, and omission uses that setting or
available CPU parallelism. This execution limit does not create a generation or rebuild unchanged
history. A deferred or already-caught-up import does not change active work. A foreground wait
keeps its resolved limit, publishes that choice, and starts no background worker unless a later
request is queued.

The selected --write-memory-limit-bytes ceiling stays with pending background work across restart.
Different Contexts keep their own ceilings; shared pending work uses the lower ceiling. The worker
releases writable state before reopening under a changed ceiling and refuses if it does not fit.
This limits engine write working memory, not total process RAM. `cg status` reports a refused
background write with its needed and allowed memory; retry with an affordable ceiling.

Pass --search-order newest|oldest to choose processing order and --search-sessions N|all to bound
embedding to the N sessions chosen by --search-order (default: the 10 most recent). These flags override [search].order and
[search].sessions for this invocation; set [search] to make a choice persistent. `all` can make
the first search build much longer.

background and wait need the local Nomic model (`cg model fetch --defaults`); nothing is
downloaded by import. JSON is the default output; --format text prints a human summary.
--dry-run resolves the sources and the Context and writes nothing.

Exit: 0 complete / scheduled / deferred; 1 captured_search_behind, source_partial,
schema_mismatch, store_needs_recovery, tenant_policy_blocked, config_error; 2 project_history_not_found,
context_binding_required, context_binding_conflict, usage_error; 4 store_busy.";
