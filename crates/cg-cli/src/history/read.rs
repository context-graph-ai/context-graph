//! `cg status`, `cg find` and `cg show`: read-only, through the library's read-only reader.
//!
//! None of these opens a writable Store, so none is ever refused as busy while an import or a
//! projection worker holds the database.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use cg_connector_claude_code::{ClaudeHistoryLocator, effective_history_root};
use context_graph::{
    CgError, ConsumerReader, ContextSelector, ConversationEvent, EventHistoryContextStatus,
    EventHistorySearchState, EventHistoryStatusSet, EventPlacementState, EventSearchContextState,
    EventSearchFilter, EventSearchHit, EventSearchPassage, EventSearchPassageId,
    PassageEventSource,
};
use serde_json::{Value, json};

use super::{
    OutputFormat, Selection, db_path, grouped, import_retry_command, library_refusal, open_reader,
    parse, print_json, refuse, resolve_selection, search_artifact_available, search_registration,
    selection_refusal, store_not_found, usage,
};

/// Open the read-only reader, or answer for a store that is not there.
fn reader_or_refusal(
    format: OutputFormat,
    root: &Path,
    with_next: bool,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> Result<ConsumerReader, i32> {
    match open_reader(root, read_memory_limit_bytes) {
        Ok(reader) => Ok(reader),
        Err(CgError::StoreMissing { .. }) => {
            Err(store_not_found(format, &db_path(root), with_next))
        }
        Err(error) => Err(library_refusal(format, &error)),
    }
}

fn state_name(state: EventHistorySearchState) -> &'static str {
    match state {
        EventHistorySearchState::Pending => "pending",
        EventHistorySearchState::Running => "running",
        EventHistorySearchState::Blocked => "blocked",
        EventHistorySearchState::Failed => "failed",
        EventHistorySearchState::Ready => "ready",
    }
}

fn placement_name(placement: EventPlacementState) -> &'static str {
    match placement {
        EventPlacementState::LocalOnly => "local_only",
        EventPlacementState::TenantHubUnbound => "tenant_hub_unbound",
        EventPlacementState::TenantHubBound => "tenant_hub",
    }
}

// ---------------------------------------------------------------------------------------------
// cg status
// ---------------------------------------------------------------------------------------------

pub(crate) fn handle_status(
    args: &[String],
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> i32 {
    let format = super::super::output_format(args);
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{STATUS_HELP}");
        return 0;
    }
    let parsed = match parse(
        args,
        &["--context", "--tenant-id", "--format"],
        &["--all-contexts"],
    ) {
        Ok(parsed) => parsed,
        Err(message) => return usage(format, "status", &message),
    };
    if let Some(extra) = parsed.positionals.first() {
        return usage(
            format,
            "status",
            &format!(
                "unexpected argument `{extra}`; model connections are under `cg model status`"
            ),
        );
    }
    let root = super::super::data_root();
    let reader = match open_reader(&root, read_memory_limit_bytes) {
        Ok(reader) => reader,
        Err(CgError::StoreMissing { .. }) => {
            return store_not_found(format, &db_path(&root), false);
        }
        Err(CgError::TenantPolicyBindingRequired { .. }) => {
            let message = "this store expects tenant-hub event placement and has no authenticated hub binding yet; it holds no application data";
            let value = json!({
                "ok": true,
                "state": "ok",
                "store": {
                    "db_path": db_path(&root).display().to_string(),
                    "placement": "tenant_hub_unbound",
                    "expected_policy": "tenant_hub",
                    "binding": "blocked",
                },
                "contexts": [],
                "message": message,
            });
            if format == OutputFormat::Text {
                println!("event placement: tenant_hub — blocked: {message}");
            } else {
                print_json(&value);
            }
            return 0;
        }
        Err(error) => return library_refusal(format, &error),
    };
    let selection = match resolve_selection(
        &reader,
        &root,
        parsed.value("--context"),
        parsed.has("--all-contexts"),
        parsed.value("--tenant-id"),
    ) {
        Ok(selection) => selection,
        Err(error) => return selection_refusal(format, "status", error),
    };
    let db_path_text = db_path(&root).display().to_string();
    let mut set = match ContextSelector::new(selection.contexts.iter().map(|(id, _)| *id)) {
        Ok(selector) => match injected_read_failure("counts")
            .and_then(|()| reader.event_history_status(selector))
        {
            Ok(set) => Some(set.with_search_artifact_available(search_artifact_available())),
            Err(error) => return status_read_refusal(format, &db_path_text, &error),
        },
        Err(_) => None,
    };
    let refusals = match super::worker::background_memory_refusals(&root) {
        Ok(refusals) => refusals,
        Err(error) => return library_refusal(format, &CgError::Engine(error)),
    };
    if let Some(set) = &mut set {
        for context in &mut set.contexts {
            if let Some(refusal) = refusals.get(&context.context_id.to_string()) {
                context.search.state = EventHistorySearchState::Failed;
                context.search.last_failure = Some(refusal.message());
            }
        }
    }
    let placement = match &set {
        Some(set) => (set.placement, set.expected_policy),
        None => match reader.event_history_placement() {
            Ok(placement) => placement,
            Err(error) => return library_refusal(format, &error),
        },
    };
    // Store-level graph and fabric state that is not history: the last understanding run, the
    // fabric sharing posture with the input copies in the push set, and the last injection.
    let last_understanding_run = match injected_read_failure("understanding_run").and_then(|()| {
        context_graph::tier0::get_local_setting_read_only(
            &reader,
            super::super::LAST_UNDERSTANDING_RUN_KEY,
        )
    }) {
        Ok(raw) => raw.and_then(|raw| serde_json::from_str::<Value>(&raw).ok()),
        Err(error) => return status_read_refusal(format, &db_path_text, &error),
    };
    let fabric = match context_graph::fabric::FabricConfig::load(&root) {
        Ok(Some(config)) => {
            match injected_read_failure("fabric_membership").and_then(|()| {
                context_graph::fabric::input_material_in_push_set_read_only(&reader, &config)
            }) {
                Ok(inputs) => Some(json!({
                    "sharing": if config.sharing { "on" } else { "off" },
                    "inputs_in_push_set": inputs,
                })),
                Err(error) => return status_read_refusal(format, &db_path_text, &error),
            }
        }
        Ok(None) => None,
        Err(error) => return library_refusal(format, &error),
    };
    let last_injection = super::super::read_last_injection(&root);
    drop(reader);
    let helpers = selection
        .bound_directory
        .as_deref()
        .and_then(helper_transcripts_not_imported);

    let writer_memory = crate::write_config::in_force();
    if format == OutputFormat::Text {
        render_status_text(&selection, set.as_ref(), placement.0, helpers);
        if let Some(fabric) = &fabric {
            println!(
                "fabric: sharing {}, {} understanding input copies in push set",
                fabric["sharing"].as_str().unwrap_or("off"),
                fabric["inputs_in_push_set"]
            );
        }
        // What one write on this machine is allowed to hold, and which of the three answers set
        // it: an operator whose import was refused for want of memory reads the number they have
        // to raise right here.
        println!(
            "write memory: {} — {}",
            context_graph::write_budget::human_bytes(writer_memory.bytes.get() as u64),
            writer_memory.origin()
        );
        return 0;
    }
    let contexts = set
        .as_ref()
        .map(|set| {
            set.contexts
                .iter()
                .map(|context| context_status_json(context, helpers, fabric.as_ref()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut value = json!({
        "ok": true,
        "state": "ok",
        "store": {
            "db_path": db_path(&root).display().to_string(),
            "placement": placement_name(placement.0),
            "expected_policy": serde_json::to_value(placement.1).unwrap_or(Value::Null),
            "write_memory_limit_bytes": writer_memory.bytes.get(),
            "write_memory_limit_source": writer_memory.origin(),
        },
        "selection": selection.json(),
        "contexts": contexts,
        "last_understanding_run": last_understanding_run,
        "last_injection": last_injection,
    });
    if let Some(fabric) = fabric {
        value["fabric"] = fabric;
    }
    if let [only] = contexts.as_slice() {
        for key in ["context", "memory", "totals", "summary", "history"] {
            value[key] = only[key].clone();
        }
        // The single-Context answer also carries the graph totals at the top level, where the
        // record and understanding commands' callers read them.
        for key in ["goals", "decisions", "observations"] {
            value[key] = only["memory"][key].clone();
        }
    }
    print_json(&value);
    0
}

/// The test-build door that makes one named status read refuse (`CG_FAIL_STATUS_METRIC_READ`,
/// compiled only under the `status-fault-injection` feature); inert in every shipped build.
fn injected_read_failure(read: &str) -> context_graph::Result<()> {
    super::super::injected_status_read_failure(read)
        .map_err(|error| CgError::Engine(format!("{error:?}")))
}

/// A status read that failed after the store opened: a refusal naming the store, never an answer.
fn status_read_refusal(format: OutputFormat, db_path: &str, error: &CgError) -> i32 {
    let message = format!("the store at {db_path} was opened but a status read failed: {error}");
    refuse(
        format,
        1,
        json!({
            "ok": false,
            "state": "status_read_failed",
            "condition": "MemoryError",
            "store": {"db_path": db_path},
            "message": message,
        }),
        &format!("error (MemoryError): {message}"),
    )
}

/// Helper transcripts Claude holds for the bound project that a directory import leaves out.
fn helper_transcripts_not_imported(project: &Path) -> Option<usize> {
    let locator = ClaudeHistoryLocator::new(effective_history_root(
        &super::super::home_dir(),
        std::env::var_os("CLAUDE_CONFIG_DIR")
            .map(PathBuf::from)
            .as_deref(),
    ));
    locator
        .resolve_project(project)
        .ok()
        .map(|resolved| resolved.helper_transcripts_not_imported)
}

fn search_retry(context: &EventHistoryContextStatus) -> String {
    import_retry_command(
        "<project-directory>",
        Some(&context.context_name),
        Some("wait"),
        context.search.work_policy,
        None,
    )
}

fn search_next(context: &EventHistoryContextStatus) -> Option<String> {
    match context.search.state {
        EventHistorySearchState::Blocked => Some(format!(
            "cg model fetch --defaults && {}",
            search_retry(context)
        )),
        EventHistorySearchState::Pending | EventHistorySearchState::Failed => {
            Some(search_retry(context))
        }
        _ => None,
    }
}

fn context_status_json(
    context: &EventHistoryContextStatus,
    helpers: Option<usize>,
    fabric: Option<&Value>,
) -> Value {
    let fabric_summary = fabric
        .map(|fabric| {
            format!(
                " | fabric: sharing {}, {} understanding input copies in push set",
                fabric["sharing"].as_str().unwrap_or("off"),
                fabric["inputs_in_push_set"]
            )
        })
        .unwrap_or_default();
    let delivery = context.delivery.as_ref().map(|delivery| {
        json!({
            "placement": placement_name(delivery.placement),
            "expected_policy": serde_json::to_value(delivery.expected_policy).unwrap_or(Value::Null),
            "authoritative_policy": delivery.authoritative_policy.map(|policy| serde_json::to_value(policy).unwrap_or(Value::Null)),
            "authoritative_hub_id": delivery.authoritative_hub_id,
            "state": serde_json::to_value(delivery.state).unwrap_or(Value::Null),
            "eligible": delivery.eligible,
            "accepted": delivery.accepted,
            "equivalent": delivery.equivalent,
            "pending": delivery.pending,
            "refused": delivery.refused,
        })
    });
    let memory = json!({
        "goals": context.memory.goals,
        "decisions": context.memory.decisions,
        "observations": context.memory.observations,
    });
    json!({
        "context": {"id": context.context_id.to_string(), "name": context.context_name},
        "memory": memory,
        "totals": memory,
        "summary": format!(
            "context: {} goals: {} decisions: {} observations: {} | history: {} events, search {}{fabric_summary}",
            context.context_name,
            context.memory.goals,
            context.memory.decisions,
            context.memory.observations,
            context.captured_events,
            state_name(context.search.state),
        ),
        "history": {
            "captured_events": context.captured_events,
            "capture": {"status": "complete", "events": context.captured_events},
            "sources": {
                "files": context.source_streams.len(),
                "sessions": context.sessions,
                "streams": context.source_streams.iter().map(|stream| json!({
                    "connector_id": stream.connector_id,
                    "source_instance_ref": stream.source_instance_ref,
                    "session_ref": stream.session_ref,
                    "adapter_version": stream.adapter_version,
                    "anchor_declaration_version": stream.anchor_declaration_version,
                    "events": stream.event_count,
                    "latest_seq": stream.latest_event_seq,
                })).collect::<Vec<_>>(),
            },
            "coverage": {
                "complete": context.coverage.complete,
                "opaque": context.coverage.opaque,
                "payload_refused": context.coverage.payload_refused,
                "source_declared_missing": context.coverage.source_declared_missing,
            },
            "coverage_gaps": context.coverage_gaps,
            "helper_transcripts_not_imported": helpers,
            "interpretation": {
                "active": context.active_interpretations.iter().map(|head| json!({
                    "connector_id": head.connector_id,
                    "generation_id": head.generation_id.to_string(),
                })).collect::<Vec<_>>(),
                "last_failure": context.last_interpretation_failure,
            },
            "delivery": delivery,
            "search": {
                "status": state_name(context.search.state),
                "work_policy": context.search.work_policy,
                "covered_events": context.search.covered_events,
                "pending_events": context.search.pending_events,
                "outside_scope_events": context.search.outside_scope_events,
                "outside_scope_indexed_events": context.search.outside_scope_indexed_events,
                "no_searchable_text_events": context.search.no_searchable_text_events,
                "active_generation_id": context.search.active_generation_id.map(|id| id.to_string()),
                "building_generation_id": context.search.building_generation_id.map(|id| id.to_string()),
                "building_profile": context.search.building_profile,
                "building_covered_events": context.search.building_covered_events,
                "building_pending_events": context.search.building_pending_events,
                "embedding_space_id": context.search.embedding_space_id,
                "renderer": {
                    "id": context.search.renderer_id,
                    "version": context.search.renderer_version,
                    "profile": context.search.profile,
                },
                "live_projection_lease": context.search.live_projection_lease,
                "last_failure": context.search.last_failure,
                "next": search_next(context),
            },
        },
    })
}

fn render_status_text(
    selection: &Selection,
    set: Option<&EventHistoryStatusSet>,
    placement: EventPlacementState,
    helpers: Option<usize>,
) {
    let Some(set) = set.filter(|set| !set.contexts.is_empty()) else {
        match &selection.missing_default {
            Some(name) => println!(
                "context: none — no project binding here and no \"{name}\" Context exists; import a project with `cg import <project-directory>`"
            ),
            None => println!("contexts: none"),
        }
        println!("event placement: {}", placement_name(placement));
        return;
    };
    for (index, context) in set.contexts.iter().enumerate() {
        if index > 0 {
            println!();
        }
        println!(
            "context: {} ({})",
            context.context_name,
            selection.source.label()
        );
        println!();
        println!("memory");
        println!("  goals: {}", grouped(context.memory.goals));
        println!("  decisions: {}", grouped(context.memory.decisions));
        println!("  observations: {}", grouped(context.memory.observations));
        println!();
        println!("history");
        println!(
            "  sources: {} files / {} sessions",
            grouped(context.source_streams.len() as u64),
            grouped(context.sessions)
        );
        println!(
            "  capture: complete — {} events",
            grouped(context.captured_events)
        );
        println!(
            "  coverage: {} complete / {} opaque / {} payload-refused / {} source-declared-missing",
            grouped(context.coverage.complete),
            grouped(context.coverage.opaque),
            grouped(context.coverage.payload_refused),
            grouped(context.coverage.source_declared_missing)
        );
        println!("  event placement: {}", placement_name(placement));
        if let Some(helpers) = helpers {
            println!("  helper transcripts: {helpers} not imported");
        }
        let search = &context.search;
        let line = match search.state {
            EventHistorySearchState::Ready => {
                format!("ready — {} events covered", grouped(search.covered_events))
            }
            EventHistorySearchState::Running => format!(
                "running in background — {} events covered / {} pending",
                grouped(search.covered_events),
                grouped(search.pending_events)
            ),
            EventHistorySearchState::Blocked => format!(
                "blocked — local Nomic artifact is not present; {} events pending",
                grouped(search.pending_events)
            ),
            EventHistorySearchState::Failed => format!(
                "failed — {} events pending; last error: {}",
                grouped(search.pending_events),
                search.last_failure.as_deref().unwrap_or("unknown")
            ),
            EventHistorySearchState::Pending => format!(
                "pending — no projection holds a live lease; {} events covered / {} pending",
                grouped(search.covered_events),
                grouped(search.pending_events)
            ),
        };
        println!("  search: {line}");
        if search.outside_scope_events > 0 || search.no_searchable_text_events > 0 {
            println!(
                "  search scope: {} outside current build scope ({} already indexed) / {} with no searchable text",
                grouped(search.outside_scope_events),
                grouped(search.outside_scope_indexed_events),
                grouped(search.no_searchable_text_events)
            );
        }
        if let Some(generation_id) = search.building_generation_id {
            println!(
                "  building search generation {generation_id}: {} events covered / {} pending",
                grouped(search.building_covered_events),
                grouped(search.building_pending_events)
            );
        }
        if let Some(profile) = &search.building_profile {
            println!(
                "  building passage profile: target {} tokens / hard limit {} tokens",
                profile.target_passage_tokens, profile.hard_passage_tokens
            );
        }
        if search.state != EventHistorySearchState::Failed
            && let Some(failure) = &search.last_failure
        {
            println!("  last search attempt failed: {failure}");
        }
        if let Some(next) = search_next(context) {
            println!("  next: {next}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// cg find
// ---------------------------------------------------------------------------------------------

pub(crate) fn handle_find(args: &[String], read_memory_limit_bytes: std::num::NonZeroUsize) -> i32 {
    let format = super::super::output_format(args);
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{FIND_HELP}");
        return 0;
    }
    let parsed = match parse(
        args,
        &["--context", "--limit", "--format"],
        &["--all-contexts"],
    ) {
        Ok(parsed) => parsed,
        Err(message) => return usage(format, "find", &message),
    };
    let query = match parsed.positionals.as_slice() {
        [] => return usage(format, "find", "find requires a question"),
        [query] => query.clone(),
        [_, extra, ..] => {
            return usage(
                format,
                "find",
                &format!("find takes one quoted question; unexpected `{extra}`"),
            );
        }
    };
    if query.trim().is_empty() {
        return usage(
            format,
            "find",
            "an empty query finds nothing; ask a question",
        );
    }
    let limit = match parsed.value("--limit") {
        None => context_graph::EVENT_SEARCH_DEFAULT_LIMIT,
        Some(raw) => match raw.parse::<u32>() {
            Ok(limit) if (1..=context_graph::EVENT_SEARCH_MAX_LIMIT).contains(&limit) => limit,
            _ => {
                return usage(
                    format,
                    "find",
                    &format!(
                        "--limit `{raw}` must be a whole number from 1 to {}",
                        context_graph::EVENT_SEARCH_MAX_LIMIT
                    ),
                );
            }
        },
    };
    if parsed.value("--context").is_some() && parsed.has("--all-contexts") {
        return usage(
            format,
            "find",
            "--context and --all-contexts are mutually exclusive",
        );
    }
    let root = super::super::data_root();
    let mut reader = match reader_or_refusal(format, &root, true, read_memory_limit_bytes) {
        Ok(reader) => reader,
        Err(code) => return code,
    };
    let selection = match resolve_selection(
        &reader,
        &root,
        parsed.value("--context"),
        parsed.has("--all-contexts"),
        parsed.value("--tenant-id"),
    ) {
        Ok(selection) => selection,
        Err(error) => return selection_refusal(format, "find", error),
    };
    let Ok(selector) = ContextSelector::new(selection.contexts.iter().map(|(id, _)| *id)) else {
        return no_matching_history(format, &selection);
    };
    if search_artifact_available()
        && let Ok(registration) = search_registration()
    {
        let _ = reader.attach_event_search_embedder_registrations(vec![registration]);
    }
    let filter = EventSearchFilter {
        limit: NonZeroU32::new(limit).expect("validated non-zero"),
        ..EventSearchFilter::default()
    };
    let result = match reader.find_event_history(selector, &query, filter) {
        Ok(result) => result,
        Err(CgError::HistorySearchNotReady { .. }) => {
            let selector = ContextSelector::new(selection.contexts.iter().map(|(id, _)| *id))
                .expect("the selected Context set was non-empty before the search");
            let statuses = match reader.event_history_status(selector) {
                Ok(statuses) => statuses.contexts,
                Err(error) => return library_refusal(format, &error),
            };
            // A worker can finish after the rejected search but before this status read. Keep the
            // original selected Context actions in that race rather than rendering an empty name
            // list and a null retry.
            let not_ready = statuses
                .iter()
                .filter(|context| context.search.state != EventHistorySearchState::Ready)
                .collect::<Vec<_>>();
            let actions = not_ready
                .iter()
                .copied()
                .chain(not_ready.is_empty().then_some(()).into_iter().flat_map(|_| statuses.iter()))
                .map(|context| {
                    json!({
                        "context": {"id": context.context_id.to_string(), "name": context.context_name},
                        "retry": search_retry(context),
                    })
                })
                .collect::<Vec<_>>();
            let names = not_ready
                .iter()
                .copied()
                .chain(
                    not_ready
                        .is_empty()
                        .then_some(())
                        .into_iter()
                        .flat_map(|_| statuses.iter()),
                )
                .map(|context| context.context_name.clone())
                .collect::<Vec<_>>();
            let retry = if actions.len() == 1 {
                actions[0]["retry"].clone()
            } else {
                Value::Null
            };
            let message = if actions.len() == 1 {
                format!(
                    "captured history in {} is not searchable yet; nothing was searched. Make it searchable with `{}` or wait for the background worker (`cg status`)",
                    names.join(", "),
                    actions[0]["retry"]
                        .as_str()
                        .expect("retry action is a string"),
                )
            } else {
                format!(
                    "captured history in {} is not searchable yet; nothing was searched. Each Context action carries that Context's persisted search policy, so run each action or wait for the background worker (`cg status`)",
                    names.join(", "),
                )
            };
            return refuse(
                format,
                1,
                json!({
                    "ok": false,
                    "state": "history_search_not_ready",
                    "selection": selection.json(),
                    "not_ready_contexts": names,
                    "retry": retry,
                    "retry_actions": actions,
                    "message": message,
                }),
                &format!("error: {message}"),
            );
        }
        Err(CgError::IncomparableEmbeddingSpaces {
            embedding_space_ids,
        }) => {
            let message = "the selected Contexts were indexed by different search models and cannot be ranked together; search them separately or rebuild their search indexes";
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "incomparable_embedding_spaces", "embedding_space_ids": embedding_space_ids, "message": message}),
                &format!("error: {message}"),
            );
        }
        Err(CgError::UnknownEmbeddingSpace { .. }) => {
            let message = "the local Nomic search model is not installed, so the question cannot be embedded; run `cg model fetch --defaults`";
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "search_blocked", "retry": "cg model fetch --defaults", "message": message}),
                &format!("error: {message}"),
            );
        }
        Err(error) => return library_refusal(format, &error),
    };

    let omitted = result
        .coverage
        .iter()
        .filter(|coverage| coverage.state == EventSearchContextState::NotReady)
        .map(|coverage| json!({"id": coverage.context_id.to_string(), "name": selection.name_of(coverage.context_id)}))
        .collect::<Vec<_>>();
    if format == OutputFormat::Text {
        for coverage in &result.coverage {
            let name = selection.name_of(coverage.context_id);
            match coverage.state {
                EventSearchContextState::NotReady => println!(
                    "note: \"{name}\" has captured history that is not searchable yet and was not searched"
                ),
                EventSearchContextState::Ready if coverage.pending_event_count > 0 => println!(
                    "note: \"{name}\" has {} newer captured events that search does not cover yet",
                    grouped(coverage.pending_event_count)
                ),
                _ => {}
            }
            if coverage.outside_scope_event_count > 0 {
                println!(
                    "note: \"{name}\" has {} events outside the current build scope; {} of them remain indexed",
                    grouped(coverage.outside_scope_event_count),
                    grouped(coverage.outside_scope_indexed_event_count)
                );
            }
        }
        if result.hits.is_empty() {
            println!("no matching history");
        }
        for hit in &result.hits {
            let date = hit
                .first_occurred_at
                .map(|time| time.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "undated".to_string());
            println!(
                "{}  {}  {date}  session {}",
                hit.passage_id,
                selection.name_of(hit.context_id),
                hit.session_ref
            );
            println!("{}", hit.text.trim_end());
            println!();
        }
        return 0;
    }
    let value = json!({
        "ok": true,
        "state": if result.hits.is_empty() { "no_matching_history" } else { "ok" },
        "query": query,
        "selection": selection.json(),
        "selected_context_ids": result.selected_context_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "searched_context_ids": result.searched_context_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "coverage": result.coverage.iter().map(|coverage| json!({
            "context": {"id": coverage.context_id.to_string(), "name": selection.name_of(coverage.context_id)},
            "state": match coverage.state {
                EventSearchContextState::Empty => "empty",
                EventSearchContextState::NotReady => "not_ready",
                EventSearchContextState::Ready => "ready",
            },
            "generation_id": coverage.generation_id.map(|id| id.to_string()),
            "captured_events": coverage.captured_event_count,
            "searchable_events": coverage.searchable_event_count,
            "pending_events": coverage.pending_event_count,
            "outside_scope_events": coverage.outside_scope_event_count,
            "outside_scope_indexed_events": coverage.outside_scope_indexed_event_count,
            "no_searchable_text_events": coverage.no_searchable_text_event_count,
            "source_watermark": coverage.source_watermark.as_ref().map(|watermark| json!({
                "event_count": watermark.event_count,
                "ordered_source_set_sha256": watermark.ordered_source_set_sha256,
            })),
        })).collect::<Vec<_>>(),
        "omitted_contexts": omitted,
        "hits": result.hits.iter().map(|hit| hit_json(hit, &selection)).collect::<Vec<_>>(),
    });
    print_json(&value);
    0
}

fn no_matching_history(format: OutputFormat, selection: &Selection) -> i32 {
    if format == OutputFormat::Text {
        println!("no matching history");
    } else {
        print_json(&json!({
            "ok": true,
            "state": "no_matching_history",
            "selection": selection.json(),
            "coverage": [],
            "hits": [],
        }));
    }
    0
}

fn source_json(source: &PassageEventSource) -> Value {
    json!({
        "event_id": source.event_id.to_string(),
        "context_id": source.context_id.to_string(),
        "connector_id": source.connector_id,
        "session_ref": source.session_ref,
        "source_instance_ref": source.source_instance_ref,
        "source_order": source.source_order,
        "event_start_byte": source.event_start_byte,
        "event_end_byte": source.event_end_byte,
        "passage_start_byte": source.passage_start_byte,
        "passage_end_byte": source.passage_end_byte,
    })
}

fn hit_json(hit: &EventSearchHit, selection: &Selection) -> Value {
    json!({
        "passage_id": hit.passage_id.to_string(),
        "context_id": hit.context_id.to_string(),
        "context_name": selection.name_of(hit.context_id),
        "connector_id": hit.connector_id,
        "session_ref": hit.session_ref,
        "source_instance_ref": hit.source_instance_ref,
        "first_event_seq": hit.first_event_seq,
        "last_event_seq": hit.last_event_seq,
        "first_occurred_at": hit.first_occurred_at,
        "last_occurred_at": hit.last_occurred_at,
        "text": hit.text,
        "cosine_similarity": hit.cosine_similarity,
        "applied_filter": {
            "connector_id": hit.applied_filter.connector_id,
            "session_ref": hit.applied_filter.session_ref,
            "source_instance_ref": hit.applied_filter.source_instance_ref,
            "limit": hit.applied_filter.limit.get(),
        },
        "embedding_space_id": hit.embedding_space_id,
        "generation_id": hit.generation_id.to_string(),
        "source_links": hit.sources.iter().map(source_json).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------------------------
// cg show
// ---------------------------------------------------------------------------------------------

pub(crate) fn handle_show(args: &[String], read_memory_limit_bytes: std::num::NonZeroUsize) -> i32 {
    let format = super::super::output_format(args);
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{SHOW_HELP}");
        return 0;
    }
    let parsed = match parse(
        args,
        &["--context", "--tenant-id", "--format"],
        &["--all-contexts"],
    ) {
        Ok(parsed) => parsed,
        Err(message) => return usage(format, "show", &message),
    };
    let raw = match parsed.positionals.as_slice() {
        [] => return usage(format, "show", "show requires a passage id from `cg find`"),
        [raw] => raw.clone(),
        [_, extra, ..] => {
            return usage(format, "show", &format!("unexpected argument `{extra}`"));
        }
    };
    let Ok(uuid) = uuid::Uuid::parse_str(raw.trim()) else {
        return usage(
            format,
            "show",
            &format!("`{raw}` is not a passage id; use the passage_id `cg find` printed"),
        );
    };
    let passage_id = EventSearchPassageId::from(uuid);
    if parsed.value("--context").is_some() && parsed.has("--all-contexts") {
        return usage(
            format,
            "show",
            "--context and --all-contexts are mutually exclusive",
        );
    }
    let root = super::super::data_root();
    let reader = match reader_or_refusal(format, &root, true, read_memory_limit_bytes) {
        Ok(reader) => reader,
        Err(code) => return code,
    };
    let selection = match resolve_selection(
        &reader,
        &root,
        parsed.value("--context"),
        parsed.has("--all-contexts"),
        parsed.value("--tenant-id"),
    ) {
        Ok(selection) => selection,
        Err(error) => return selection_refusal(format, "show", error),
    };
    let passage = match ContextSelector::new(selection.contexts.iter().map(|(id, _)| *id)) {
        Ok(selector) => match reader.get_event_search_passage(selector, passage_id) {
            Ok(passage) => passage,
            Err(error) => return library_refusal(format, &error),
        },
        Err(_) => None,
    };
    let Some(passage) = passage else {
        let message = format!(
            "no passage {raw} in the selected Contexts; a handle from a wider search needs its Context (--context NAME) or --all-contexts"
        );
        return refuse(
            format,
            1,
            json!({"ok": false, "state": "passage_not_found", "passage_id": raw, "selection": selection.json(), "message": message}),
            &format!("error: {message}"),
        );
    };
    let mut events = Vec::with_capacity(passage.sources.len());
    for source in &passage.sources {
        match reader.graph().get_event(source.context_id, source.event_id) {
            Ok(Some(event)) => events.push(event),
            Ok(None) => {
                return library_refusal(
                    format,
                    &CgError::Engine(format!(
                        "passage {} links event {} that is not stored",
                        passage.id, source.event_id
                    )),
                );
            }
            Err(error) => return library_refusal(format, &error),
        }
    }
    if format == OutputFormat::Text {
        render_show_text(&passage, &events, &selection);
        return 0;
    }
    print_json(&json!({
        "ok": true,
        "state": "ok",
        "selection": selection.json(),
        "passage": passage_json(&passage, &selection),
        "events": events.iter().map(event_json).collect::<Vec<_>>(),
    }));
    0
}

fn passage_json(passage: &EventSearchPassage, selection: &Selection) -> Value {
    json!({
        "passage_id": passage.id.to_string(),
        "context_id": passage.context_id.to_string(),
        "context_name": selection.name_of(passage.context_id),
        "generation_id": passage.generation_id.to_string(),
        "connector_id": passage.connector_id,
        "session_ref": passage.session_ref,
        "source_instance_ref": passage.source_instance_ref,
        "ordinal": passage.ordinal,
        "first_event_seq": passage.first_event_seq,
        "last_event_seq": passage.last_event_seq,
        "first_occurred_at": passage.first_occurred_at,
        "last_occurred_at": passage.last_occurred_at,
        "text": passage.text,
        "content_sha256": passage.content_sha256,
        "embedding_id": passage.embedding_id.to_string(),
        "embedding_space_id": passage.embedding_space_id,
        "dimension": passage.dimension,
        "is_current": passage.is_current,
        "source_links": passage.sources.iter().map(source_json).collect::<Vec<_>>(),
    })
}

fn event_json(event: &ConversationEvent) -> Value {
    json!({
        "id": event.id.to_string(),
        "context_id": event.context_id.to_string(),
        "connector_id": event.connector_id,
        "adapter_version": event.adapter_version,
        "anchor_declaration_version": event.anchor_declaration_version,
        "source_instance_ref": event.source_instance_ref,
        "session_ref": event.session_ref,
        "source_event_key": event.source_event_key,
        "source_position": event.source_position,
        "seq": event.seq,
        "content_hash": event.content_hash,
        "anchor_count": event.anchor_count,
        "anchor_set_hash": event.anchor_set_hash,
        "occurred_at": event.occurred_at,
        "author": event.author,
        "kind": event.kind,
        "render_role": event.render_role,
        "interaction_ref": event.interaction_ref,
        "text": event.text,
        "payload": event.payload,
        "anchors": event.anchors,
        "source_remainder": event.source_remainder,
        "coverage": event.coverage,
        "active_interpretation_version": Value::Null,
    })
}

fn render_show_text(
    passage: &EventSearchPassage,
    events: &[ConversationEvent],
    selection: &Selection,
) {
    let date = passage
        .first_occurred_at
        .map(|time| time.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "undated".to_string());
    println!(
        "{}  {}  {date}  session {}",
        passage.id,
        selection.name_of(passage.context_id),
        passage.session_ref
    );
    for event in events {
        println!();
        let time = event
            .occurred_at
            .map(|time| time.to_rfc3339())
            .unwrap_or_else(|| "no source time".to_string());
        println!("#{} {} ({}) — {time}", event.seq, event.author, event.kind);
        if let Some(text) = &event.text {
            println!("{}", text.trim_end());
        }
        if event.payload != json!({}) {
            println!(
                "payload: {}",
                serde_json::to_string(&event.payload).unwrap_or_default()
            );
        }
    }
}

const STATUS_HELP: &str =
    "Usage: cg status [--context NAME[,NAME...] | --all-contexts] [--format json|text]

Example:
  cd ~/projects/my-app && cg status --format text

Reads this machine's Context Graph store without changing it: graph memory totals and imported
conversation history — captured events, coverage, event placement and whether semantic search has
caught up. It reaches the store at $CG_DATA_ROOT/context-graph.db.

With no selector it uses the Context linked to the current project (or a parent directory);
--context names one or more Contexts, --all-contexts selects every Context in the store. Each
selected Context is reported separately.

Search is exactly one of: ready (caught up), running (a projection holds a live lease), pending
(no worker is running), blocked (the local Nomic model is not installed) or failed.
A background write that exceeds its request's persisted ceiling is failed with the needed and
allowed memory, even if the store could not record that failure. Retry import with an affordable
--write-memory-limit-bytes value.

A store that does not exist is reported as store_not_found (exit 1) and nothing is created. A
store that exists and cannot be read is reported by its typed condition, never as an empty store.
status never opens the store for writing, so it is never refused as busy. The commands that write
— `cg import` and `cg forget` — refuse a store another process is writing as store busy (exit 4),
and a store direct readers are holding under condition StoreHeldByReaders, naming each
verified reader by process id and name and disclosing any holder the store observed and could not
identify.
Model connections are reported by `cg model status`.";

const FIND_HELP: &str = "Usage: cg find <question> [--context NAME[,NAME...] | --all-contexts] [--limit N] [--format json|text]

Example:
  cg find \"why did we choose Iroh instead of NATS?\" --format text

Searches imported conversation history for passages that match the question and returns the
passages — never a generated answer. It reaches the history of the Context linked to the current
project; --context searches several named Contexts and --all-contexts every Context in this store,
ranked together. Each result carries its Context and a passage_id to pass to `cg show`.

find never starts or waits for search indexing; history that is not searchable yet is named as
omitted. The default limit is 10 and the maximum is 100.";

const SHOW_HELP: &str =
    "Usage: cg show <passage-id> [--context NAME[,NAME...] | --all-contexts] [--format json|text]

Example:
  cg show 0190a3b2-6c1d-7c3e-9f00-4b5c6d7e8f90 --format text

Shows one passage returned by `cg find` and the exact stored events behind it, in order, read
from this machine's Context Graph store — the original session file is not needed. The passage must
belong to the Context linked to the current project, or to a Context named with --context or
--all-contexts; a passage id alone never widens access.";
