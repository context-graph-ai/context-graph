//! `cg forget --session <ref>` / `cg forget --context <name>`.

use context_graph::{CgError, ContextSelector, ForgetSelection, Store};
use serde_json::{Value, json};

use super::{
    OutputFormat, SelectionError, Terminal, configured_passage_profile, context_names, db_path,
    grouped, library_refusal, parse, print_json, refuse, resolve_selection,
    search_artifact_available, selection_refusal, store_config, store_not_found, usage,
};

pub(crate) fn handle_forget(
    args: &[String],
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> i32 {
    let format = super::super::output_format(args);
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{FORGET_HELP}");
        return 0;
    }
    let parsed = match parse(
        args,
        &["--session", "--context", "--tenant-id", "--format"],
        &["--yes"],
    ) {
        Ok(parsed) => parsed,
        Err(message) => return usage(format, "forget", &message),
    };
    if let Some(extra) = parsed.positionals.first() {
        return usage(format, "forget", &format!("unexpected argument `{extra}`"));
    }
    let session = parsed.value("--session").map(str::trim).map(str::to_string);
    if session.as_deref().is_some_and(str::is_empty) {
        return usage(format, "forget", "--session requires a session reference");
    }
    let explicit = parsed.value("--context");
    if session.is_none() && explicit.is_none() {
        return usage(
            format,
            "forget",
            "forget requires exactly one selector: --session <ref> or --context <name>",
        );
    }
    if let Some(raw) = explicit {
        match context_names(raw) {
            Ok(names) if names.len() == 1 => {}
            Ok(_) => {
                return usage(
                    format,
                    "forget",
                    &format!("--context `{raw}` must name exactly one Context"),
                );
            }
            Err(message) => return usage(format, "forget", &message),
        }
    }
    let root = super::super::data_root();
    let reader = match super::open_reader(&root, read_memory_limit_bytes) {
        Ok(reader) => reader,
        Err(CgError::StoreMissing { .. }) => return store_not_found(format, &db_path(&root), true),
        Err(error) => return library_refusal(format, &error),
    };
    let selection =
        match resolve_selection(&reader, &root, explicit, false, parsed.value("--tenant-id")) {
            Ok(selection) => selection,
            Err(error) => return selection_refusal(format, "forget", error),
        };
    let Some((context_id, context_name)) = selection.contexts.first().cloned() else {
        return selection_refusal(
            format,
            "forget",
            SelectionError::UnknownContext(vec![
                selection
                    .missing_default
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
            ]),
        );
    };
    let status = match reader.event_history_status(ContextSelector::one(context_id)) {
        Ok(mut set) => set.contexts.remove(0),
        Err(error) => return library_refusal(format, &error),
    };
    let (selector, events, retry) = match &session {
        Some(session_ref) => {
            let streams = status
                .source_streams
                .iter()
                .filter(|stream| stream.session_ref == *session_ref)
                .collect::<Vec<_>>();
            let connector_id = streams
                .first()
                .map(|stream| stream.connector_id.clone())
                .unwrap_or_else(|| cg_connector_claude_code::CONNECTOR_ID.to_string());
            let events = streams.iter().map(|stream| stream.event_count).sum::<u64>();
            let retry = match explicit {
                Some(name) => format!("cg forget --session {session_ref} --context {name} --yes"),
                None => format!("cg forget --session {session_ref} --yes"),
            };
            (
                ForgetSelection::Session {
                    context_id,
                    connector_id,
                    session_ref: session_ref.clone(),
                },
                events,
                retry,
            )
        }
        None => (
            ForgetSelection::Context { context_id },
            status.captured_events,
            format!("cg forget --context {context_name} --yes"),
        ),
    };
    let selection_json = match &session {
        Some(session_ref) => json!({
            "selector": "session",
            "session_ref": session_ref,
            "context": {"id": context_id.to_string(), "name": context_name},
        }),
        None => json!({
            "selector": "context",
            "context": {"id": context_id.to_string(), "name": context_name},
        }),
    };

    // Status counts sessions with captured events only. The planner also selects checkpoint and
    // locator sessions whose sources had no decodable records, so confirmation uses that count.
    let sessions = match reader.forget_selection_session_count(&selector) {
        Ok(count) => count,
        Err(error) => return library_refusal(format, &error),
    };
    drop(reader);

    if sessions == 0 {
        if format == OutputFormat::Text {
            println!("nothing to forget in \"{context_name}\"");
        } else {
            print_json(&json!({
                "ok": true,
                "state": "nothing_to_forget",
                "selection": selection_json,
                "counts": {"sessions": 0, "events": 0, "anchors": 0, "passages": 0, "vectors": 0},
                "tables": [],
                "remaining_events": status.captured_events,
                "search": {"status": "nothing_to_index", "pending_events": 0},
                "retry": Value::Null,
            }));
        }
        return 0;
    }

    if !parsed.has("--yes") {
        let prompt =
            format!("Forget {sessions} sessions ({events} events) from \"{context_name}\"? [y/N] ");
        let confirmed = match Terminal::open() {
            Some(mut terminal) => terminal.confirm(&prompt),
            None => {
                let message = format!(
                    "forgetting deletes {sessions} sessions ({} events) from \"{context_name}\" and needs confirmation; nothing was changed. Run `{retry}`",
                    grouped(events)
                );
                return refuse(
                    format,
                    2,
                    json!({
                        "ok": false,
                        "state": "confirmation_required",
                        "selection": selection_json,
                        "counts": {"sessions": sessions, "events": events},
                        "retry": retry,
                        "message": message,
                    }),
                    &format!("error: {message}"),
                );
            }
        };
        if !confirmed {
            let message = "not confirmed; nothing was changed".to_string();
            return refuse(
                format,
                2,
                json!({
                    "ok": false,
                    "state": "confirmation_required",
                    "selection": selection_json,
                    "retry": retry,
                    "message": message,
                }),
                &message,
            );
        }
    }

    let store = match Store::open(store_config(&root, None, read_memory_limit_bytes)) {
        Ok(store) => store,
        Err(error @ CgError::WriteDidNotFitTheMemoryLimit { .. }) => {
            return erasure_memory_refusal(format, &selection_json, &error);
        }
        Err(error) => return library_refusal(format, &error),
    };
    #[cfg(all(feature = "event-search-test-support", unix))]
    pause_before_forget_for_test();
    let receipt = match store.forget_event_history(selector) {
        Ok(receipt) => receipt,
        Err(CgError::EventHistoryForgetRequiresEdgeDiscard { .. }) => {
            let message = "this store places events at a tenant hub; forgetting its local copies needs the engine's edge-local discard, which this build does not issue. Nothing was changed";
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "discard_denied", "selection": selection_json, "message": message}),
                &format!("error: {message}"),
            );
        }
        Err(error @ CgError::WriteDidNotFitTheMemoryLimit { .. }) => {
            return erasure_memory_refusal(format, &selection_json, &error);
        }
        Err(
            error @ (CgError::StoreLocked { .. }
            | CgError::StoreHeldByReaders { .. }
            | CgError::ReaderCoordinationUnavailable { .. }
            | CgError::NotFound { .. }),
        ) => return library_refusal(format, &error),
        Err(error) => {
            let message = format!("the store refused the erasure; no history was erased: {error}");
            return refuse(
                format,
                1,
                json!({"ok": false, "state": "purge_refused", "selection": selection_json, "message": message}),
                &format!("error: {message}"),
            );
        }
    };
    drop(store);
    let state = if receipt.changed {
        "forgotten"
    } else {
        "nothing_to_forget"
    };
    let remaining = status.captured_events.saturating_sub(receipt.events);
    // The erasure cleared the Context's search index, so the history that remains is re-indexed
    // the way a background import schedules it: the same detached one-shot worker.
    let search = if !receipt.changed || remaining == 0 {
        json!({"status": "nothing_to_index", "pending_events": 0})
    } else if !search_artifact_available() {
        json!({
            "status": "blocked",
            "pending_events": remaining,
            "reason": "the local Nomic search model is not installed",
            "next": "cg model fetch --defaults",
        })
    } else {
        match configured_passage_profile(&root) {
            Err(reason) => json!({
                "status": "rebuild_not_started",
                "pending_events": remaining,
                "reason": reason,
            }),
            Ok(profile) => {
                match super::worker::kick(&root, &profile, context_id, read_memory_limit_bytes) {
                    Ok(worker) => json!({
                        "status": "rebuild_scheduled",
                        "pending_events": remaining,
                        "worker": worker,
                    }),
                    Err(reason) => json!({
                        "status": "rebuild_not_started",
                        "pending_events": remaining,
                        "reason": reason,
                    }),
                }
            }
        }
    };
    if format == OutputFormat::Text {
        if receipt.changed {
            println!(
                "forgot {} sessions from \"{context_name}\": {} events, {} anchors, {} search passages and {} vectors deleted",
                receipt.sessions,
                grouped(receipt.events),
                grouped(receipt.anchors),
                grouped(receipt.passages),
                grouped(receipt.vectors)
            );
            match search["status"].as_str() {
                Some("rebuild_scheduled") => println!(
                    "search: rebuild scheduled in the background — the remaining {} events are being re-indexed; `cg status` reports when it is ready",
                    grouped(remaining)
                ),
                Some("blocked") => println!(
                    "search: blocked — the remaining {} events need re-indexing and the local Nomic artifact is not present; next: cg model fetch --defaults",
                    grouped(remaining)
                ),
                Some("rebuild_not_started") => println!(
                    "search: the remaining {} events need re-indexing and the background worker could not be started ({}); `cg status` shows them as pending",
                    grouped(remaining),
                    search["reason"].as_str().unwrap_or("unknown reason")
                ),
                _ => {}
            }
        } else {
            println!("nothing to forget in \"{context_name}\"");
        }
        return 0;
    }
    let value = json!({
        "ok": true,
        "state": state,
        "selection": selection_json,
        "counts": {
            "sessions": receipt.sessions,
            "events": receipt.events,
            "anchors": receipt.anchors,
            "passages": receipt.passages,
            "vectors": receipt.vectors,
        },
        "tables": receipt.tables.iter().map(|table| json!({"table": table.table, "rows": table.rows})).collect::<Vec<_>>(),
        "remaining_events": remaining,
        "search": search,
        "retry": Value::Null,
    });
    print_json(&value);
    0
}

fn erasure_memory_refusal(format: OutputFormat, selection: &Value, error: &CgError) -> i32 {
    let CgError::WriteDidNotFitTheMemoryLimit {
        requested_bytes,
        limit_bytes,
    } = error
    else {
        unreachable!("erasure memory refusal requires a typed memory condition")
    };
    let message = format!(
        "erasure needed {} of working memory but cg is allowed {} on this machine; no history was erased. {}",
        context_graph::write_budget::human_bytes(*requested_bytes as u64),
        context_graph::write_budget::human_bytes(*limit_bytes as u64),
        context_graph::write_budget::how_to_raise_the_writer_budget()
    );
    refuse(
        format,
        1,
        json!({"ok": false, "state": "purge_refused", "condition": "WriteDidNotFitTheMemoryLimit", "selection": selection, "message": message}),
        &format!("error (WriteDidNotFitTheMemoryLimit): {message}"),
    )
}

#[cfg(all(feature = "event-search-test-support", unix))]
fn pause_before_forget_for_test() {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    let Some(socket) = std::env::var_os("CG_FORGET_TEST_BEFORE_PURGE_SOCKET") else {
        return;
    };
    let mut stream = UnixStream::connect(socket).expect("connect forget test barrier");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(60)))
        .expect("bound forget test barrier");
    stream.write_all(b"r").expect("report writable open");
    let mut release = [0_u8; 1];
    stream
        .read_exact(&mut release)
        .expect("wait for purge release");
}

const FORGET_HELP: &str =
    "Usage: cg forget --session <ref> [--context NAME] [--yes] [--format json|text]
       cg forget --context NAME [--yes] [--format json|text]

Examples:
  cg forget --session 5f1c2d3e-0000-4000-8000-000000000001
  cg forget --context my-app --yes

Deletes imported conversation history from this machine's Context Graph store. --session deletes
one imported session of the Context linked to the current project (or the Context named with
--context); --context alone deletes every imported session in that Context. The events, their
exact anchors, file checkpoints and every search passage and vector derived from them go in one
authoritative erasure, or nothing changes. Erasure uses the writer limit shown by `cg status`;
it has no capture proof floor. If erasure exceeds that limit, `purge_refused` reports
`WriteDidNotFitTheMemoryLimit` and leaves history unchanged. Set --write-memory-limit-bytes <BYTES>
for one attempt, or [write] memory_limit_bytes in cg/fabric.toml for every command.
A selection with no events,
checkpoints or source locators returns nothing_to_forget after read-only inspection, without
opening the store for writing. The Context and its project link remain, so the next import lands
in the same place. Forgetting one session also clears that Context's search index; forget then
schedules the re-indexing of remaining history in the background (or reports that the local model
must first be installed with `cg model fetch --defaults`).

On a terminal it asks `Forget <N> sessions (<M> events) from \"<context>\"? [y/N]` and Enter means
no. Without a terminal it requires --yes.";
