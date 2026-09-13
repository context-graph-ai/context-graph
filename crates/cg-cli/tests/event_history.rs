//! The `cg` event-history journey: import a project's Claude Code history, then find, show, status
//! and forget it — all through the compiled binary against isolated synthetic history.

mod event_support;

use std::fs;

use event_support::{Machine, fingerprint, str_at, u64_at, write_helper_transcript, write_session};
use serde_json::Value;

fn hits(value: &Value) -> Vec<Value> {
    value
        .get("hits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

#[test]
fn cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history() {
    let machine = Machine::without_model();
    let project = machine.project("ledger");
    let other = machine.project("other-ledger");
    let history = machine.history_root();
    write_session(
        &history,
        &project,
        "sess-one",
        &[("alpha question", "alpha answer")],
    );
    write_session(
        &history,
        &project,
        "sess-two",
        &[
            ("beta question", "beta answer"),
            ("gamma question", "gamma answer"),
        ],
    );
    write_session(
        &history,
        &other,
        "sess-other",
        &[("other project question", "other project answer")],
    );
    let helper = write_helper_transcript(&history, &project, "sess-one", "helper1", "helper text");
    fs::write(
        project.join("stray.jsonl"),
        format!(
            "{}\n",
            serde_json::json!({
                "type": "user",
                "uuid": "00000000-0000-4000-8000-0000000000aa",
                "sessionId": "stray",
                "cwd": project.display().to_string(),
                "message": {"role": "user", "content": "working tree file"}
            })
        ),
    )
    .expect("stray working-tree jsonl");

    let imported = machine.run(
        &project,
        &["import", ".", "--context", "ledger", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_deferred");
    assert_eq!(receipt.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(str_at(&receipt, "/selection/source"), "explicit");
    assert_eq!(str_at(&receipt, "/context/name"), "ledger");
    assert_eq!(
        u64_at(&receipt, "/sources/files"),
        2,
        "only this project's two session files"
    );
    assert_eq!(u64_at(&receipt, "/sources/sessions"), 2);
    assert_eq!(
        u64_at(&receipt, "/sources/helper_transcripts_not_imported"),
        1,
        "the helper transcript is counted, not silently dropped"
    );
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 6);
    assert_eq!(u64_at(&receipt, "/capture/local_embedding_calls"), 0);
    assert_eq!(u64_at(&receipt, "/capture/understanding_calls"), 0);
    assert_eq!(u64_at(&receipt, "/capture/hosted_provider_calls"), 0);
    assert_eq!(str_at(&receipt, "/search/status"), "pending");
    assert!(
        !imported.stdout.contains("working tree file")
            && !imported.stdout.contains("other project")
            && !imported.stdout.contains(&project.display().to_string()),
        "the receipt carries neither source text nor the project path: {}",
        imported.stdout
    );
    let context_id = str_at(&receipt, "/context/id").to_owned();

    // A symlink alias and an absolute path resolve the same project and the same binding.
    let link = machine.root.join("work").join("ledger-link");
    std::os::unix::fs::symlink(&project, &link).expect("project symlink");
    for target in [link.display().to_string(), project.display().to_string()] {
        let again = machine.run(&machine.root, &["import", &target, "--search", "defer"]);
        assert_eq!(again.code, 0, "{}", again.describe());
        let again = again.json();
        assert_eq!(str_at(&again, "/selection/source"), "project_binding");
        assert_eq!(str_at(&again, "/context/id"), context_id);
        assert_eq!(u64_at(&again, "/capture/events/inserted"), 0);
        assert_eq!(u64_at(&again, "/sources/files"), 2);
    }

    // Reads from a descendant directory reuse the project binding.
    let deep = project.join("src").join("deep");
    fs::create_dir_all(&deep).expect("descendant directory");
    let status = machine.run(&deep, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(str_at(&status, "/selection/source"), "project_binding");
    assert_eq!(str_at(&status, "/context/name"), "ledger");
    assert_eq!(u64_at(&status, "/history/captured_events"), 6);
    assert_eq!(u64_at(&status, "/history/sources/sessions"), 2);

    // A named helper transcript imports on its own, under its parent session.
    let named = machine.run(
        &machine.root,
        &[
            "import",
            &helper.display().to_string(),
            "--context",
            "ledger",
            "--search",
            "defer",
        ],
    );
    assert_eq!(named.code, 0, "{}", named.describe());
    let named = named.json();
    assert_eq!(u64_at(&named, "/capture/events/inserted"), 1);
    assert_eq!(str_at(&named, "/selection/source"), "explicit");
    let status = machine.run(&deep, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 7);
    assert_eq!(
        u64_at(&status, "/history/sources/sessions"),
        2,
        "the helper stream belongs to its parent session"
    );
}

#[test]
fn cg_import_project_without_matching_claude_history_creates_no_context_or_binding_and_reports_not_found()
 {
    let machine = Machine::without_model();
    let project = machine.project("no-history");
    let before = fingerprint(&machine.root);
    let output = machine.run(&project, &["import", ".", "--search", "defer"]);
    assert_eq!(output.code, 2, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(receipt.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(str_at(&receipt, "/state"), "project_history_not_found");
    assert!(
        !machine.data_root.exists(),
        "no data root or database was created"
    );
    assert_eq!(
        fingerprint(&machine.root),
        before,
        "no write anywhere on the machine"
    );
}

#[test]
fn noninteractive_unbound_project_import_returns_the_retry_without_writing() {
    let machine = Machine::without_model();
    let project = machine.project("harbor");
    write_session(
        &machine.history_root(),
        &project,
        "sess-harbor",
        &[("harbor question", "harbor answer")],
    );
    let output = machine.run(&project, &["import", "."]);
    assert_eq!(output.code, 2, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(str_at(&receipt, "/state"), "context_binding_required");
    assert_eq!(str_at(&receipt, "/suggested_context"), "harbor");
    let retry = str_at(&receipt, "/retry");
    assert!(
        retry.contains("cg import") && retry.contains("--context harbor"),
        "the retry is the exact command: {retry}"
    );
    assert!(
        receipt.get("capture").is_none(),
        "no fabricated capture progress"
    );
    assert!(
        !output.stdout.contains("Context name") && !output.stderr.contains("Context name"),
        "no prompt is written without a controlling terminal"
    );
    assert!(!machine.data_root.exists(), "nothing was written");
}

#[test]
fn a_bound_project_refuses_a_different_explicit_context_without_changing_either() {
    let machine = Machine::without_model();
    let project = machine.project("quarry");
    write_session(
        &machine.history_root(),
        &project,
        "sess-quarry",
        &[("quarry question", "quarry answer")],
    );
    let first = machine.run(
        &project,
        &["import", ".", "--context", "quarry", "--search", "defer"],
    );
    assert_eq!(first.code, 0, "{}", first.describe());
    let same = machine.run(
        &project,
        &["import", ".", "--context", "quarry", "--search", "defer"],
    );
    assert_eq!(
        same.code,
        0,
        "an explicit same name is accepted: {}",
        same.describe()
    );
    let conflict = machine.run(
        &project,
        &["import", ".", "--context", "mine", "--search", "defer"],
    );
    assert_eq!(conflict.code, 2, "{}", conflict.describe());
    assert_eq!(
        str_at(&conflict.json(), "/state"),
        "context_binding_conflict"
    );
    let other = machine.run(&project, &["status", "--context", "mine"]);
    assert_ne!(
        other.code,
        0,
        "the conflicting Context was never created: {}",
        other.describe()
    );
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(str_at(&status, "/context/name"), "quarry");
}

#[test]
fn import_usage_errors_name_the_token_and_do_no_work() {
    let machine = Machine::without_model();
    let project = machine.project("usage");
    let session = write_session(
        &machine.history_root(),
        &project,
        "sess-usage",
        &[("usage question", "usage answer")],
    );
    let session = session.display().to_string();
    for (args, token) in [
        (vec!["import", ".", "--bogus"], "--bogus"),
        (vec!["import", ".", "--search", "later"], "later"),
        (vec!["import", ".", "another"], "another"),
        (vec!["import", ".", "--context"], "--context"),
        (vec!["import", ".", "--understand-now"], "--understand-now"),
        (
            vec!["import", ".", "--chunk-tokens", "10"],
            "--chunk-tokens",
        ),
        (vec!["import", &session, "--search", "defer"], "--context"),
        (vec!["import", &session, "--context", "a,b"], "a,b"),
        (vec!["import", &session, "--all-contexts"], "--all-contexts"),
        (vec!["import", ".", "--session", "sess-usage"], "--session"),
    ] {
        let output = machine.run(&project, &args);
        assert_eq!(output.code, 2, "{args:?}: {}", output.describe());
        assert!(
            output.stdout.contains(token) || output.stderr.contains(token),
            "the usage error for {args:?} names {token}: {}",
            output.describe()
        );
    }
    assert!(
        !machine.data_root.exists(),
        "a usage error performs no work"
    );
}

#[test]
fn cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file()
 {
    let machine = Machine::without_model();
    let project = machine.project("partial");
    let history = machine.history_root();
    let bad = write_session(&history, &project, "sess-a-bad", &[("first", "second")]);
    let mut bytes = fs::read(&bad).expect("read bad session");
    bytes.extend_from_slice(b"{\"type\": \"user\", not json\n");
    fs::write(&bad, bytes).expect("corrupt the session tail");
    write_session(
        &history,
        &project,
        "sess-b-good",
        &[("good question", "good answer")],
    );

    let output = machine.run(
        &project,
        &["import", ".", "--context", "partial", "--search", "defer"],
    );
    assert_eq!(output.code, 1, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(str_at(&receipt, "/state"), "source_partial");
    assert_eq!(receipt.get("ok"), Some(&Value::Bool(false)));
    let failed = receipt
        .pointer("/capture/failed_sources")
        .and_then(Value::as_array)
        .expect("failed sources");
    assert_eq!(failed.len(), 1);
    assert_eq!(str_at(&failed[0], "/file"), "sess-a-bad.jsonl");
    assert_eq!(u64_at(&failed[0], "/position/line"), 3);
    assert!(!str_at(&failed[0], "/reason").is_empty());
    assert_eq!(str_at(&receipt, "/search/worker"), "not_started");
    assert_eq!(
        u64_at(&receipt, "/capture/events/inserted"),
        2,
        "every other file is captured; the failed file keeps only batches it committed"
    );
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 2);
}

#[test]
fn with_no_data_root_override_the_store_lives_at_dot_cg_under_home() {
    let machine = Machine::without_model();
    let project = machine.project("unrooted");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cg"))
        .args(["status"])
        .current_dir(&project)
        .env_remove("CG_DATA_ROOT")
        .env("HOME", &machine.home)
        .env("CG_MODEL_CACHE_ROOT", &machine.model_cache)
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("XDG_CACHE_HOME")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run the compiled cg binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    let value: Value = serde_json::from_str(&stdout).expect("one JSON object");
    assert_eq!(str_at(&value, "/state"), "store_not_found");
    assert_eq!(
        str_at(&value, "/store/db_path"),
        machine
            .home
            .join(".cg")
            .join("context-graph.db")
            .display()
            .to_string()
    );
    assert!(
        !machine.home.join(".cg").exists(),
        "a status read never creates the default data root"
    );
}

#[test]
fn find_in_a_context_with_no_imported_history_answers_no_matching_history() {
    let machine = Machine::without_model();
    let history = machine.history_root();
    let project = machine.project("pantry");
    write_session(
        &history,
        &project,
        "sess-pantry",
        &[(
            "Where do the spare jars go?",
            "Top shelf, behind the flour.",
        )],
    );
    let imported = machine.run(
        &project,
        &["import", ".", "--context", "pantry", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let forgotten = machine.run(&project, &["forget", "--context", "pantry", "--yes"]);
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());

    let found = machine.run(&project, &["find", "spare jars"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    let found = found.json();
    assert_eq!(str_at(&found, "/state"), "no_matching_history");
    assert!(hits(&found).is_empty());
}

#[test]
fn cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either()
 {
    let machine = Machine::without_model();
    let project = machine.project("absent");
    let db_path = machine.db_path().display().to_string();
    for prepare in [false, true] {
        if prepare {
            fs::create_dir_all(&machine.data_root).expect("an existing data root with no database");
        }
        let before = fingerprint(&machine.root);
        let json = machine.run(&project, &["status"]);
        assert_eq!(json.code, 1, "{}", json.describe());
        assert_eq!(
            json.json(),
            serde_json::json!({
                "ok": false,
                "state": "store_not_found",
                "store": {"db_path": db_path},
                "retry": "create_store_with_write_command"
            })
        );
        let text = machine.run(&project, &["status", "--format", "text"]);
        assert_eq!(text.code, 1);
        assert_eq!(text.stdout, "");
        assert_eq!(
            text.stderr,
            format!(
                "error: Context Graph store not found: {db_path}\nnext: create it with the intended write command; for project history run `cg import <project-directory>`\n"
            )
        );
        for args in [
            vec!["find", "anything"],
            vec!["show", "0190a3b2-0000-7000-8000-000000000000"],
        ] {
            let read = machine.run(&project, &args);
            assert_eq!(read.code, 1, "{args:?}: {}", read.describe());
            let value = read.json();
            assert_eq!(str_at(&value, "/state"), "store_not_found");
            assert!(value.get("hits").is_none(), "no fabricated hits");
            assert!(str_at(&value, "/next").contains("cg import"));
        }
        assert_eq!(
            fingerprint(&machine.root),
            before,
            "status, find and show created nothing"
        );
    }
}

#[test]
fn missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight()
{
    let machine = Machine::without_model();
    let project = machine.project("offline");
    write_session(
        &machine.history_root(),
        &project,
        "sess-offline",
        &[("offline question", "offline answer")],
    );
    let background = machine.run(&project, &["import", ".", "--context", "offline"]);
    assert_eq!(background.code, 1, "{}", background.describe());
    let receipt = background.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(str_at(&receipt, "/search/status"), "blocked");
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);
    assert_eq!(u64_at(&receipt, "/search/pending_events"), 2);
    assert!(str_at(&receipt, "/retry").contains("cg model fetch --defaults"));
    assert!(str_at(&receipt, "/retry").contains("--search wait"));
    assert!(
        fs::read_dir(&machine.model_cache)
            .expect("model cache")
            .next()
            .is_none(),
        "no model was downloaded"
    );

    let wait = machine.run(&project, &["import", ".", "--search", "wait"]);
    assert_eq!(wait.code, 1, "{}", wait.describe());
    assert_eq!(str_at(&wait.json(), "/state"), "captured_search_behind");

    let defer = machine.run(&project, &["import", ".", "--search", "defer"]);
    assert_eq!(defer.code, 0, "{}", defer.describe());
    assert_eq!(str_at(&defer.json(), "/state"), "captured_search_deferred");

    let status = machine.run(&project, &["status"]).json();
    assert_eq!(str_at(&status, "/history/search/status"), "blocked");
    assert_eq!(u64_at(&status, "/history/search/pending_events"), 2);
    let text = machine.run(&project, &["status", "--format", "text"]);
    assert!(
        text.stdout
            .contains("blocked — local Nomic artifact is not present"),
        "{}",
        text.describe()
    );
    assert!(text.stdout.contains("cg model fetch --defaults"));
}

#[test]
fn cg_import_search_wait_embedding_failure_returns_durable_capture_receipt_and_typed_behind_status()
{
    let mut machine = Machine::without_model();
    // An installed-looking artifact whose weights cannot load: the preflight accepts it, the
    // projection fails when it loads the model.
    let real = event_support::existing_nomic_cache_root().join(event_support::NOMIC_MODEL_ID);
    let broken = machine
        .root
        .join("broken-cache")
        .join(event_support::NOMIC_MODEL_ID);
    fs::create_dir_all(&broken).expect("broken cache");
    for file in ["config.json", "tokenizer.json", "REVISION"] {
        fs::copy(real.join(file), broken.join(file)).expect("copy artifact metadata");
    }
    fs::write(broken.join("model.safetensors"), vec![7_u8; 2_000_000]).expect("broken weights");
    machine.model_cache = machine.root.join("broken-cache");
    let project = machine.project("broken");
    write_session(
        &machine.history_root(),
        &project,
        "sess-broken",
        &[("broken question", "broken answer")],
    );
    let output = machine.run(
        &project,
        &["import", ".", "--context", "broken", "--search", "wait"],
    );
    assert_eq!(output.code, 1, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);
    assert_eq!(str_at(&receipt, "/search/status"), "failed");
    assert!(!str_at(&receipt, "/search/reason").is_empty());
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        2,
        "capture stays durable"
    );
    assert_ne!(str_at(&status, "/history/search/status"), "ready");
}

#[test]
fn find_and_show_parse_their_arguments_before_reading() {
    let machine = Machine::without_model();
    let project = machine.project("args");
    for (args, token) in [
        (vec!["find"], "find"),
        (vec!["find", ""], "query"),
        (vec!["find", "q", "--limit", "0"], "--limit"),
        (
            vec!["find", "q", "--context", "a", "--all-contexts"],
            "--all-contexts",
        ),
        (vec!["find", "q", "--bogus"], "--bogus"),
        (vec!["show"], "show"),
        (vec!["show", "not-a-passage-id"], "not-a-passage-id"),
        (vec!["forget"], "--session"),
        (vec!["forget", "--session", "x", "--bogus"], "--bogus"),
    ] {
        let output = machine.run(&project, &args);
        assert_eq!(output.code, 2, "{args:?}: {}", output.describe());
        assert!(
            output.stdout.contains(token) || output.stderr.contains(token),
            "{args:?} names {token}: {}",
            output.describe()
        );
    }
}

#[test]
fn event_history_journey_with_real_nomic_imports_finds_shows_and_forgets() {
    let machine = Machine::with_nomic();
    let history = machine.history_root();
    let atlas = machine.project("atlas");
    let garden = machine.project("garden");
    write_session(
        &history,
        &atlas,
        "sess-transport",
        &[(
            "Why did we pick Iroh instead of NATS for the transport?",
            "Iroh keeps a direct peer-to-peer path; NATS would add a broker deployment we must operate.",
        )],
    );
    write_session(
        &history,
        &atlas,
        "sess-ledger",
        &[(
            "How should the ledger reconcile foreign currency entries?",
            "Convert at the booking-date rate and keep the original amount beside it.",
        )],
    );
    write_session(
        &history,
        &garden,
        "sess-garden",
        &[(
            "When should the tomato seedlings go outside?",
            "After the last frost, once nights stay above ten degrees.",
        )],
    );

    let imported = machine.run(&atlas, &["import", ".", "--context", "atlas"]);
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_scheduled");
    assert_eq!(str_at(&receipt, "/search/status"), "scheduled");
    assert_eq!(u64_at(&receipt, "/search/pending_events"), 4);
    machine.wait_for_search_ready(&atlas, "atlas");
    // The detached worker catches up every pending Context on the data root; wait for it to exit
    // after its final re-check before capturing a Context that must stay unsearchable.
    machine.wait_for_search_worker_exit();
    let garden_import = machine.run(
        &garden,
        &["import", ".", "--context", "garden", "--search", "defer"],
    );
    assert_eq!(garden_import.code, 0, "{}", garden_import.describe());

    // An unchanged, caught-up rerun only verifies the committed prefixes.
    let rerun = machine.run(&atlas, &["import", ".", "--search", "wait"]);
    assert_eq!(rerun.code, 0, "{}", rerun.describe());
    let rerun = rerun.json();
    assert_eq!(str_at(&rerun, "/state"), "complete");
    assert_eq!(u64_at(&rerun, "/capture/events/inserted"), 0);
    assert_eq!(str_at(&rerun, "/search/worker"), "not_started");
    let text_receipt = machine.run(&atlas, &["import", ".", "--format", "text"]);
    assert_eq!(text_receipt.code, 0, "{}", text_receipt.describe());
    for internal in ["generation", "vector", "embedding"] {
        assert!(
            !text_receipt.stdout.to_lowercase().contains(internal),
            "human import output hides {internal}: {}",
            text_receipt.stdout
        );
    }
    assert!(text_receipt.stdout.contains("atlas"));

    let found = machine.run(&atlas, &["find", "why iroh rather than nats"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    let found = found.json();
    assert_eq!(str_at(&found, "/selection/source"), "project_binding");
    let found_hits = hits(&found);
    assert!(
        !found_hits.is_empty(),
        "the transport discussion is found: {found}"
    );
    let top = &found_hits[0];
    assert!(str_at(top, "/text").contains("Iroh"), "{top}");
    assert_eq!(str_at(top, "/context_name"), "atlas");
    assert!(top.get("cosine_similarity").is_some());
    assert!(top.get("embedding_space_id").is_some());
    let passage_id = str_at(top, "/passage_id").to_owned();
    let linked = top
        .get("source_links")
        .and_then(Value::as_array)
        .expect("source links")
        .iter()
        .map(|link| str_at(link, "/event_id").to_owned())
        .collect::<Vec<_>>();
    assert!(!linked.is_empty());

    let text = machine.run(
        &atlas,
        &["find", "why iroh rather than nats", "--format", "text"],
    );
    assert_eq!(text.code, 0, "{}", text.describe());
    assert!(text.stdout.contains(&passage_id) && text.stdout.contains("atlas"));
    for internal in ["generation", "embedding", "vector", "cosine"] {
        assert!(
            !text.stdout.to_lowercase().contains(internal),
            "human find output hides {internal}: {}",
            text.stdout
        );
    }

    fs::remove_dir_all(&history).expect("remove the source history");
    let shown = machine.run(&atlas, &["show", &passage_id]);
    assert_eq!(shown.code, 0, "{}", shown.describe());
    let shown = shown.json();
    let events = shown
        .get("events")
        .and_then(Value::as_array)
        .expect("shown events");
    assert_eq!(
        events
            .iter()
            .map(|event| str_at(event, "/id").to_owned())
            .collect::<Vec<_>>(),
        linked,
        "show expands exactly the linked events"
    );
    assert!(events.iter().any(|event| {
        event
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.contains("broker deployment"))
    }));
    for event in events {
        for field in [
            "seq",
            "author",
            "kind",
            "render_role",
            "payload",
            "anchors",
            "coverage",
            "source_remainder",
        ] {
            assert!(event.get(field).is_some(), "show exposes {field}: {event}");
        }
    }
    let shown_text = machine.run(&atlas, &["show", &passage_id, "--format", "text"]);
    assert_eq!(shown_text.code, 0);
    assert!(shown_text.stdout.contains("broker deployment"));

    let foreign = machine.run(&atlas, &["show", &passage_id, "--context", "garden"]);
    assert_eq!(foreign.code, 1, "{}", foreign.describe());
    assert_eq!(str_at(&foreign.json(), "/state"), "passage_not_found");
    assert!(
        !foreign.stdout.contains("Iroh") && !foreign.stdout.contains("atlas"),
        "a foreign handle leaks no text or Context: {}",
        foreign.stdout
    );

    let wider = machine.run(&garden, &["show", &passage_id, "--all-contexts"]);
    assert_eq!(
        wider.code,
        0,
        "an explicit wider selection shows it: {}",
        wider.describe()
    );

    let mixed = machine.run(&atlas, &["find", "iroh nats", "--context", "atlas,garden"]);
    assert_eq!(mixed.code, 0, "{}", mixed.describe());
    let mixed = mixed.json();
    assert_eq!(str_at(&mixed, "/selection/source"), "explicit");
    assert!(
        hits(&mixed)
            .iter()
            .all(|hit| str_at(hit, "/context_name") == "atlas")
    );
    let omitted = mixed
        .get("omitted_contexts")
        .and_then(Value::as_array)
        .expect("omitted contexts");
    assert!(
        omitted
            .iter()
            .any(|context| str_at(context, "/name") == "garden")
    );

    let not_ready = machine.run(&garden, &["find", "tomato"]);
    assert_eq!(not_ready.code, 1, "{}", not_ready.describe());
    assert_eq!(
        str_at(&not_ready.json(), "/state"),
        "history_search_not_ready"
    );
    let still_pending = machine.run(&garden, &["status"]).json();
    assert_eq!(
        str_at(&still_pending, "/history/search/status"),
        "pending",
        "find never starts projection"
    );

    let unknown = machine.run(&atlas, &["find", "iroh", "--context", "atlas,nowhere"]);
    assert_ne!(unknown.code, 0);
    assert!(hits(&unknown.json()).is_empty());
    assert_eq!(str_at(&unknown.json(), "/state"), "unknown_context");

    let all = machine.run(&machine.root, &["find", "iroh nats", "--all-contexts"]);
    assert_eq!(all.code, 0, "{}", all.describe());
    assert_eq!(str_at(&all.json(), "/selection/source"), "all_authorized");

    let multi_status = machine.run(&atlas, &["status", "--all-contexts"]);
    assert_eq!(multi_status.code, 0, "{}", multi_status.describe());
    let multi_status = multi_status.json();
    assert_eq!(
        multi_status
            .get("contexts")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(2),
        "one tagged status per Context"
    );

    let unconfirmed = machine.run(&atlas, &["forget", "--session", "sess-transport"]);
    assert_eq!(unconfirmed.code, 2, "{}", unconfirmed.describe());
    let unconfirmed = unconfirmed.json();
    assert_eq!(str_at(&unconfirmed, "/state"), "confirmation_required");
    assert!(str_at(&unconfirmed, "/retry").contains("--yes"));
    assert_eq!(
        u64_at(
            &machine.run(&atlas, &["status"]).json(),
            "/history/captured_events"
        ),
        4,
        "no confirmation, no change"
    );

    let forgotten = machine.run(&atlas, &["forget", "--session", "sess-transport", "--yes"]);
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    let forgotten = forgotten.json();
    assert_eq!(str_at(&forgotten, "/state"), "forgotten");
    assert_eq!(u64_at(&forgotten, "/counts/events"), 2);
    assert!(u64_at(&forgotten, "/counts/passages") > 0);
    assert_eq!(
        str_at(&forgotten, "/search/status"),
        "rebuild_scheduled",
        "forget re-indexes the remaining history itself: {forgotten}"
    );
    assert_eq!(u64_at(&forgotten, "/search/pending_events"), 2);
    assert!(
        !forgotten.to_string().contains("cg import"),
        "the user is never told to re-import: {forgotten}"
    );

    // The remaining session becomes searchable again without another import.
    machine.wait_for_search_ready(&atlas, "atlas");
    let remaining = machine.run(&atlas, &["find", "ledger foreign currency"]);
    assert_eq!(remaining.code, 0, "{}", remaining.describe());
    assert!(
        hits(&remaining.json())
            .iter()
            .any(|hit| str_at(hit, "/text").contains("booking-date rate")),
        "the remaining session is searchable after the rebuild: {}",
        remaining.describe()
    );

    let after = machine.run(&atlas, &["find", "why iroh rather than nats"]);
    assert!(
        hits(&after.json())
            .iter()
            .all(|hit| !str_at(hit, "/text").contains("Iroh")),
        "find returns nothing from forgotten history: {}",
        after.describe()
    );
    let old_handle = machine.run(&atlas, &["show", &passage_id]);
    assert_eq!(old_handle.code, 1, "{}", old_handle.describe());
    let status = machine.run(&atlas, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 2);
    assert_eq!(u64_at(&status, "/history/sources/sessions"), 1);

    let nothing = machine.run(&atlas, &["forget", "--session", "sess-transport", "--yes"]);
    assert_eq!(nothing.code, 0, "{}", nothing.describe());
    assert_eq!(str_at(&nothing.json(), "/state"), "nothing_to_forget");

    let whole = machine.run(&atlas, &["forget", "--context", "atlas", "--yes"]);
    assert_eq!(whole.code, 0, "{}", whole.describe());
    assert_eq!(str_at(&whole.json(), "/state"), "forgotten");
    let emptied = machine.run(&atlas, &["status"]);
    assert_eq!(
        emptied.code,
        0,
        "the Context and its binding remain: {}",
        emptied.describe()
    );
    let emptied = emptied.json();
    assert_eq!(str_at(&emptied, "/context/name"), "atlas");
    assert_eq!(u64_at(&emptied, "/history/captured_events"), 0);
}

#[test]
fn import_dry_run_verifies_committed_prefixes_reports_what_would_be_captured_and_writes_nothing() {
    let machine = Machine::without_model();
    let project = machine.project("rehearsal");
    let history = machine.history_root();
    let unchanged = write_session(&history, &project, "sess-unchanged", &[("one", "two")]);
    let appended = write_session(&history, &project, "sess-appended", &[("three", "four")]);
    let rewritten = write_session(&history, &project, "sess-rewritten", &[("five", "six")]);

    // Before anything is stored, every selected file is new.
    let before = fingerprint(&machine.root);
    let fresh = machine.run(
        &project,
        &["import", ".", "--context", "rehearsal", "--dry-run"],
    );
    assert_eq!(fresh.code, 0, "{}", fresh.describe());
    let fresh = fresh.json();
    assert_eq!(str_at(&fresh, "/state"), "dry_run");
    assert_eq!(u64_at(&fresh, "/sources/files"), 3);
    assert_eq!(u64_at(&fresh, "/capture/new_sources"), 3);
    assert_eq!(
        fingerprint(&machine.root),
        before,
        "a dry run writes nothing"
    );
    assert!(!machine.data_root.exists());

    let imported = machine.run(
        &project,
        &["import", ".", "--context", "rehearsal", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    // One file grows, one is rewritten, one is new, one stays as it was.
    let mut grown = fs::read(&appended).expect("read appended session");
    let extra = fs::read(write_session(
        &history,
        &project,
        "sess-scratch",
        &[("seven", "eight")],
    ))
    .expect("scratch records");
    fs::remove_file(
        history
            .join(event_support::claude_project_key(&project))
            .join("sess-scratch.jsonl"),
    )
    .expect("remove scratch session");
    grown.extend_from_slice(&extra);
    fs::write(&appended, &grown).expect("append records");
    // Change one committed byte inside the recorded text, keeping every record valid JSON so the
    // file is still this project's session.
    let mut changed = fs::read(&rewritten).expect("read rewritten session");
    let at = changed
        .windows(4)
        .position(|window| window == b"five")
        .expect("the rewritten session carries its text");
    changed[at + 3] = b'x';
    fs::write(&rewritten, &changed).expect("rewrite a committed byte");
    write_session(&history, &project, "sess-new", &[("nine", "ten")]);

    let before = fingerprint(&machine.root);
    let rehearsed = machine.run(&project, &["import", ".", "--dry-run"]);
    assert_eq!(rehearsed.code, 0, "{}", rehearsed.describe());
    let rehearsed = rehearsed.json();
    assert_eq!(str_at(&rehearsed, "/state"), "dry_run");
    assert_eq!(str_at(&rehearsed, "/selection/source"), "project_binding");
    assert_eq!(u64_at(&rehearsed, "/capture/unchanged_sources"), 1);
    assert_eq!(u64_at(&rehearsed, "/capture/appended_sources"), 1);
    assert_eq!(u64_at(&rehearsed, "/capture/diverged_sources"), 1);
    assert_eq!(u64_at(&rehearsed, "/capture/new_sources"), 1);
    assert_eq!(
        u64_at(&rehearsed, "/capture/bytes_to_capture"),
        (extra.len()
            + fs::metadata(
                history
                    .join(event_support::claude_project_key(&project))
                    .join("sess-new.jsonl")
            )
            .expect("new file")
            .len() as usize) as u64,
        "a real run would read the appended tail and the whole new file"
    );
    let sources = rehearsed
        .pointer("/capture/sources")
        .and_then(Value::as_array)
        .expect("per-source plan");
    let status_of = |file: &str| {
        sources
            .iter()
            .find(|source| str_at(source, "/file") == file)
            .map(|source| str_at(source, "/prefix").to_owned())
            .unwrap_or_else(|| panic!("{file} is planned: {rehearsed}"))
    };
    assert_eq!(status_of("sess-unchanged.jsonl"), "unchanged");
    assert_eq!(status_of("sess-appended.jsonl"), "appended");
    assert_eq!(status_of("sess-rewritten.jsonl"), "diverged");
    assert_eq!(status_of("sess-new.jsonl"), "new");
    assert_eq!(
        fingerprint(&machine.root),
        before,
        "a dry run writes nothing"
    );
    let _ = unchanged;
}

#[test]
fn cg_import_background_spawn_failure_leaves_capture_durable_and_returns_typed_behind() {
    let machine = Machine::with_nomic();
    let project = machine.project("unspawnable");
    write_session(
        &machine.history_root(),
        &project,
        "sess-unspawnable",
        &[("unspawnable question", "unspawnable answer")],
    );
    // A private copy of the binary that is removed while the import waits at its Context
    // question, so the executable path it would launch its background worker from is gone.
    let binary = machine.root.join("bin").join("cg");
    fs::create_dir_all(binary.parent().expect("bin dir")).expect("bin dir");
    fs::copy(env!("CARGO_BIN_EXE_cg"), &binary).expect("copy the cg binary");
    let (imported, transcript) = machine.run_interactive_using(
        &binary,
        &project,
        &["import", "."],
        &[("Context name [unspawnable]: ", "\r")],
        &mut |_| fs::remove_file(&binary).expect("remove the worker executable"),
    );
    assert_eq!(
        imported.code,
        1,
        "{} transcript={transcript:?}",
        imported.describe()
    );
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);
    assert_eq!(str_at(&receipt, "/search/worker"), "spawn_failed");
    assert!(!str_at(&receipt, "/search/reason").is_empty());
    assert!(str_at(&receipt, "/retry").contains("--search wait"));

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        2,
        "capture stays durable"
    );
    assert_eq!(str_at(&status, "/history/search/status"), "pending");
}

/// A read must never make the background projection give up.
///
/// `cg status` is read-only and reads through ContextDB's read route, so it is never refused while
/// the worker holds the Store; the converse must hold too — readers arriving throughout the build
/// must not cost the worker the work it was started for. Three readers poll here at the journey's
/// own cadence, which is faster than any human would, for as long as the build takes.
#[test]
fn search_reaches_ready_while_three_readers_poll_the_store_throughout_the_build() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let machine = Machine::with_nomic();
    let history = machine.history_root();
    let atlas = machine.project("atlas");
    write_session(
        &history,
        &atlas,
        "sess-transport",
        &[(
            "Why did we pick Iroh instead of NATS for the transport?",
            "Iroh keeps a direct peer-to-peer path; NATS would add a broker deployment we must operate.",
        )],
    );
    write_session(
        &history,
        &atlas,
        "sess-ledger",
        &[(
            "How should the ledger reconcile foreign currency entries?",
            "Convert at the booking-date rate and keep the original amount beside it.",
        )],
    );

    let imported = machine.run(&atlas, &["import", ".", "--context", "atlas"]);
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(str_at(&imported.json(), "/search/status"), "scheduled");

    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let machine = &machine;
            let atlas = &atlas;
            let stop = &stop;
            scope.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let status = machine.run(atlas, &["status", "--context", "atlas"]);
                    assert_eq!(
                        status.code,
                        0,
                        "a read is never refused while the projection runs: {}",
                        status.describe()
                    );
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            });
        }
        let ready = machine.wait_for_search_ready(&atlas, "atlas");
        stop.store(true, Ordering::Relaxed);
        assert_eq!(u64_at(&ready, "/history/search/pending_events"), 0);
        assert_eq!(u64_at(&ready, "/history/search/covered_events"), 4);
    });
}

/// A worker lock this user cannot open is a failure to report, not a worker already running.
///
/// `already_running` means another worker holds the lock and will do the work, so the import is
/// honest to say the search was scheduled. Every other reason the lock cannot be taken means
/// nothing was scheduled at all, and reporting `captured_search_scheduled` there would leave the
/// user waiting for a worker that was never started.
#[test]
fn cg_import_reports_search_behind_when_the_worker_lock_cannot_be_opened() {
    use std::os::unix::fs::PermissionsExt;

    let machine = Machine::with_nomic();
    let project = machine.project("unlockable");
    write_session(
        &machine.history_root(),
        &project,
        "sess-unlockable",
        &[("unlockable question", "unlockable answer")],
    );

    // Capture first with search deferred: the data root and its store exist, search has pending
    // work, and no worker has run.
    let deferred = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "unlockable",
            "--search",
            "defer",
        ],
    );
    assert_eq!(deferred.code, 0, "{}", deferred.describe());
    assert_eq!(
        str_at(&deferred.json(), "/state"),
        "captured_search_deferred"
    );

    // The lock file is present but unopenable by this user: not a lock somebody holds, an
    // unusable one.
    let lock_path = machine.data_root.join("event-search-worker.lock");
    fs::write(&lock_path, b"").expect("create the worker lock file");
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o000))
        .expect("make the worker lock file unopenable");

    let imported = machine.run(&project, &["import", "."]);
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o600))
        .expect("restore the worker lock file");
    assert_eq!(imported.code, 1, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(str_at(&receipt, "/search/worker"), "spawn_failed");
    assert!(
        !str_at(&receipt, "/search/reason").is_empty(),
        "the receipt discloses why nothing was started: {receipt}"
    );
    assert!(str_at(&receipt, "/retry").contains("--search wait"));

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        2,
        "capture stays durable"
    );
    assert_eq!(str_at(&status, "/history/search/status"), "pending");
}

/// `--search wait` reports an unusable worker lock instead of waiting for a turn that never comes.
///
/// Waiting is right for a lock another worker holds: that worker finishes and lets go. A lock that
/// cannot be opened at all is not a turn that is coming, so a foreground import that waits for it
/// hangs forever rather than telling the user their search did not run.
#[test]
fn cg_import_search_wait_reports_behind_when_the_worker_lock_cannot_be_opened() {
    use std::os::unix::fs::PermissionsExt;

    let machine = Machine::with_nomic();
    let project = machine.project("unlockable-wait");
    write_session(
        &machine.history_root(),
        &project,
        "sess-unlockable-wait",
        &[("unlockable wait question", "unlockable wait answer")],
    );

    let deferred = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "unlockable-wait",
            "--search",
            "defer",
        ],
    );
    assert_eq!(deferred.code, 0, "{}", deferred.describe());

    let lock_path = machine.data_root.join("event-search-worker.lock");
    fs::write(&lock_path, b"").expect("create the worker lock file");
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o000))
        .expect("make the worker lock file unopenable");

    let waited = machine.run(&project, &["import", ".", "--search", "wait"]);
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o600))
        .expect("restore the worker lock file");
    assert_eq!(waited.code, 1, "{}", waited.describe());
    let receipt = waited.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert!(
        !str_at(&receipt, "/search/reason").is_empty(),
        "the receipt discloses why the turn could not be taken: {receipt}"
    );

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(
        u64_at(&status.json(), "/history/captured_events"),
        2,
        "capture stays durable"
    );
}
