//! `cg forget` confirms on the controlling terminal, where Enter means no, and without a terminal
//! it requires `--yes`.

mod event_support;

use event_support::{Machine, str_at, u64_at, write_session};

#[test]
fn whitespace_only_session_forget_erases_checkpoint_and_locator() {
    whitespace_only_import_is_forgotten("session");
}

#[test]
fn whitespace_only_context_forget_erases_checkpoint_and_locator() {
    whitespace_only_import_is_forgotten("context");
}

fn whitespace_only_import_is_forgotten(selector: &str) {
    use context_graph::{ConsumerReader, event_history_reader_declaration};
    use contextdb_core::Value;
    use std::collections::HashMap;

    let machine = Machine::without_model();
    let project = machine.project(&format!("whitespace-{selector}"));
    let source = write_session(&machine.history_root(), &project, "sess-whitespace", &[]);
    std::fs::write(&source, " \t\n\n").expect("whitespace-only source");
    let context = format!("whitespace-{selector}");
    let imported = machine.run(
        &project,
        &[
            "import",
            source.to_str().expect("UTF-8 test path"),
            "--context",
            &context,
            "--search",
            "defer",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    let source_counts = || {
        let reader = ConsumerReader::open(machine.db_path(), event_history_reader_declaration())
            .expect("read source state");
        ["event_source_checkpoints", "event_source_locators"].map(|table| {
            let answer = reader
                .execute(
                    &format!("SELECT COUNT(*) AS n FROM {table}"),
                    &HashMap::new(),
                )
                .expect("count source state");
            match answer.rows.first().and_then(|row| row.first()) {
                Some(Value::Int64(count)) => *count,
                other => panic!("invalid {table} count: {other:?}"),
            }
        })
    };
    assert_eq!(source_counts(), [1, 1], "import persists both source rows");

    let selection = if selector == "session" {
        vec![
            "forget",
            "--session",
            "sess-whitespace",
            "--context",
            &context,
        ]
    } else {
        vec!["forget", "--context", &context]
    };
    let mut under_budget = selection.clone();
    under_budget.extend(["--yes", "--write-memory-limit-bytes", "1"]);
    let refused = machine.run(&project, &under_budget);
    assert_eq!(refused.code, 1, "{}", refused.describe());
    assert_eq!(str_at(&refused.json(), "/state"), "purge_refused");
    assert_eq!(
        str_at(&refused.json(), "/condition"),
        "WriteDidNotFitTheMemoryLimit"
    );
    let refusal = refused.json();
    let message = str_at(&refusal, "/message");
    assert!(
        message.contains("erasure needed")
            && message.contains("no history was erased")
            && message.contains("--write-memory-limit-bytes")
            && message.contains("[write] memory_limit_bytes"),
        "the typed memory refusal gives erasure-specific guidance: {}",
        refused.describe()
    );
    assert_eq!(source_counts(), [1, 1], "memory refusal changes nothing");

    let unconfirmed = machine.run(&project, &selection);
    assert_eq!(unconfirmed.code, 2, "{}", unconfirmed.describe());
    assert_eq!(
        str_at(&unconfirmed.json(), "/state"),
        "confirmation_required"
    );
    assert_eq!(u64_at(&unconfirmed.json(), "/counts/sessions"), 1);
    assert_eq!(u64_at(&unconfirmed.json(), "/counts/events"), 0);
    assert_eq!(source_counts(), [1, 1], "confirmation changes nothing");

    let mut authorized = selection.clone();
    authorized.push("--yes");
    let forgotten = machine.run(&project, &authorized);
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    assert_eq!(str_at(&forgotten.json(), "/state"), "forgotten");
    assert_eq!(u64_at(&forgotten.json(), "/counts/sessions"), 1);
    assert_eq!(u64_at(&forgotten.json(), "/counts/events"), 0);
    assert_eq!(source_counts(), [0, 0], "both source rows are erased");

    let empty = machine.run(&project, &selection);
    assert_eq!(empty.code, 0, "{}", empty.describe());
    assert_eq!(str_at(&empty.json(), "/state"), "nothing_to_forget");
}

#[cfg(all(unix, feature = "event-search-test-support"))]
#[test]
fn forget_keeps_the_reader_coordination_condition_when_erasure_is_refused() {
    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;
    use std::process::{Command, Stdio};
    use std::time::Duration;

    use event_support::wait_for_state;

    struct RestoreDirectory(std::path::PathBuf);
    impl Drop for RestoreDirectory {
        fn drop(&mut self) {
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o700));
        }
    }

    let machine = Machine::without_model();
    let project = machine.project("coordination");
    // The owner channel binds a Unix socket below this directory, and a socket address holds only
    // about a hundred bytes, so the directory sits under `/tmp` rather than below the long
    // per-user temporary directory macOS hands out.
    let runtime_root = tempfile::Builder::new()
        .prefix("cgr")
        .tempdir_in("/tmp")
        .expect("a coordination root short enough for local channel addresses");
    let runtime = runtime_root.path().to_path_buf();
    fs::set_permissions(&runtime, fs::Permissions::from_mode(0o700))
        .expect("make coordination root private");
    write_session(
        &machine.history_root(),
        &project,
        "sess-coordination",
        &[("question", "answer")],
    );
    let imported = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "coordination",
            "--search",
            "defer",
        ],
        "CONTEXTDB_OWNER_READ_RUNTIME_DIR",
        &runtime,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let before = machine
        .run_with_env(
            &project,
            &["status"],
            "CONTEXTDB_OWNER_READ_RUNTIME_DIR",
            &runtime,
        )
        .json();
    assert_eq!(u64_at(&before, "/history/captured_events"), 2);

    let socket = machine.root.join("before-purge.sock");
    let listener = UnixListener::bind(&socket).expect("bind forget barrier");
    listener.set_nonblocking(true).expect("nonblocking barrier");
    let mut child = Command::new(env!("CARGO_BIN_EXE_cg"))
        .args([
            "forget",
            "--context",
            "coordination",
            "--yes",
            "--format",
            "json",
        ])
        .current_dir(&project)
        .env("CG_DATA_ROOT", &machine.data_root)
        .env("HOME", &machine.home)
        .env("CG_MODEL_CACHE_ROOT", &machine.model_cache)
        .env("CONTEXTDB_OWNER_READ_RUNTIME_DIR", &runtime)
        .env("CG_FORGET_TEST_BEFORE_PURGE_SOCKET", &socket)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run forget command");
    let reached =
        wait_for_state(
            Duration::from_secs(10),
            Duration::from_millis(20),
            || match listener.accept() {
                Ok((stream, _)) => Ok::<_, std::io::Error>(Some(stream)),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
                Err(error) => Err(error),
            },
        )
        .expect("accept forget barrier");
    let Some(mut stream) = reached else {
        let _ = child.kill();
        let output = child.wait_with_output().expect("collect forget diagnostic");
        panic!(
            "forget did not open its writable store before erasure: status={:?} stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    };
    let mut ready = [0_u8; 1];
    stream.read_exact(&mut ready).expect("writable open marker");
    assert_eq!(ready, *b"r");

    fs::set_permissions(&runtime, fs::Permissions::from_mode(0o000))
        .expect("make reader coordination unavailable after open");
    let restore = RestoreDirectory(runtime.clone());
    stream.write_all(b"g").expect("release erasure");
    let output = child.wait_with_output().expect("forget exit");
    drop(restore);
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 JSON");
    assert!(!output.status.success(), "{stdout}");
    let refusal: serde_json::Value = serde_json::from_str(&stdout).expect("forget JSON");
    assert_eq!(
        str_at(&refusal, "/state"),
        "reader_coordination_unavailable"
    );
    assert_eq!(
        str_at(&refusal, "/condition"),
        "ReaderCoordinationUnavailable"
    );
    let message = str_at(&refusal, "/message");
    assert!(message.contains("authoritative purge"), "{message}");
    assert!(
        message.contains(&runtime.display().to_string()),
        "{message}"
    );
    let (_, explanation) = message
        .split_once(": ")
        .expect("coordination refusal retains its explanation");
    assert!(!explanation.trim().is_empty(), "{message}");

    let status = machine.run_with_env(
        &project,
        &["status"],
        "CONTEXTDB_OWNER_READ_RUNTIME_DIR",
        &runtime,
    );
    assert_eq!(status.code, 0, "{}", status.describe());
    let after = status.json();
    assert_eq!(u64_at(&after, "/history/captured_events"), 2);
    assert_eq!(
        after.pointer("/history/sources"),
        before.pointer("/history/sources")
    );
    assert_eq!(
        after.pointer("/history/coverage"),
        before.pointer("/history/coverage")
    );
}

#[test]
fn cg_forget_confirms_on_the_terminal_and_enter_changes_nothing() {
    let machine = Machine::without_model();
    let project = machine.project("pantry");
    write_session(
        &machine.history_root(),
        &project,
        "sess-pantry-one",
        &[("pantry question", "pantry answer")],
    );
    write_session(
        &machine.history_root(),
        &project,
        "sess-pantry-two",
        &[("second question", "second answer")],
    );
    let imported = machine.run(
        &project,
        &["import", ".", "--context", "pantry", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    let unattended = machine.run(&project, &["forget", "--context", "pantry"]);
    assert_eq!(unattended.code, 2, "{}", unattended.describe());
    let unattended = unattended.json();
    assert_eq!(str_at(&unattended, "/state"), "confirmation_required");
    assert_eq!(
        str_at(&unattended, "/retry"),
        "cg forget --context pantry --yes"
    );

    let (declined, transcript) = machine.run_interactive(
        &project,
        &["forget", "--session", "sess-pantry-one"],
        &[("Forget 1 sessions (2 events) from \"pantry\"? [y/N] ", "\r")],
    );
    assert_eq!(
        declined.code,
        2,
        "{} transcript={transcript:?}",
        declined.describe()
    );
    assert!(!declined.stdout.contains("Forget 1 sessions"));
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        4,
        "Enter means no"
    );

    let (session, transcript) = machine.run_interactive(
        &project,
        &["forget", "--session", "sess-pantry-one"],
        &[(
            "Forget 1 sessions (2 events) from \"pantry\"? [y/N] ",
            "y\r",
        )],
    );
    assert_eq!(
        session.code,
        0,
        "{} transcript={transcript:?}",
        session.describe()
    );
    let session = session.json();
    assert_eq!(str_at(&session, "/state"), "forgotten");
    assert_eq!(u64_at(&session, "/counts/events"), 2);
    assert_eq!(
        str_at(&session, "/search/status"),
        "blocked",
        "without the local model the rebuild of the remaining history is blocked: {session}"
    );
    assert!(str_at(&session, "/search/next").contains("cg model fetch --defaults"));
    assert!(
        !session.to_string().contains("cg import"),
        "the user is never told to re-import: {session}"
    );
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 2);

    let (whole, transcript) = machine.run_interactive(
        &project,
        &["forget", "--context", "pantry", "--format", "text"],
        &[(
            "Forget 1 sessions (2 events) from \"pantry\"? [y/N] ",
            "y\r",
        )],
    );
    assert_eq!(
        whole.code,
        0,
        "{} transcript={transcript:?}",
        whole.describe()
    );
    assert!(whole.stdout.contains("pantry"), "{}", whole.describe());
    assert!(!whole.stdout.contains("cg import"), "{}", whole.describe());
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 0);
    assert_eq!(
        str_at(&status, "/context/name"),
        "pantry",
        "the Context remains"
    );
}
