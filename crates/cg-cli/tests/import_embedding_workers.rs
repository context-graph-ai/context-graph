//! CLI contract for the event-search embedding worker cap.

mod event_support;

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
use event_support::wait_for_state;
use event_support::{Machine, fingerprint, str_at, u64_at, write_session};

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
use cg_node_fs::write_event_search_worker_policy;
use cg_node_fs::{
    EVENT_SEARCH_WORKER_POLICY, acquire_worker_lock, read_event_search_worker_policy,
};

#[test]
fn a_panicking_async_fixture_kills_and_reaps_only_its_owned_child() {
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let spawn_waiting_child = || {
        let mut command = Command::new("sh");
        command
            .args(["-c", "read release"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        // SAFETY: setsid is async-signal-safe and detaches only this fixture child.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        event_support::TestChild::new(command.spawn().expect("spawn a waiting fixture child"))
    };
    let mut unrelated = spawn_waiting_child();
    let failed = spawn_waiting_child();
    let failed_pid = failed.id() as i32;
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _owned_until_unwind = failed;
        panic!("force an assertion failure before the fixture is released");
    }));
    assert!(unwound.is_err());

    let mut status = 0;
    // SAFETY: failed_pid is the exact child spawned above. WNOHANG observes whether cleanup
    // reaped it without waiting indefinitely if the owner regresses.
    let observed = unsafe { libc::waitpid(failed_pid, &mut status, libc::WNOHANG) };
    let error = std::io::Error::last_os_error();
    if observed == 0 {
        // SAFETY: waitpid just verified that this process still owns this live child. Clean up
        // that exact PID even when the assertion below detects a broken unwind owner.
        unsafe {
            libc::kill(failed_pid, libc::SIGKILL);
            loop {
                let reaped = libc::waitpid(failed_pid, &mut status, 0);
                if reaped != -1
                    || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR)
                {
                    break;
                }
            }
        }
    }
    assert_eq!(observed, -1, "the panicking fixture must already be reaped");
    assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
    assert!(
        unrelated
            .try_wait()
            .expect("observe the unrelated child")
            .is_none(),
        "unwind cleanup must leave the unrelated fixture running"
    );
    unrelated.kill().expect("stop the unrelated fixture");
    unrelated.wait().expect("reap the unrelated fixture");
}

fn worker_policy(root: &std::path::Path) -> serde_json::Value {
    serde_json::from_str(
        &read_event_search_worker_policy(root)
            .expect("read delivered worker policy")
            .expect("worker policy exists"),
    )
    .expect("worker policy is JSON")
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
const REQUEST_OBSERVER_ENV: &str = "CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER";

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn request_observer(machine: &Machine, name: &str) -> std::path::PathBuf {
    let path = machine.root.join(name);
    std::fs::create_dir_all(&path).expect("create request observer directory");
    event_support::write_test_file(path.join("release"), b"do not pause a background worker\n")
        .expect("pre-release the existing handoff barrier");
    path
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observed_request_caps(observer: &std::path::Path) -> Vec<usize> {
    std::fs::read_to_string(observer.join("embedding-worker-requests"))
        .unwrap_or_default()
        .lines()
        .map(|line| line.parse::<usize>().expect("observed cap is a usize"))
        .collect()
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn batch_observer(machine: &Machine, name: &str) -> std::path::PathBuf {
    let path = machine.root.join(name);
    std::fs::create_dir_all(&path).expect("create batch observer directory");
    event_support::write_test_file(
        path.join("observe-embedding-worker-batches"),
        b"observe and pause\n",
    )
    .expect("opt into batch observation");
    path
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observed_batch_caps(observer: &std::path::Path) -> Vec<usize> {
    std::fs::read_to_string(observer.join("embedding-worker-batches"))
        .unwrap_or_default()
        .lines()
        .map(|line| {
            line.parse::<usize>()
                .expect("observed batch cap is a usize")
        })
        .collect()
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn terminate_observed_background_worker(observer: &std::path::Path) {
    let process_group = std::fs::read_to_string(observer.join("embedding-worker-process-group"))
        .expect("read owned background worker process group")
        .trim()
        .parse::<i32>()
        .expect("owned background worker process group is a pid");
    assert!(
        process_group > 0,
        "owned background worker process group is positive"
    );
    // SAFETY: the production worker starts this test-observed child in a new process group, and
    // this isolated fixture owns that group until the lock confirms it has exited.
    let result = unsafe { libc::kill(-process_group, libc::SIGTERM) };
    if result == -1 {
        let error = std::io::Error::last_os_error();
        assert_eq!(
            error.raw_os_error(),
            Some(libc::ESRCH),
            "terminate owned background worker process group {process_group}: {error}"
        );
        // A completed worker may already have exited after its last observed callback. Require
        // that its group is gone; callers still verify release of the owned worker lock.
        assert_eq!(unsafe { libc::kill(-process_group, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    } else {
        assert_eq!(result, 0);
    }
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn write_config(machine: &Machine, body: &str) {
    let config_dir = machine.data_root.join("cg");
    std::fs::create_dir_all(&config_dir).expect("create cg config directory");
    event_support::write_test_file(config_dir.join("fabric.toml"), body).expect("write cg config");
}

#[test]
fn deferred_import_never_creates_or_changes_the_active_worker_policy() {
    let machine = Machine::without_model();
    let project = machine.project("worker-cap");
    write_session(
        &machine.history_root(),
        &project,
        "worker-cap-session",
        &[("why parallelize passages?", "to reduce first-import time")],
    );
    let config_dir = machine.data_root.join("cg");
    std::fs::create_dir_all(&config_dir).expect("create cg config directory");
    let config_path = config_dir.join("fabric.toml");
    let configured = "[search]\nembedding_workers = 1\n";
    event_support::write_test_file(&config_path, configured).expect("write saved worker cap");

    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "worker-cap",
            "--search",
            "defer",
            "--embedding-workers",
            "2",
        ],
    );

    assert_eq!(
        imported.code,
        0,
        "a positive CLI cap must parse even when search is deferred: {}",
        imported.describe()
    );
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_deferred");
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);
    assert_eq!(
        std::fs::read_to_string(config_path).expect("read saved config after import"),
        configured,
        "a one-run CLI override must not edit the saved [search] cap"
    );
    let policy_path = machine.data_root.join(EVENT_SEARCH_WORKER_POLICY);
    assert!(
        !policy_path.exists(),
        "a deferred import created active-worker runtime policy"
    );

    let active = r#"{"publication":"active-fixture","embedding_workers":2}"#;
    event_support::write_test_file(&policy_path, active)
        .expect("seed another build's active policy");
    let deferred_again = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "worker-cap",
            "--search",
            "defer",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(deferred_again.code, 0, "{}", deferred_again.describe());
    assert_eq!(
        std::fs::read_to_string(policy_path).expect("read retained active policy"),
        active,
        "a deferred import changed another build's active request"
    );
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn foreground_wait_cli_override_reaches_the_production_request_builder() {
    let machine = Machine::with_nomic();
    let project = machine.project("wait-cli-request");
    write_session(
        &machine.history_root(),
        &project,
        "wait-cli-request-session",
        &[("which cap reaches wait?", "the explicit cap two")],
    );
    let observer = request_observer(&machine, "wait-cli-observer");
    let imported = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "wait-cli-request",
            "--search",
            "wait",
            "--embedding-workers",
            "2",
        ],
        REQUEST_OBSERVER_ENV,
        &observer,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(observed_request_caps(&observer), vec![2]);
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn foreground_wait_configured_cap_reaches_the_production_request_builder() {
    let machine = Machine::with_nomic();
    let project = machine.project("wait-config-request");
    write_session(
        &machine.history_root(),
        &project,
        "wait-config-request-session",
        &[("which cap reaches wait?", "the configured cap one")],
    );
    write_config(&machine, "[search]\nembedding_workers = 1\n");
    let observer = request_observer(&machine, "wait-config-observer");
    let imported = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "wait-config-request",
            "--search",
            "wait",
        ],
        REQUEST_OBSERVER_ENV,
        &observer,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(observed_request_caps(&observer), vec![1]);
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn foreground_wait_omission_reaches_the_production_request_builder_as_available_parallelism() {
    let machine = Machine::with_nomic();
    let project = machine.project("wait-default-request");
    write_session(
        &machine.history_root(),
        &project,
        "wait-default-request-session",
        &[("which cap reaches wait?", "available parallelism")],
    );
    let observer = request_observer(&machine, "wait-default-observer");
    let imported = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "wait-default-request",
            "--search",
            "wait",
        ],
        REQUEST_OBSERVER_ENV,
        &observer,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(
        observed_request_caps(&observer),
        vec![std::thread::available_parallelism().unwrap().get()]
    );
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn detached_background_import_delivers_default_cap_to_its_production_request_builder() {
    let initial_embedding_workers = std::thread::available_parallelism()
        .expect("host parallelism is available")
        .get();
    let replacement_embedding_workers = if initial_embedding_workers == 2 { 1 } else { 2 };
    let machine = Machine::with_nomic();
    let project = machine.project("background-default-request");
    // All turns are in one claim. With the valid narrow profile, each user boundary closes the
    // preceding turn, so 34 tiny turns exceed the production 32-passage batch bound. Releasing
    // the first callback therefore makes the runner reclaim a narrowed batch and call the
    // resolver again before any Nomic forward call.
    write_config(
        &machine,
        "[search.passages]\ntarget_tokens = 8\nhard_tokens = 64\n",
    );
    let turns = vec![
        (
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten",
        );
        34
    ];
    write_session(
        &machine.history_root(),
        &project,
        "background-default-request-session",
        &turns,
    );
    // macOS TMPDIR paths can already consume most of a Unix socket address.
    let observation = tempfile::Builder::new()
        .prefix("cgb")
        .tempdir_in("/tmp")
        .expect("short batch rendezvous directory");
    let observer = observation.path().to_path_buf();
    event_support::write_test_file(
        observer.join("observe-embedding-worker-batches"),
        b"observe",
    )
    .expect("opt into batch observation");
    let batches = std::os::unix::net::UnixListener::bind(observer.join("batch-events.sock"))
        .expect("bind batch rendezvous before spawning the worker");
    let receive_batch = || {
        use std::io::Read;
        let (mut stream, _) = batches.accept().expect("batch callback connects");
        let mut cap = [0; 8];
        stream.read_exact(&mut cap).expect("batch cap is complete");
        (
            stream,
            usize::try_from(u64::from_le_bytes(cap)).expect("cap fits"),
        )
    };
    let imported = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "background-default-request",
            "--search",
            "background",
        ],
        REQUEST_OBSERVER_ENV,
        &observer,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    struct StopObservedWorker<'a>(&'a std::path::Path);
    impl Drop for StopObservedWorker<'_> {
        fn drop(&mut self) {
            terminate_observed_background_worker(self.0);
        }
    }

    let (mut first_batch, first_cap) = receive_batch();
    let stop_worker = StopObservedWorker(&observer);
    assert_eq!(first_cap, initial_embedding_workers);
    assert_eq!(
        observed_batch_caps(&observer),
        vec![initial_embedding_workers]
    );

    // A callback cannot proceed until this policy is durably replaced and the
    // test releases it. The next callback announces the resolver's new choice.
    let mut replacement_policy = worker_policy(&machine.data_root);
    replacement_policy["publication"] = "replacement".into();
    replacement_policy["embedding_workers"] = replacement_embedding_workers.into();
    write_event_search_worker_policy(&machine.data_root, &replacement_policy.to_string())
        .expect("replace worker policy while its first batch is held");
    use std::io::Write;
    first_batch.write_all(&[1]).expect("drive the next batch");
    let (_held_second_batch, second_cap) = receive_batch();
    assert_eq!(second_cap, replacement_embedding_workers);
    let later = observed_batch_caps(&observer);
    drop(stop_worker);
    machine.wait_for_search_worker_exit();
    assert_eq!(
        observed_request_caps(&observer),
        vec![initial_embedding_workers],
        "the detached worker request builder did not carry the available-CPU default"
    );
    assert_eq!(
        later,
        vec![initial_embedding_workers, replacement_embedding_workers],
        "the detached worker did not apply the replacement cap at its next batch boundary"
    );
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn resumed_deferred_projection_delivers_the_new_request_to_the_production_builder() {
    let machine = Machine::with_nomic();
    let project = machine.project("resume-request");
    write_session(
        &machine.history_root(),
        &project,
        "resume-request-session",
        &[("was this deferred?", "yes, then resumed at cap two")],
    );
    let deferred = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "resume-request",
            "--search",
            "defer",
        ],
    );
    assert_eq!(deferred.code, 0, "{}", deferred.describe());
    write_config(&machine, "[search]\nembedding_workers = 2\n");
    let observer = request_observer(&machine, "resume-observer");
    let resumed = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "resume-request",
            "--search",
            "wait",
        ],
        REQUEST_OBSERVER_ENV,
        &observer,
    );
    assert_eq!(resumed.code, 0, "{}", resumed.describe());
    assert_eq!(observed_request_caps(&observer), vec![2]);
}

#[test]
fn requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff() {
    let machine = Machine::with_nomic();
    let project = machine.project("worker-handoff");
    write_session(
        &machine.history_root(),
        &project,
        "worker-handoff-session",
        &[(
            "which cap reaches the worker?",
            "the requesting import's CLI cap",
        )],
    );
    std::fs::create_dir_all(&machine.data_root).expect("create isolated data root");
    let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
        .expect("hold the existing worker handoff without spawning a process");

    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "worker-handoff",
            "--search",
            "background",
            "--embedding-workers",
            "2",
            "--search-order",
            "oldest",
            "--search-sessions",
            "all",
        ],
    );
    drop(held);

    assert_eq!(imported.code, 0, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_scheduled");
    assert_eq!(str_at(&receipt, "/search/worker"), "already_running");
    assert_eq!(worker_policy(&machine.data_root)["embedding_workers"], 2);
    assert_eq!(
        worker_policy(&machine.data_root)["work_policy"]["order"],
        "oldest"
    );
    assert_eq!(
        worker_policy(&machine.data_root)["work_policy"]["sessions"],
        "all"
    );
    assert_eq!(
        worker_policy(&machine.data_root)["pending_background_request"]["work_policy"]["order"],
        "oldest"
    );
    assert_eq!(
        worker_policy(&machine.data_root)["pending_background_request"]["work_policy"]["sessions"],
        "all"
    );
    assert!(worker_policy(&machine.data_root)["publication"].is_string());
    assert!(machine.data_root.join(EVENT_SEARCH_WORKER_POLICY).is_file());
}

#[test]
fn interrupted_all_scope_retry_resumes_all_sessions_in_the_same_oldest_order() {
    let machine = Machine::with_nomic();
    let project = machine.project("retry-all-oldest");
    for index in 0..12 {
        write_session(
            &machine.history_root(),
            &project,
            &format!("retry-all-oldest-{index:02}"),
            &[(
                &format!("oldest retry question {index:02}"),
                &format!("oldest retry answer {index:02}"),
            )],
        );
    }

    // A directory at the worker-lock path makes the background handoff fail after durable capture.
    let lock_path = machine.data_root.join("event-search-worker.lock");
    std::fs::create_dir_all(&lock_path).expect("make the worker lock unusable");
    let interrupted = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "retry-all-oldest",
            "--search",
            "background",
            "--search-sessions",
            "all",
            "--search-order",
            "oldest",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(interrupted.code, 1, "{}", interrupted.describe());
    let interrupted_receipt = interrupted.json();
    assert_eq!(
        str_at(&interrupted_receipt, "/state"),
        "captured_search_behind"
    );
    assert_eq!(
        str_at(&interrupted_receipt, "/retry"),
        "cg import '<project-directory>' --search wait --search-sessions all --search-order oldest --embedding-workers 1"
    );
    assert_eq!(u64_at(&interrupted_receipt, "/capture/events/inserted"), 24);

    std::fs::remove_dir(&lock_path).expect("restore the worker lock path");
    // The receipt deliberately replaces only its path placeholder; every policy token below is
    // verbatim from the printed command and therefore resumes the all-session oldest ordering.
    let resumed = machine.run(
        &project,
        &[
            "import",
            ".",
            "--search",
            "wait",
            "--search-sessions",
            "all",
            "--search-order",
            "oldest",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(resumed.code, 0, "{}", resumed.describe());
    let resumed_receipt = resumed.json();
    assert_eq!(str_at(&resumed_receipt, "/state"), "complete");
    assert_eq!(u64_at(&resumed_receipt, "/capture/events/inserted"), 0);
    assert_eq!(u64_at(&resumed_receipt, "/search/covered_events"), 24);
    assert_eq!(u64_at(&resumed_receipt, "/search/pending_events"), 0);
    assert_eq!(
        worker_policy(&machine.data_root)["work_policy"]["order"],
        "oldest"
    );
    assert_eq!(
        worker_policy(&machine.data_root)["work_policy"]["sessions"],
        "all"
    );
}

#[test]
fn missing_model_retry_keeps_the_resolved_scope_without_a_hosted_or_downloaded_fallback() {
    let machine = Machine::without_model();
    let project = machine.project("missing-model-retry-scope");
    write_session(
        &machine.history_root(),
        &project,
        "missing-model-retry-scope",
        &[("what must retry preserve?", "the all-session oldest policy")],
    );

    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "missing-model-retry-scope",
            "--search",
            "wait",
            "--search-sessions",
            "all",
            "--search-order",
            "oldest",
        ],
    );
    assert_eq!(imported.code, 1, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    let retry = str_at(&receipt, "/retry");
    assert_eq!(
        retry,
        "cg model fetch --defaults && cg import '<project-directory>' --search wait --search-sessions all --search-order oldest"
    );
    assert!(!retry.contains("hosted"));
    assert!(!retry.contains("download"));
    assert_eq!(u64_at(&receipt, "/capture/local_embedding_calls"), 0);
    assert_eq!(u64_at(&receipt, "/capture/hosted_provider_calls"), 0);
}

#[test]
fn later_requesting_import_without_a_flag_replaces_the_one_off_override() {
    let machine = Machine::with_nomic();
    let project = machine.project("override-reset");
    write_session(
        &machine.history_root(),
        &project,
        "override-reset-first",
        &[("first request", "temporarily use cap two")],
    );
    std::fs::create_dir_all(&machine.data_root).expect("create isolated data root");
    let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
        .expect("hold the existing worker handoff");

    let first = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "override-reset",
            "--search",
            "background",
            "--embedding-workers",
            "2",
        ],
    );
    assert_eq!(first.code, 0, "{}", first.describe());
    let first_policy = worker_policy(&machine.data_root);
    write_session(
        &machine.history_root(),
        &project,
        "override-reset-second",
        &[("second request", "return to config or default")],
    );
    let second = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "override-reset",
            "--search",
            "background",
        ],
    );
    drop(held);

    assert_eq!(second.code, 0, "{}", second.describe());
    let replacement = worker_policy(&machine.data_root);
    assert_eq!(
        replacement["embedding_workers"],
        std::thread::available_parallelism().unwrap().get(),
        "the earlier explicit cap stuck to a later no-flag request"
    );
    assert_ne!(replacement["publication"], first_policy["publication"]);
}

#[test]
fn already_caught_up_import_does_not_change_an_active_worker_request() {
    let machine = Machine::with_nomic();
    let project = machine.project("caught-up-policy");
    write_session(
        &machine.history_root(),
        &project,
        "caught-up-policy-session",
        &[("finish first", "then leave another request alone")],
    );
    let built = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "caught-up-policy",
            "--search",
            "wait",
        ],
    );
    assert_eq!(built.code, 0, "{}", built.describe());
    let policy_path = machine.data_root.join(EVENT_SEARCH_WORKER_POLICY);
    let active = r#"{"publication":"active-fixture","embedding_workers":2}"#;
    event_support::write_test_file(&policy_path, active)
        .expect("seed another active worker request");

    let unchanged = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "caught-up-policy",
            "--search",
            "background",
        ],
    );
    assert_eq!(unchanged.code, 0, "{}", unchanged.describe());
    assert_eq!(str_at(&unchanged.json(), "/search/worker"), "not_started");
    assert_eq!(
        std::fs::read_to_string(policy_path).expect("read retained policy"),
        active,
        "an unchanged import changed another build's active request"
    );
}

#[test]
fn invalid_embedding_workers_flags_are_usage_errors_before_any_data_root_write() {
    for (case, value) in [("zero", "0"), ("negative", "-1"), ("non-number", "many")] {
        let machine = Machine::without_model();
        let project = machine.project(case);
        write_session(
            &machine.history_root(),
            &project,
            &format!("{case}-session"),
            &[("does this get captured?", "invalid policy must stop it")],
        );
        let before = fingerprint(&machine.data_root);

        let imported = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                case,
                "--search",
                "defer",
                "--embedding-workers",
                value,
            ],
        );

        assert_eq!(imported.code, 2, "{case}: {}", imported.describe());
        let receipt = imported.json();
        assert_eq!(str_at(&receipt, "/state"), "usage_error", "{case}");
        let message = str_at(&receipt, "/message");
        assert!(
            message.contains("--embedding-workers") && message.contains("positive whole number"),
            "{case} must name the positive worker-count rule: {message}"
        );
        assert_eq!(
            fingerprint(&machine.data_root),
            before,
            "{case} changed the data root before refusing the CLI value"
        );
    }
}

#[test]
fn invalid_search_scope_flags_and_config_are_typed_and_refused_before_writes() {
    for (flag, value) in [("--search-order", "sideways"), ("--search-sessions", "0")] {
        let machine = Machine::without_model();
        let project = machine.project("invalid-search-flag");
        write_session(
            &machine.history_root(),
            &project,
            "invalid-search-flag-session",
            &[(
                "does this get captured?",
                "invalid search policy must stop it",
            )],
        );
        let before = fingerprint(&machine.data_root);
        let imported = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "invalid-search-flag",
                flag,
                value,
            ],
        );
        assert_eq!(imported.code, 2, "{}", imported.describe());
        assert_eq!(str_at(&imported.json(), "/state"), "usage_error");
        assert_eq!(fingerprint(&machine.data_root), before);
    }

    let machine = Machine::without_model();
    let project = machine.project("invalid-search-config");
    write_session(
        &machine.history_root(),
        &project,
        "invalid-search-config-session",
        &[("does config validate first?", "it must precede capture")],
    );
    let config_dir = machine.data_root.join("cg");
    std::fs::create_dir_all(&config_dir).expect("create cg config directory");
    event_support::write_test_file(
        config_dir.join("fabric.toml"),
        "[search]\norder = \"sideways\"\nsessions = 10\n",
    )
    .expect("write invalid search policy");
    let before = fingerprint(&machine.data_root);
    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "invalid-search-config",
            "--search-order",
            "newest",
            "--search-sessions",
            "all",
        ],
    );
    assert_eq!(imported.code, 1, "{}", imported.describe());
    assert_eq!(str_at(&imported.json(), "/state"), "config_error");
    assert_eq!(fingerprint(&machine.data_root), before);
}

#[test]
fn invalid_configured_embedding_workers_are_refused_before_event_or_store_writes() {
    for (case, configured) in [
        ("zero", "[search]\nembedding_workers = 0\n"),
        ("negative", "[search]\nembedding_workers = -1\n"),
        ("non-number", "[search]\nembedding_workers = \"many\"\n"),
    ] {
        let machine = Machine::without_model();
        let project = machine.project(case);
        write_session(
            &machine.history_root(),
            &project,
            &format!("configured-{case}-session"),
            &[("does config validate first?", "it must precede capture")],
        );
        let config_dir = machine.data_root.join("cg");
        std::fs::create_dir_all(&config_dir).expect("create cg config directory");
        event_support::write_test_file(config_dir.join("fabric.toml"), configured)
            .expect("write invalid worker policy");
        let before = fingerprint(&machine.data_root);

        let imported = machine.run(
            &project,
            &["import", ".", "--context", case, "--search", "defer"],
        );

        assert_eq!(imported.code, 1, "{case}: {}", imported.describe());
        let receipt = imported.json();
        assert_eq!(str_at(&receipt, "/state"), "config_error", "{case}");
        let message = str_at(&receipt, "/message");
        assert!(
            message.contains("search.embedding_workers")
                && message.contains("positive whole number"),
            "{case} must name the configured positive worker-count rule: {message}"
        );
        assert_eq!(
            fingerprint(&machine.data_root),
            before,
            "{case} wrote a store or event before refusing configuration"
        );
    }
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn detached_projection_uses_the_imports_explicit_write_memory_limit() {
    use context_graph::write_budget::{MachineMemory, WriterMemoryLimit};

    let derived = WriterMemoryLimit::derived_from(MachineMemory {
        total_bytes: Some(8 * 1024 * 1024 * 1024),
        cgroup_limit_bytes: Some(512 * 1024 * 1024),
    });
    assert_eq!(derived.bytes.get(), 128 * 1024 * 1024);

    let machine = Machine::with_nomic();
    let project = machine.project("bounded-background-search");
    write_session(
        &machine.history_root(),
        &project,
        "bounded-session",
        &[(
            "remember the chosen write limit",
            "keep it through projection",
        )],
    );
    let observer = request_observer(&machine, "bounded-worker-observer");
    let imported = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "bounded-background-search",
            "--search",
            "background",
            "--write-memory-limit-bytes",
            "268435456",
        ],
        REQUEST_OBSERVER_ENV,
        &observer,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let observed = wait_for_state(
        std::time::Duration::from_secs(30),
        std::time::Duration::from_millis(20),
        || {
            Ok::<_, ()>(
                std::fs::read_to_string(observer.join("worker-store-write-memory-limit-bytes"))
                    .ok(),
            )
        },
    )
    .expect("worker budget observation cannot fail")
    .expect("detached worker reported its resolved budget");
    assert_eq!(
        observed, "268435456",
        "the child must keep the parent's 256 MiB limit"
    );
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
mod request_write_ceilings {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    const HIGH: usize = 512 * 1024 * 1024;
    const LOW: usize = 256 * 1024 * 1024;

    fn envelope() -> String {
        serde_json::json!({
            "profile": context_graph::EventSearchProfile::default(),
            "embedding_workers": 1,
            "ignored_publication": null,
            "retry_worker_lock": false,
        })
        .to_string()
    }

    fn import(
        machine: &Machine,
        project: &Path,
        context: &str,
        limit: usize,
        observer: &Path,
    ) -> event_support::Output {
        machine.run_with_env(
            project,
            &[
                "import",
                ".",
                "--context",
                context,
                "--search",
                "background",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &limit.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            observer,
        )
    }

    fn await_marker(observer: &Path, marker: &str) {
        assert_eq!(
            wait_for_state(Duration::from_secs(30), Duration::from_millis(20), || {
                Ok::<_, ()>(observer.join(marker).exists().then_some(()))
            })
            .expect("observe worker rendezvous"),
            Some(()),
            "worker did not reach {marker}"
        );
    }

    fn budgets(observer: &Path) -> Vec<(String, usize)> {
        std::fs::read_to_string(observer.join("write-memory-batches"))
            .unwrap_or_default()
            .lines()
            .map(|line| {
                let (context, bytes) = line.split_once(' ').expect("context and writer ceiling");
                (
                    context.to_owned(),
                    bytes.parse().expect("writer ceiling bytes"),
                )
            })
            .collect()
    }

    fn await_acknowledgement(machine: &Machine) {
        assert_eq!(
            wait_for_state(Duration::from_secs(120), Duration::from_millis(20), || {
                Ok::<_, ()>(
                    worker_policy(&machine.data_root)["pending_background_request"]
                        .is_null()
                        .then_some(()),
                )
            })
            .expect("queue acknowledgement observation"),
            Some(()),
            "worker did not acknowledge: {}",
            worker_policy(&machine.data_root)
        );
    }

    fn assert_ready(machine: &Machine, project: &Path, context: &str) {
        let status = machine.run(project, &["status", "--context", context]);
        assert_eq!(status.code, 0, "{}", status.describe());
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "ready",
            "{}",
            status.describe()
        );
    }

    fn import_selected(
        machine: &Machine,
        project: &Path,
        context: &str,
        limit: usize,
        sessions: &str,
        observer: &Path,
    ) -> event_support::Output {
        machine.run_with_env(
            project,
            &[
                "import",
                ".",
                "--context",
                context,
                "--search",
                "background",
                "--search-sessions",
                sessions,
                "--search-order",
                "newest",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &limit.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            observer,
        )
    }

    fn actual_budgets(observer: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(observer.join("actual-write-memory-batches"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).expect("actual engine admission JSON"))
            .collect()
    }

    fn capture_b(machine: &Machine, project: &Path) {
        let source = write_session(
            &machine.history_root(),
            project,
            "session-B",
            &[("B request selected before C exists", "B answer")],
        );
        let text = std::fs::read_to_string(&source).expect("B source");
        event_support::write_test_file(source, text.replace("2026-09-01", "2026-08-01"))
            .expect("B precedes C");
    }

    fn assert_selected_b_and_c_completed(
        machine: &Machine,
        project: &Path,
        observer: &Path,
        b_limit: usize,
        c_limit: usize,
    ) {
        let actual = actual_budgets(observer);
        for (session, expected) in [("session-B", b_limit), ("session-C", c_limit)] {
            let admissions = actual
                .iter()
                .filter(|batch| batch["session"] == session)
                .collect::<Vec<_>>();
            assert!(
                !admissions.is_empty(),
                "missing {session} admission: {actual:?}"
            );
            assert!(
                admissions
                    .iter()
                    .all(|batch| batch["limit"].as_u64() == Some(expected as u64)),
                "{session} used another request's budget: {actual:?}"
            );
        }
        assert!(
            worker_policy(&machine.data_root)["pending_background_request"].is_null(),
            "both accepted requests must be acknowledged"
        );
        assert_ready(machine, project, "durable-selection");
        let shown = machine.run(project, &["status", "--context", "durable-selection"]);
        assert_eq!(
            u64_at(&shown.json(), "/history/search/covered_events"),
            4,
            "{}",
            shown.describe()
        );
    }

    fn assert_selection_counts(machine: &Machine, expected: i64) {
        assert_selection_row_counts(machine, expected, expected);
        if expected == 0
            && worker_policy(&machine.data_root)["pending_background_request"].is_null()
        {
            assert_completion_row_counts(machine, 0, 0);
        }
    }

    fn assert_completion_row_counts(machine: &Machine, headers: i64, frontiers: i64) {
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("completion reader");
        for (table, expected) in [
            ("event_search_refusal_headers", headers),
            ("event_search_refusal_frontiers", frontiers),
        ] {
            let rows = reader
                .execute(
                    &format!("SELECT COUNT(*) AS total FROM {table}"),
                    &std::collections::HashMap::new(),
                )
                .expect("completion count");
            assert_eq!(
                rows.rows[0][0],
                contextdb_core::Value::Int64(expected),
                "unexpected retained rows in {table}"
            );
        }
    }

    fn assert_selection_row_counts(machine: &Machine, headers: i64, selections: i64) {
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("selection reader");
        for (table, expected) in [
            ("event_search_request_headers", headers),
            ("event_search_request_selections", selections),
        ] {
            let rows = reader
                .execute(
                    &format!("SELECT COUNT(*) AS total FROM {table}"),
                    &std::collections::HashMap::new(),
                )
                .expect("selection count");
            assert_eq!(
                rows.rows[0][0],
                contextdb_core::Value::Int64(expected),
                "unexpected retained rows in {table}"
            );
        }
    }

    fn refuse_queue(machine: &Machine, limit: usize) -> serde_json::Value {
        let mut policy = worker_policy(&machine.data_root);
        let pending = &mut policy["pending_background_request"];
        let context: context_graph::ContextId =
            serde_json::from_value(pending["write_requests"][0]["context_id"].clone())
                .expect("queued context uses the production ContextId representation");
        let context = context.to_string();
        for request in pending["write_requests"].as_array_mut().unwrap() {
            request["completed"] = true.into();
            request["refused_memory"] = true.into();
            request["memory_refusal"] = serde_json::json!({"Recorded": {"ordinal": 1, "refusal": {"requested_bytes": 42, "limit_bytes": limit}}});
            request["memory_limit_bytes"] = limit.into();
        }
        pending["write_memory_refusals"] =
            serde_json::json!({context: {"requested_bytes": 42, "limit_bytes": limit}});
        write_event_search_worker_policy(&machine.data_root, &policy.to_string()).unwrap();
        policy
    }

    #[test]
    fn refused_only_restart_stops_preserves_refusal_and_serves_a_shutdown_arrival() {
        // A real competing attempt keeps A incomplete. An unrelated publication must not
        // replenish A's allowance, and exhaustion of A must leave a distinct same-Context B
        // serviceable. The initial generation creation changes the first assessment once.
        for unrelated_publication in [true, false] {
            let machine = Machine::with_nomic();
            let project = machine.project("stale-selection-allowance");
            let observer = request_observer(&machine, "stale-selection-observer");
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let mut accepted = Vec::new();
            for (session, day) in [("stale-A", "01"), ("serviceable-B", "02")] {
                let path = write_session(
                    &machine.history_root(),
                    &project,
                    session,
                    &[("accepted selection", "answer")],
                );
                let text = std::fs::read_to_string(&path).unwrap();
                event_support::write_test_file(
                    path,
                    text.replace("2026-09-01", &format!("2026-09-{day}")),
                )
                .unwrap();
                let imported =
                    import_selected(&machine, &project, "stale-selection", HIGH, "1", &observer);
                assert_eq!(imported.code, 0, "{}", imported.describe());
                assert_eq!(
                    str_at(&imported.json(), "/search/worker"),
                    "already_running"
                );
                let policy = worker_policy(&machine.data_root);
                accepted.push(
                    policy["pending_background_request"]["write_requests"]
                        .as_array()
                        .unwrap()
                        .last()
                        .unwrap()
                        .clone(),
                );
            }
            assert_ne!(accepted[0]["selected_work"], accepted[1]["selected_work"]);
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            for (request, session) in accepted.iter().zip(["stale-A", "serviceable-B"]) {
                let selection = reader
                    .event_search_selected_work(
                        serde_json::from_value(request["context_id"].clone()).unwrap(),
                        serde_json::from_value(request["selected_work"].clone()).unwrap(),
                    )
                    .unwrap();
                assert_eq!(selection.streams.len(), 1);
                assert_eq!(selection.streams[0].session_ref, session);
            }
            drop(reader);
            let mut selected_envelope: serde_json::Value =
                serde_json::from_str(&envelope()).unwrap();
            selected_envelope["work_policy"] = accepted[0]["work_policy"].clone();
            event_support::write_test_file(observer.join("supersede-selected-session"), b"stale-A")
                .unwrap();
            std::fs::remove_file(observer.join("release")).unwrap();
            event_support::write_test_file(observer.join("numbered-passes"), b"4").unwrap();
            drop(held);
            let worker = machine.spawn_with_env(
                &project,
                &[
                    "__event-search-worker",
                    &selected_envelope.to_string(),
                    "--write-memory-limit-bytes",
                    &HIGH.to_string(),
                ],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            for pass in 1..=4 {
                await_marker(&observer, &format!("entered.{pass}"));
                if pass == 4 && unrelated_publication {
                    let other = machine.project("unrelated-stale-arrival");
                    write_session(
                        &machine.history_root(),
                        &other,
                        "unrelated-C",
                        &[("independent publication", "answer")],
                    );
                    let arrival = import_selected(
                        &machine,
                        &other,
                        "unrelated-stale-arrival",
                        HIGH,
                        "1",
                        &observer,
                    );
                    assert_eq!(arrival.code, 0, "{}", arrival.describe());
                    assert_eq!(str_at(&arrival.json(), "/search/worker"), "already_running");
                }
                event_support::write_test_file(
                    observer.join(format!("release.{pass}")),
                    b"continue bounded work",
                )
                .unwrap();
            }
            let output = worker.wait_with_output().unwrap();
            let attempts = std::fs::read_to_string(observer.join("superseded-selected-attempts"))
                .unwrap()
                .lines()
                .count();
            eprintln!(
                "stale reproduction unrelated_publication={unrelated_publication} attempts={attempts} status={:?} policy={}",
                output.status.code(),
                worker_policy(&machine.data_root)
            );
            assert_eq!(
                attempts, 4,
                "queue publication replenished the unchanged selection's allowance"
            );
            assert_eq!(
                output.status.code(),
                Some(1),
                "exhausted unfinished work must retain a nonzero outcome: {output:?}"
            );
            let retained = worker_policy(&machine.data_root);
            let requests = retained["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap();
            assert_eq!(
                requests.len(),
                1,
                "distinct accepted work was stranded: {retained}"
            );
            assert_eq!(requests[0]["selected_work"], accepted[0]["selected_work"]);
            assert_eq!(requests[0]["completed"], false);
            assert!(
                actual_budgets(&observer)
                    .iter()
                    .any(|batch| batch["session"] == "serviceable-B"),
                "exhausted A starved its distinct same-Context B selection"
            );
            if unrelated_publication {
                assert!(
                    actual_budgets(&observer)
                        .iter()
                        .any(|batch| batch["session"] == "unrelated-C")
                );
            }
        }
        // The third failed aggregate assessment must still hand off an accepted arrival
        // published before unlock, including a new physical selection in the same Context.
        {
            let machine = Machine::with_nomic();
            let project = machine.project("aggregate-read-arrival");
            write_session(
                &machine.history_root(),
                &project,
                "old-unreadable-selection",
                &[("older accepted work", "older answer")],
            );
            let observer = request_observer(&machine, "aggregate-read-observer");
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let initial = import_selected(
                &machine,
                &project,
                "aggregate-read-arrival",
                HIGH,
                "1",
                &observer,
            );
            assert_eq!(initial.code, 0, "{}", initial.describe());
            let original =
                worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                    [0]
                .clone();
            // Three owners of one accepted identity still receive three aggregate observations,
            // rather than consuming the whole retry allowance on their first failed read.
            let mut shared = worker_policy(&machine.data_root);
            let owners = shared["pending_background_request"]["write_requests"]
                .as_array_mut()
                .unwrap();
            for owner in 1..=2 {
                let mut request = original.clone();
                request["token"] = serde_json::json!(format!(
                    "{}-shared-{owner}",
                    original["token"].as_str().unwrap()
                ));
                owners.push(request);
            }
            write_event_search_worker_policy(&machine.data_root, &shared.to_string()).unwrap();
            std::fs::remove_file(observer.join("release")).unwrap();
            event_support::write_test_file(observer.join("numbered-passes"), b"2").unwrap();
            event_support::write_test_file(observer.join("refuse-aggregate-read"), b"refuse")
                .unwrap();
            drop(held);
            let worker = machine.spawn_with_envs(
                &project,
                &["__event-search-worker", &envelope()],
                &[(REQUEST_OBSERVER_ENV, &observer)],
            );
            await_marker(&observer, "entered.1");
            event_support::write_test_file(observer.join("release.1"), b"second assessment")
                .unwrap();
            await_marker(&observer, "entered.2");
            let path = write_session(
                &machine.history_root(),
                &project,
                "new-serviceable-selection",
                &[("accepted at exhausted assessment", "served answer")],
            );
            let text = std::fs::read_to_string(&path).unwrap();
            event_support::write_test_file(path, text.replace("2026-09-01", "2026-09-02")).unwrap();
            let arrival = import_selected(
                &machine,
                &project,
                "aggregate-read-arrival",
                HIGH,
                "1",
                &observer,
            );
            assert_eq!(arrival.code, 0, "{}", arrival.describe());
            assert_eq!(str_at(&arrival.json(), "/search/worker"), "already_running");
            let queued = worker_policy(&machine.data_root);
            assert_ne!(
                queued["pending_background_request"]["write_requests"][3]["selected_work"],
                original["selected_work"]
            );
            std::fs::remove_file(observer.join("refuse-aggregate-read")).unwrap();
            event_support::write_test_file(observer.join("release.2"), b"serve new identity")
                .unwrap();
            let output = worker.wait_with_output().unwrap();
            assert_eq!(output.status.code(), Some(1), "{output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stderr)
                    .matches("deterministic aggregate search read refusal")
                    .count(),
                3
            );
            let retained = worker_policy(&machine.data_root);
            let requests = retained["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap();
            assert_eq!(requests.len(), 3, "{retained}");
            for request in requests {
                assert_eq!(request["selected_work"], original["selected_work"]);
                assert_eq!(request["completed"], false);
                assert_eq!(request["memory_limit_bytes"], HIGH);
            }
            assert!(
                actual_budgets(&observer)
                    .iter()
                    .any(|batch| batch["session"] == "new-serviceable-selection")
            );
        }

        // Initial and completion reads must suppress their exact failed identity even when
        // the aggregate is caught up. An arrival owns a different selection.
        for completion_read in [false, true] {
            let machine = Machine::with_nomic();
            let project = machine.project("unreadable-selection");
            capture_b(&machine, &project);
            let observer = request_observer(&machine, "unreadable-selection-observer");
            let initial = import_selected(
                &machine,
                &project,
                "unreadable-selection",
                HIGH,
                "1",
                &observer,
            );
            assert_eq!(initial.code, 0, "{}", initial.describe());
            await_acknowledgement(&machine);
            machine.wait_for_search_worker_exit();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let path = write_session(
                &machine.history_root(),
                &project,
                "opaque-selected",
                &[("unused", "unused")],
            );
            let opaque = serde_json::json!({
                "type": "opaque-lifecycle-fixture", "uuid": uuid::Uuid::new_v4().to_string(),
                "sessionId": "opaque-selected", "cwd": project.display().to_string(),
                "timestamp": "2026-09-02T09:00:00Z",
            });
            event_support::write_test_file(path, format!("{opaque}\n")).unwrap();
            let first = import_selected(
                &machine,
                &project,
                "unreadable-selection",
                HIGH,
                "1",
                &observer,
            );
            assert_eq!(first.code, 0, "{}", first.describe());
            let mut saved = worker_policy(&machine.data_root);
            drop(held);
            let completed = machine.run_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            assert_eq!(completed.code, 0, "{}", completed.describe());
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let store = context_graph::Store::open(context_graph::StoreConfig {
                db_path: machine.db_path(),
                default_text_embedder: None,
                write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
                ..Default::default()
            })
            .unwrap();
            let request = &mut saved["pending_background_request"]["write_requests"][0];
            let context = serde_json::from_value(request["context_id"].clone()).unwrap();
            let policy = serde_json::from_value(request["work_policy"].clone()).unwrap();
            let selection = store.select_event_search_work(context, policy).unwrap();
            request["selected_work"] = serde_json::to_value(
                store
                    .persist_event_search_selected_work(context, &selection)
                    .unwrap(),
            )
            .unwrap();
            drop(store);
            write_event_search_worker_policy(&machine.data_root, &saved.to_string()).unwrap();
            let original = saved["pending_background_request"]["write_requests"][0].clone();
            // Status independently establishes the completed aggregate before the selected read.
            assert_ready(&machine, &project, "unreadable-selection");
            std::fs::remove_file(observer.join("actual-write-memory-batches")).ok();
            std::fs::remove_file(observer.join("release")).unwrap();
            std::fs::remove_file(observer.join("entered")).ok();
            event_support::write_test_file(observer.join("numbered-passes"), b"1").unwrap();
            let assessment = machine.root.join("completion-read-assessment");
            std::fs::create_dir_all(&assessment).unwrap();
            event_support::write_test_file(
                assessment.join("pre-catch-up-release"),
                b"initial read may run",
            )
            .unwrap();
            if completion_read {
                event_support::write_test_file(
                    assessment.join("pause-after-work-assessment"),
                    b"pause",
                )
                .unwrap();
            } else {
                let db = contextdb_engine::Database::open(machine.db_path()).unwrap();
                db.execute(
                    "DELETE FROM event_search_request_selections WHERE request_id = $identity",
                    &std::collections::HashMap::from([(
                        "identity".into(),
                        contextdb_core::Value::Uuid(
                            uuid::Uuid::parse_str(original["selected_work"].as_str().unwrap())
                                .unwrap(),
                        ),
                    )]),
                )
                .unwrap();
                db.execute(
                    "DELETE FROM event_search_request_headers WHERE request_id = $identity",
                    &std::collections::HashMap::from([(
                        "identity".into(),
                        contextdb_core::Value::Uuid(
                            uuid::Uuid::parse_str(original["selected_work"].as_str().unwrap())
                                .unwrap(),
                        ),
                    )]),
                )
                .unwrap();
            }
            drop(held);
            let worker = machine.spawn_with_envs(
                &project,
                &["__event-search-worker", &envelope()],
                &[
                    (REQUEST_OBSERVER_ENV, &observer),
                    ("CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER", &assessment),
                ],
            );
            if completion_read {
                await_marker(&assessment, "work-assessment-complete");
                let db = contextdb_engine::Database::open(machine.db_path()).unwrap();
                db.execute(
                    "DELETE FROM event_search_request_selections WHERE request_id = $identity",
                    &std::collections::HashMap::from([(
                        "identity".into(),
                        contextdb_core::Value::Uuid(
                            uuid::Uuid::parse_str(original["selected_work"].as_str().unwrap())
                                .unwrap(),
                        ),
                    )]),
                )
                .unwrap();
                db.execute(
                    "DELETE FROM event_search_request_headers WHERE request_id = $identity",
                    &std::collections::HashMap::from([(
                        "identity".into(),
                        contextdb_core::Value::Uuid(
                            uuid::Uuid::parse_str(original["selected_work"].as_str().unwrap())
                                .unwrap(),
                        ),
                    )]),
                )
                .unwrap();
                drop(db);
                event_support::write_test_file(
                    assessment.join("work-assessment-release"),
                    b"read deleted header",
                )
                .unwrap();
            }
            await_marker(&observer, "entered.1");
            assert!(
                actual_budgets(&observer).is_empty(),
                "failure precedes projection"
            );
            let other = machine.project("read-failure-arrival");
            write_session(
                &machine.history_root(),
                &other,
                "session-C",
                &[("accepted after read failure", "served answer")],
            );
            let arrival = import_selected(
                &machine,
                &other,
                "read-failure-arrival",
                HIGH,
                "1",
                &observer,
            );
            assert_eq!(arrival.code, 0, "{}", arrival.describe());
            assert_eq!(str_at(&arrival.json(), "/search/worker"), "already_running");
            event_support::write_test_file(observer.join("release.1"), b"serve arrival").unwrap();
            let output = worker.wait_with_output().unwrap();
            assert_eq!(output.status.code(), Some(1), "{output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stderr)
                    .matches("pending search request has no durable selected-work identity")
                    .count(),
                1
            );
            let retained = worker_policy(&machine.data_root);
            let requests = retained["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap();
            assert_eq!(requests.len(), 1, "{retained}");
            assert_eq!(requests[0]["token"], original["token"]);
            assert_eq!(requests[0]["selected_work"], original["selected_work"]);
            assert_eq!(requests[0]["completed"], false);
            assert_eq!(requests[0]["memory_limit_bytes"], HIGH);
            assert_selection_row_counts(&machine, 0, 0);
            assert_ready(&machine, &other, "read-failure-arrival");
        }

        // Distinct accepted no-text frontiers must not spend a later selection's
        // unchanged-work allowance while the aggregate newest-session assessment is stable.
        {
            let machine = Machine::with_nomic();
            let project = machine.project("distinct-no-text-selections");
            write_session(
                &machine.history_root(),
                &project,
                "bootstrap",
                &[("matching generation", "answer")],
            );
            let observer = request_observer(&machine, "distinct-no-text-observer");
            let initial =
                import_selected(&machine, &project, "distinct-no-text", HIGH, "1", &observer);
            assert_eq!(initial.code, 0, "{}", initial.describe());
            await_acknowledgement(&machine);
            machine.wait_for_search_worker_exit();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let mut accepted = Vec::new();
            for index in 1..=5 {
                let session = format!("selected-{index}");
                let path = write_session(
                    &machine.history_root(),
                    &project,
                    &session,
                    &[("later searchable selection", "answer")],
                );
                if index < 5 {
                    let opaque = serde_json::json!({
                        "type": "opaque-lifecycle-fixture", "uuid": uuid::Uuid::new_v4().to_string(),
                        "sessionId": session, "cwd": project.display().to_string(),
                        "timestamp": format!("2026-09-{:02}T09:00:00Z", index + 1),
                    });
                    event_support::write_test_file(&path, format!("{opaque}\n")).unwrap();
                } else {
                    let text = std::fs::read_to_string(&path).unwrap();
                    event_support::write_test_file(&path, text.replace("2026-09-01", "2026-09-06"))
                        .unwrap();
                }
                let imported =
                    import_selected(&machine, &project, "distinct-no-text", HIGH, "1", &observer);
                assert_eq!(imported.code, 0, "{}", imported.describe());
                assert_eq!(
                    str_at(&imported.json(), "/search/worker"),
                    "already_running",
                    "selection {index} was not accepted: {}",
                    imported.describe()
                );
                let pending = worker_policy(&machine.data_root);
                let requests = pending["pending_background_request"]["write_requests"]
                    .as_array()
                    .unwrap();
                assert_eq!(requests.len(), index);
                accepted.push(requests[index - 1].clone());
                let reader = context_graph::ConsumerReader::open(
                    machine.db_path(),
                    context_graph::event_history_reader_declaration(),
                )
                .unwrap();
                let request = &accepted[index - 1];
                let selection = reader
                    .event_search_selected_work(
                        serde_json::from_value(request["context_id"].clone()).unwrap(),
                        serde_json::from_value(request["selected_work"].clone()).unwrap(),
                    )
                    .unwrap();
                assert_eq!(selection.streams.len(), 2);
                assert_eq!(selection.streams[1].session_ref, "bootstrap");
                assert_eq!(selection.streams[1].frontier, 2);
                assert_eq!(selection.streams[0].session_ref, session);
                assert_eq!(selection.streams[0].frontier, if index < 5 { 1 } else { 2 });
            }
            drop(held);
            let worker = machine.run_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            assert_eq!(worker.code, 0, "{}", worker.describe());
            assert!(
                worker_policy(&machine.data_root)["pending_background_request"].is_null(),
                "accepted distinct selections stranded: {}",
                worker_policy(&machine.data_root)
            );
            assert_selection_counts(&machine, 0);
            assert_ready(&machine, &project, "distinct-no-text");
            assert!(
                actual_budgets(&observer)
                    .iter()
                    .any(|batch| batch["session"] == "selected-5"),
                "later text selection was not served"
            );
        }
        for (racing, cleanup_failure, same_context) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, false, true),
            (true, true, true),
        ] {
            let machine = Machine::with_nomic();
            let project = machine.project("refused-only");
            write_session(
                &machine.history_root(),
                &project,
                "refused-session",
                &[("retained history", "answer")],
            );
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer = request_observer(&machine, "refused-restart");
            let imported = import(&machine, &project, "refused-only", HIGH, &observer);
            assert_eq!(imported.code, 0, "{}", imported.describe());
            let refused_limit = if cleanup_failure { 1 } else { LOW };
            let refused = refuse_queue(&machine, refused_limit);
            assert_selection_counts(&machine, 1);
            let original = &refused["pending_background_request"]["write_requests"][0];
            let old_context: context_graph::ContextId =
                serde_json::from_value(original["context_id"].clone()).unwrap();
            let spec = context_graph::EventSearchWorkSpec {
                profile: context_graph::EventSearchProfile::default(),
                embedding_space_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
                encoding_profile_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
            };
            let old_frontier = same_context.then(|| {
                let reader = context_graph::ConsumerReader::open(
                    machine.db_path(),
                    context_graph::event_history_reader_declaration(),
                )
                .unwrap();
                let frontier = reader
                    .event_search_selected_work(
                        old_context,
                        serde_json::from_value(original["selected_work"].clone()).unwrap(),
                    )
                    .unwrap();
                assert_eq!(frontier.streams.len(), 1);
                assert_eq!(frontier.streams[0].session_ref, "refused-session");
                assert_eq!(frontier.streams[0].frontier, 2);
                assert!(
                    !reader
                        .event_search_selected_work_complete(old_context, &spec, &frontier)
                        .unwrap()
                );
                frontier
            });
            let mut arrival_frontier = None;
            drop(held);
            if racing {
                event_support::write_test_file(
                    observer.join("pause-before-selection-cleanup"),
                    b"pause cleanup",
                )
                .unwrap();
            }
            let mut worker = machine.spawn_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            if racing {
                await_marker(&observer, "selection-cleanup-entered");
                if !cleanup_failure {
                    worker
                        .kill()
                        .expect("interrupt after durable refusal before retirement");
                    worker.wait().expect("reap interrupted worker");
                    assert_selection_counts(&machine, 1);
                    assert_eq!(
                        worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                            [0]["memory_limit_bytes"],
                        LOW
                    );
                    std::fs::remove_file(observer.join("selection-cleanup-entered")).unwrap();
                    worker = machine.spawn_with_env(
                        &project,
                        &[
                            "__event-search-worker",
                            &envelope(),
                            "--write-memory-limit-bytes",
                            &HIGH.to_string(),
                        ],
                        REQUEST_OBSERVER_ENV,
                        &observer,
                    );
                    await_marker(&observer, "selection-cleanup-entered");
                }
                let other = if same_context {
                    project.clone()
                } else {
                    machine.project("shutdown-arrival")
                };
                let context = if same_context {
                    "refused-only"
                } else {
                    "shutdown-arrival"
                };
                write_session(
                    &machine.history_root(),
                    &other,
                    "accepted-session",
                    &[("accepted at shutdown", "served answer")],
                );
                let arrived = import_selected(&machine, &other, context, HIGH, "1", &observer);
                assert_eq!(arrived.code, 0, "{}", arrived.describe());
                assert_eq!(str_at(&arrived.json(), "/search/worker"), "already_running");
                if same_context {
                    let accepted = worker_policy(&machine.data_root);
                    assert_eq!(
                        accepted["pending_background_request"]["write_memory_refusals"],
                        refused["pending_background_request"]["write_memory_refusals"]
                    );
                    let old = &accepted["pending_background_request"]["write_requests"][0];
                    assert_eq!(old["token"], original["token"]);
                    assert_eq!(old["selected_work"], original["selected_work"]);
                    assert_eq!(old["memory_refusal"], original["memory_refusal"]);
                    assert_eq!(old["refusal_superseded"], false);
                    assert_eq!(old["selection_retired"], false);
                    assert_eq!(old["memory_limit_bytes"], refused_limit);
                    let reader = context_graph::ConsumerReader::open(
                        machine.db_path(),
                        context_graph::event_history_reader_declaration(),
                    )
                    .unwrap();
                    assert!(
                        !reader
                            .event_search_selected_work_complete(
                                old_context,
                                &spec,
                                old_frontier.as_ref().unwrap(),
                            )
                            .unwrap()
                    );
                    arrival_frontier =
                        Some(
                            reader
                                .event_search_selected_work(
                                    old_context,
                                    serde_json::from_value(
                                        accepted["pending_background_request"]["write_requests"][1]
                                            ["selected_work"]
                                            .clone(),
                                    )
                                    .unwrap(),
                                )
                                .unwrap(),
                        );
                }
                // Acceptance's global reconciliation must retain the old LOW selection.
                assert_selection_counts(&machine, 2);
                if cleanup_failure {
                    capture_b(&machine, &other);
                    let overlapping =
                        import_selected(&machine, &other, context, HIGH, "all", &observer);
                    assert_eq!(overlapping.code, 0, "{}", overlapping.describe());
                    assert_eq!(
                        str_at(&overlapping.json(), "/search/worker"),
                        "already_running"
                    );
                    assert_selection_row_counts(&machine, 3, if same_context { 5 } else { 4 });
                }
                event_support::write_test_file(
                    observer.join("release-selection-cleanup"),
                    b"serve the arrival",
                )
                .unwrap();
            }
            assert_eq!(
                worker.wait().ok().map(|status| status.code()),
                Some(Some(i32::from(cleanup_failure))),
                "refused-only worker did not terminate: racing={racing}, cleanup_failure={cleanup_failure}, same_context={same_context}, policy={}, admissions={:?}, cleanup={:?}, projection_error={:?}",
                worker_policy(&machine.data_root),
                actual_budgets(&observer),
                std::fs::read_to_string(observer.join("cleanup-selection-limits")),
                std::fs::read_to_string(observer.join("projection-error")),
            );
            let output = worker.wait_with_output().unwrap();
            assert!(
                output.status.success() != cleanup_failure,
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let cleanups = std::fs::read_to_string(observer.join("cleanup-write-memory-limits"))
                .unwrap()
                .lines()
                .map(|limit| limit.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(cleanups[0], refused_limit);
            if cleanup_failure {
                assert!(
                    cleanups.contains(&HIGH),
                    "independent HIGH cleanup never admitted: {cleanups:?}"
                );
                let identities =
                    std::fs::read_to_string(observer.join("cleanup-selection-limits")).unwrap();
                let old_selection =
                    refused["pending_background_request"]["write_requests"][0]["selected_work"]
                        .as_str()
                        .unwrap();
                assert!(
                    identities
                        .lines()
                        .filter(|line| line.starts_with(old_selection))
                        .all(|line| line.ends_with(&format!(" {refused_limit}"))),
                    "{identities}"
                );
            }
            assert_selection_row_counts(
                &machine,
                if cleanup_failure { 1 } else { 0 },
                if cleanup_failure { 1 } else { 0 },
            );
            let retained = worker_policy(&machine.data_root);
            if same_context {
                let requests = retained["pending_background_request"]["write_requests"]
                    .as_array()
                    .unwrap();
                assert_eq!(requests.len(), 1, "{retained}");
                let old = &requests[0];
                assert_eq!(old["token"], original["token"]);
                assert_eq!(old["selected_work"], original["selected_work"]);
                assert_eq!(old["work_policy"], original["work_policy"]);
                assert_eq!(old["memory_limit_bytes"], refused_limit);
                assert_eq!(old["completed"], true);
                assert_eq!(old["refused_memory"], true);
                assert_eq!(old["refusal_superseded"], cleanup_failure);
                assert_eq!(old["selection_retired"], !cleanup_failure);
                assert_eq!(old["memory_refusal"], original["memory_refusal"]);
                assert_eq!(
                    old["memory_refusal"]["Recorded"]["refusal"],
                    refused["pending_background_request"]["write_memory_refusals"]
                        [old_context.to_string()]
                );
                if cleanup_failure {
                    assert_eq!(
                        retained["pending_background_request"]["write_memory_refusals"],
                        serde_json::json!({})
                    );
                } else {
                    assert_eq!(
                        retained["pending_background_request"]["write_memory_refusals"],
                        refused["pending_background_request"]["write_memory_refusals"]
                    );
                }
                let reader = context_graph::ConsumerReader::open(
                    machine.db_path(),
                    context_graph::event_history_reader_declaration(),
                )
                .unwrap();
                assert_eq!(
                    reader
                        .event_search_selected_work_complete(
                            old_context,
                            &spec,
                            old_frontier.as_ref().unwrap(),
                        )
                        .unwrap(),
                    cleanup_failure
                );
                assert!(
                    reader
                        .event_search_selected_work_complete(
                            old_context,
                            &spec,
                            arrival_frontier.as_ref().unwrap(),
                        )
                        .unwrap()
                );
                if cleanup_failure {
                    let selection = reader
                        .event_search_selected_work(
                            old_context,
                            serde_json::from_value(old["selected_work"].clone()).unwrap(),
                        )
                        .unwrap();
                    assert_eq!(selection.streams.len(), 1);
                    assert_eq!(selection.streams[0].session_ref, "refused-session");
                    assert_eq!(selection.streams[0].frontier, 2);
                }
                drop(reader);
                let status = machine.run(&project, &["status", "--context", "refused-only"]);
                assert_eq!(status.code, 0, "{}", status.describe());
                assert_eq!(
                    str_at(&status.json(), "/history/search/status"),
                    if cleanup_failure { "ready" } else { "failed" }
                );
                assert_eq!(
                    u64_at(&status.json(), "/history/search/covered_events"),
                    if cleanup_failure { 6 } else { 2 }
                );
                if !cleanup_failure {
                    assert_eq!(
                        str_at(&status.json(), "/history/search/last_failure"),
                        context_graph::CgError::WriteDidNotFitTheMemoryLimit {
                            requested_bytes: 42,
                            limit_bytes: refused_limit,
                        }
                        .to_string()
                    );
                }
                assert!(
                    actual_budgets(&observer)
                        .iter()
                        .any(|batch| batch["session"] == "accepted-session")
                );
                continue;
            }
            assert_eq!(
                retained["pending_background_request"]["write_memory_refusals"],
                refused["pending_background_request"]["write_memory_refusals"]
            );
            assert_eq!(
                retained["pending_background_request"]["write_requests"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            let status = machine.run(&project, &["status", "--context", "refused-only"]);
            assert_eq!(status.code, 0, "{}", status.describe());
            assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
            if racing {
                assert_ready(
                    &machine,
                    &machine.project("shutdown-arrival"),
                    "shutdown-arrival",
                );
                if cleanup_failure {
                    let status = machine.run(
                        &machine.project("shutdown-arrival"),
                        &["status", "--context", "shutdown-arrival"],
                    );
                    assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 4);
                    let requests = retained["pending_background_request"]["write_requests"]
                        .as_array()
                        .unwrap();
                    assert_eq!(
                        requests.len(),
                        1,
                        "unrelated completed HIGH requests retained: {retained}"
                    );
                    assert_eq!(
                        requests[0]["token"],
                        refused["pending_background_request"]["write_requests"][0]["token"]
                    );
                    assert_eq!(requests[0]["completed"], true);
                    assert_eq!(requests[0]["selection_retired"], false);
                }
            } else {
                assert!(
                    budgets(&observer).is_empty(),
                    "retained refusal must not reopen projection"
                );
            }
        }

        // Exhaust a real non-memory coordination failure whose matching failure reopen is
        // refused by a live writer. An accepted arrival still observes the worker lock.
        for (same_context, competing_owner) in [(false, false), (true, false), (false, true)] {
            let machine = Machine::with_nomic();
            let project = machine.project("unrecorded-shutdown");
            capture_b(&machine, &project);
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer = request_observer(&machine, "unrecorded-shutdown-observer");
            let first = import_selected(
                &machine,
                &project,
                "unrecorded-shutdown",
                HIGH,
                "1",
                &observer,
            );
            assert_eq!(first.code, 0, "{}", first.describe());
            let original =
                worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                    [0]
                .clone();
            std::fs::remove_file(observer.join("release")).unwrap();
            event_support::write_test_file(observer.join("numbered-passes"), b"3").unwrap();
            event_support::write_test_file(
                observer.join("refuse-selected-coordination-count"),
                b"10",
            )
            .unwrap();
            event_support::write_test_file(
                observer.join("hold-writer-during-failure-record"),
                b"hold",
            )
            .unwrap();
            if competing_owner {
                event_support::write_test_file(
                    observer.join("pause-before-failed-worker-relock"),
                    b"pause",
                )
                .unwrap();
            }
            drop(held);
            let worker = machine.spawn_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            for pass in 1..=3 {
                await_marker(&observer, "failure-record-entered");
                let writer = context_graph::Store::open(context_graph::StoreConfig {
                    db_path: machine.db_path(),
                    default_text_embedder: None,
                    write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
                    ..Default::default()
                })
                .expect("external writer owns the failure reopen");
                event_support::write_test_file(
                    observer.join("failure-record-continue"),
                    b"refuse failure reopen",
                )
                .unwrap();
                await_marker(&observer, "failure-record-returned");
                drop(writer);
                event_support::write_test_file(
                    observer.join("failure-writer-released"),
                    b"released",
                )
                .unwrap();
                await_marker(&observer, &format!("entered.{pass}"));
                for marker in [
                    "failure-record-entered",
                    "failure-record-continue",
                    "failure-record-returned",
                    "failure-writer-released",
                ] {
                    std::fs::remove_file(observer.join(marker)).unwrap();
                }
                assert_eq!(
                    std::fs::read_to_string(observer.join("refuse-selected-coordination-count"))
                        .unwrap(),
                    (10 - pass).to_string()
                );
                if pass < 3 {
                    event_support::write_test_file(
                        observer.join(format!("release.{pass}")),
                        b"retry",
                    )
                    .unwrap();
                }
            }
            let other = if same_context {
                project.clone()
            } else {
                machine.project("unrecorded-arrival")
            };
            let context = if same_context {
                "unrecorded-shutdown"
            } else {
                "unrecorded-arrival"
            };
            write_session(
                &machine.history_root(),
                &other,
                "session-C",
                &[("accepted at unrecorded failure", "served answer")],
            );
            let arrival = import_selected(&machine, &other, context, HIGH, "1", &observer);
            assert_eq!(arrival.code, 0, "{}", arrival.describe());
            assert_eq!(str_at(&arrival.json(), "/search/worker"), "already_running");
            // Keep seven injected failures available: the old allowance must not restart under
            // the new queue token. Let the newly selected session run without that injection.
            std::fs::rename(
                observer.join("refuse-selected-coordination-count"),
                observer.join("remaining-old-refusals"),
            )
            .unwrap();
            std::fs::remove_file(observer.join("hold-writer-during-failure-record")).unwrap();
            event_support::write_test_file(observer.join("release.3"), b"serve arrival").unwrap();
            let competing_lock = competing_owner.then(|| {
                await_marker(&observer, "failed-worker-relock-entered");
                let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
                    .expect("competing worker owns the released lock");
                let startup = machine.run(&project, &["__event-search-worker", &envelope()]);
                assert_eq!(startup.code, 0, "{}", startup.describe());
                event_support::write_test_file(
                    observer.join("release-failed-worker-relock"),
                    b"lose lock",
                )
                .unwrap();
                held
            });
            let output = worker.wait_with_output().unwrap();
            assert_eq!(output.status.code(), Some(1), "{output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stderr)
                    .matches("deterministic selected-work coordination refusal")
                    .count(),
                3
            );
            let retained = worker_policy(&machine.data_root);
            let requests = retained["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap();
            assert_eq!(
                requests.len(),
                if competing_owner { 2 } else { 1 },
                "{retained}"
            );
            assert_eq!(requests[0]["token"], original["token"]);
            assert_eq!(requests[0]["selected_work"], original["selected_work"]);
            assert_eq!(requests[0]["completed"], false);
            assert_eq!(requests[0]["memory_limit_bytes"], HIGH);
            if competing_owner {
                assert_eq!(requests[1]["completed"], false);
                assert!(actual_budgets(&observer).is_empty());
                assert_selection_counts(&machine, 2);
                drop(competing_lock);
                let successor = machine.run(&project, &["__event-search-worker", &envelope()]);
                assert_eq!(successor.code, 0, "{}", successor.describe());
                assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
                assert_selection_counts(&machine, 0);
                assert_ready(&machine, &other, context);
                continue;
            }
            let actual = actual_budgets(&observer);
            assert!(!actual.is_empty(), "arrival stranded: {retained}");
            assert!(
                actual
                    .iter()
                    .all(|batch| batch["session"] == "session-C" && batch["limit"] == HIGH),
                "{actual:?}"
            );
            let status = machine.run(&other, &["status", "--context", context]);
            assert_eq!(status.code, 0, "{}", status.describe());
            assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
            assert_selection_counts(&machine, 1);
        }
    }

    #[test]
    fn completion_cleanup_keeps_saved_ceilings_including_shared_lower_limit() {
        for (limit, startup, shared, physical_shared) in [
            (HIGH, 1, false, false),
            (LOW, HIGH, false, false),
            (HIGH, HIGH, true, false),
            (HIGH, HIGH, true, true),
        ] {
            let machine = Machine::with_nomic();
            let project = machine.project("cleanup-ceiling");
            write_session(
                &machine.history_root(),
                &project,
                "cleanup-session",
                &[("request ceiling through cleanup", "answer")],
            );
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer = request_observer(&machine, "cleanup-ceiling");
            let first = import(&machine, &project, "cleanup-ceiling", limit, &observer);
            assert_eq!(first.code, 0, "{}", first.describe());
            if shared {
                let second = import(&machine, &project, "cleanup-ceiling", LOW, &observer);
                assert_eq!(second.code, 0, "{}", second.describe());
            }
            let mut accepted = worker_policy(&machine.data_root);
            if physical_shared {
                let identity =
                    accepted["pending_background_request"]["write_requests"][0]["selected_work"]
                        .clone();
                accepted["pending_background_request"]["write_requests"][1]["selected_work"] =
                    identity;
                write_event_search_worker_policy(&machine.data_root, &accepted.to_string())
                    .unwrap();
            }
            drop(held);
            let resumed = machine.run_with_env(
                &project,
                &[
                    "__event-search-worker",
                    &envelope(),
                    "--write-memory-limit-bytes",
                    &startup.to_string(),
                ],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            assert_eq!(resumed.code, 0, "{}", resumed.describe());
            let expected = if shared { LOW } else { limit };
            let cleanup_admissions =
                std::fs::read_to_string(observer.join("cleanup-selection-limits")).unwrap();
            for request in accepted["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
            {
                let selection = request["selected_work"].as_str().unwrap();
                let ceiling = if physical_shared {
                    LOW
                } else {
                    request["memory_limit_bytes"].as_u64().unwrap() as usize
                };
                assert!(
                    cleanup_admissions
                        .lines()
                        .any(|line| line == format!("{selection} {ceiling}")),
                    "{cleanup_admissions}"
                );
            }
            assert!(
                budgets(&observer)
                    .iter()
                    .all(|(_, actual)| *actual == expected)
            );
            assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
            assert_selection_counts(&machine, 0);
            assert_ready(&machine, &project, "cleanup-ceiling");
            if !shared && limit == LOW {
                // A missing nonretired frontier is a typed failure, never empty=complete.
                write_event_search_worker_policy(&machine.data_root, &accepted.to_string())
                    .unwrap();
                let missing = refuse_queue(&machine, 1);
                let missing_frontier = machine.run(
                    &project,
                    &[
                        "import",
                        ".",
                        "--context",
                        "cleanup-ceiling",
                        "--search",
                        "background",
                    ],
                );
                assert_eq!(missing_frontier.code, 1, "{}", missing_frontier.describe());
                assert_eq!(
                    str_at(&missing_frontier.json(), "/condition"),
                    "MemoryError"
                );
                assert_eq!(str_at(&missing_frontier.json(), "/state"), "error");
                assert!(
                    str_at(&missing_frontier.json(), "/message")
                        .contains("no durable selected-work identity")
                );
                assert_eq!(
                    worker_policy(&machine.data_root)["pending_background_request"],
                    missing["pending_background_request"]
                );

                // Resume a real durable completed selection whose retirement has not committed.
                // Its saved one-byte cleanup ceiling must reach the real engine condition.
                let store = context_graph::Store::open(context_graph::StoreConfig {
                    db_path: machine.db_path(),
                    default_text_embedder: None,
                    write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
                    ..Default::default()
                })
                .unwrap();
                let request = &mut accepted["pending_background_request"]["write_requests"][0];
                let context = serde_json::from_value(request["context_id"].clone()).unwrap();
                let selection = store
                    .select_event_search_work(
                        context,
                        serde_json::from_value(request["work_policy"].clone()).unwrap(),
                    )
                    .unwrap();
                request["selected_work"] = serde_json::to_value(
                    store
                        .persist_event_search_selected_work(context, &selection)
                        .unwrap(),
                )
                .unwrap();
                drop(store);
                write_event_search_worker_policy(&machine.data_root, &accepted.to_string())
                    .unwrap();
                refuse_queue(&machine, 1);
                assert_selection_counts(&machine, 1);
                let refused = machine.run(
                    &project,
                    &[
                        "import",
                        ".",
                        "--context",
                        "cleanup-ceiling",
                        "--search",
                        "background",
                    ],
                );
                assert_eq!(refused.code, 1, "{}", refused.describe());
                assert_eq!(str_at(&refused.json(), "/state"), "write_memory_limit");
                assert_eq!(
                    str_at(&refused.json(), "/condition"),
                    "WriteDidNotFitTheMemoryLimit"
                );
            }
        }

        // A completed/refused LOW owner still governs its exact physical selection while a
        // HIGH owner is unfinished. A shared refusal completes both owners finitely, while
        // a distinct HIGH selection in the same Context retains independent service.
        for already_refused in [false, true] {
            let machine = Machine::with_nomic();
            let project = machine.project("shared-refused-ceiling");
            write_session(
                &machine.history_root(),
                &project,
                "shared-low-session",
                &[("shared physical selected frontier", "answer")],
            );
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer = request_observer(&machine, "shared-refused-ceiling");
            for limit in [LOW, HIGH] {
                let accepted = import_selected(
                    &machine,
                    &project,
                    "shared-refused-ceiling",
                    limit,
                    "1",
                    &observer,
                );
                assert_eq!(accepted.code, 0, "{}", accepted.describe());
            }
            let path = write_session(
                &machine.history_root(),
                &project,
                "independent-high-session",
                &[("independent physical selected frontier", "answer")],
            );
            let text = std::fs::read_to_string(&path).unwrap();
            event_support::write_test_file(path, text.replace("2026-09-01", "2026-09-02")).unwrap();
            let independent = import_selected(
                &machine,
                &project,
                "shared-refused-ceiling",
                HIGH,
                "1",
                &observer,
            );
            assert_eq!(independent.code, 0, "{}", independent.describe());
            let mut accepted = worker_policy(&machine.data_root);
            let requests = accepted["pending_background_request"]["write_requests"]
                .as_array_mut()
                .unwrap();
            assert_eq!(requests.len(), 3);
            let shared_identity = requests[0]["selected_work"].clone();
            requests[1]["selected_work"] = shared_identity.clone();
            assert_ne!(requests[2]["selected_work"], shared_identity);
            requests[0]["completed"] = already_refused.into();
            requests[0]["refused_memory"] = already_refused.into();
            if already_refused {
                requests[0]["memory_refusal"] = serde_json::json!({"Recorded": {"ordinal": 1, "refusal": {"requested_bytes": 42, "limit_bytes": LOW}}});
            }
            write_event_search_worker_policy(&machine.data_root, &accepted.to_string()).unwrap();
            event_support::write_test_file(
                observer.join("refuse-next-selected-memory"),
                LOW.to_string(),
            )
            .unwrap();
            drop(held);
            let resumed = machine.run_with_env(
                &project,
                &[
                    "__event-search-worker",
                    &envelope(),
                    "--write-memory-limit-bytes",
                    &HIGH.to_string(),
                ],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            assert_eq!(resumed.code, 0, "{}", resumed.describe());
            let admissions = budgets(&observer);
            assert_eq!(admissions.first().unwrap().1, LOW, "{admissions:?}");
            let cleanup =
                std::fs::read_to_string(observer.join("cleanup-selection-limits")).unwrap();
            assert!(
                cleanup
                    .lines()
                    .any(|line| line == format!("{} {LOW}", shared_identity.as_str().unwrap())),
                "{cleanup}"
            );
            let retained = worker_policy(&machine.data_root);
            let refused = retained["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap();
            assert_eq!(refused.len(), 2, "{retained}");
            for (request, saved) in refused.iter().zip([LOW, HIGH]) {
                assert_eq!(request["selected_work"], shared_identity);
                assert_eq!(request["memory_limit_bytes"], saved);
                assert_eq!(request["completed"], true);
                assert_eq!(request["refused_memory"], true);
                assert_eq!(request["selection_retired"], true);
            }
            let actual = actual_budgets(&observer);
            assert!(
                !actual.is_empty(),
                "independent HIGH selection was stranded"
            );
            assert!(
                actual
                    .iter()
                    .all(|batch| batch["session"] == "independent-high-session"
                        && batch["limit"] == HIGH),
                "{actual:?}"
            );
            assert_selection_counts(&machine, 0);
            let status = machine.run(&project, &["status", "--context", "shared-refused-ceiling"]);
            assert_eq!(status.code, 0, "{}", status.describe());
            assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
            let restarted = machine.run(&project, &["__event-search-worker", &envelope()]);
            assert_eq!(restarted.code, 0, "{}", restarted.describe());
            assert_eq!(
                worker_policy(&machine.data_root)["pending_background_request"],
                retained["pending_background_request"]
            );
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
                .expect("shared refusal released the worker lock");
        }
    }

    #[test]
    fn a_completed_newest_one_retry_clears_its_refusal_without_embedding_older_history() {
        struct ReleaseBackgroundTurn(std::os::unix::net::UnixStream);
        impl Drop for ReleaseBackgroundTurn {
            fn drop(&mut self) {
                use std::io::Write;
                let _ = self.0.write_all(&[1]);
            }
        }

        for retired in [false, true] {
            let machine = Machine::with_nomic();
            let project = machine.project("newest-one-refusal");
            capture_b(&machine, &project);
            write_session(
                &machine.history_root(),
                &project,
                "selected-newest-session",
                &[("complete only this selected session", "selected answer")],
            );
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer = request_observer(&machine, "newest-one-refusal");
            let accepted = import_selected(
                &machine,
                &project,
                "newest-one-refusal",
                LOW,
                "1",
                &observer,
            );
            assert_eq!(accepted.code, 0, "{}", accepted.describe());
            let original =
                worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                    [0]
                .clone();
            assert_selection_row_counts(&machine, 1, 1);
            event_support::write_test_file(
                observer.join("refuse-next-selected-memory"),
                LOW.to_string(),
            )
            .unwrap();
            if !retired {
                event_support::write_test_file(
                    observer.join("pause-before-selection-cleanup"),
                    b"retain frontier",
                )
                .unwrap();
            }
            drop(held);
            let mut worker = machine.spawn_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            if retired {
                let output = worker.wait_with_output().unwrap();
                assert!(output.status.success(), "{output:?}");
            } else {
                await_marker(&observer, "selection-cleanup-entered");
                worker.kill().unwrap();
                worker.wait().unwrap();
            }
            let refused = worker_policy(&machine.data_root);
            assert_eq!(
                refused["pending_background_request"]["write_requests"][0]["token"],
                original["token"]
            );
            assert_eq!(
                refused["pending_background_request"]["write_requests"][0]["selection_retired"],
                retired
            );
            // The retry may kick detached housekeeping. Hold that actor at its
            // existing pre-catch-up seam while asserting the saved request, so
            // selection retirement cannot race the policy snapshot.
            let housekeeping_root = tempfile::Builder::new()
                .prefix("cgh")
                .tempdir_in("/tmp")
                .expect("short housekeeping socket directory");
            let barrier = housekeeping_root.path().to_path_buf();
            let housekeeping = (!retired).then(|| {
                std::os::unix::net::UnixListener::bind(barrier.join("pre-catch-up.sock"))
                    .expect("bind housekeeping rendezvous")
            });
            let high = HIGH.to_string();
            let arguments = [
                "import",
                ".",
                "--context",
                "newest-one-refusal",
                "--search",
                "wait",
                "--search-sessions",
                "1",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &high,
            ];
            let retry = if retired {
                machine.run(&project, &arguments)
            } else {
                machine.run_with_env(
                    &project,
                    &arguments,
                    "CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER",
                    &barrier,
                )
            };
            assert_eq!(retry.code, 0, "{}", retry.describe());
            let mut release_background = housekeeping.map(|listener| {
                let (mut turn, _) = listener.accept().expect("detached housekeeping arrived");
                use std::io::Read;
                let mut entered = [0];
                turn.read_exact(&mut entered).unwrap();
                assert_eq!(entered, [1], "worker holds its turn before cleanup");
                // Hold only this turn. Later turns use the pre-released legacy
                // checkpoint after the socket is unlinked; this connection stays held.
                std::fs::remove_file(barrier.join("pre-catch-up.sock")).unwrap();
                event_support::write_test_file(
                    barrier.join("pre-catch-up-release"),
                    b"later turns pass",
                )
                .unwrap();
                ReleaseBackgroundTurn(turn)
            });
            assert_eq!(
                worker_policy(&machine.data_root)["pending_background_request"],
                refused["pending_background_request"]
            );
            let status = machine.run(&project, &["status", "--context", "newest-one-refusal"]);
            assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
            assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
            assert_eq!(
                u64_at(&status.json(), "/history/search/outside_scope_events"),
                2
            );
            let (covered, outside) = if retired {
                // New captures cannot widen the retired request or change its original identity.
                let newer = write_session(
                    &machine.history_root(),
                    &project,
                    "later-newest-session",
                    &[(
                        "later request completes a different newest session",
                        "later answer",
                    )],
                );
                let text = std::fs::read_to_string(&newer).unwrap();
                event_support::write_test_file(newer, text.replace("2026-09-01", "2026-10-01"))
                    .unwrap();
                let later = machine.run(
                    &project,
                    &[
                        "import",
                        ".",
                        "--context",
                        "newest-one-refusal",
                        "--search",
                        "wait",
                        "--search-sessions",
                        "1",
                        "--embedding-workers",
                        "1",
                        "--write-memory-limit-bytes",
                        &HIGH.to_string(),
                    ],
                );
                assert_eq!(later.code, 0, "{}", later.describe());
                assert_eq!(
                    worker_policy(&machine.data_root)["pending_background_request"],
                    refused["pending_background_request"]
                );
                let source = machine
                    .history_root()
                    .join(event_support::claude_project_key(&project))
                    .join("selected-newest-session.jsonl");
                let contents = std::fs::read_to_string(&source).unwrap();
                let mut appended: serde_json::Value =
                    serde_json::from_str(contents.lines().next().unwrap()).unwrap();
                appended["uuid"] = uuid::Uuid::new_v4().to_string().into();
                appended["timestamp"] = "2026-09-02T09:00:00Z".into();
                appended["message"]["content"] =
                    "deferred beyond the refused capture boundary".into();
                event_support::write_test_file(&source, format!("{contents}{appended}\n")).unwrap();
                (4, 5)
            } else {
                (2, 2)
            };
            if let Some(release) = release_background.take() {
                drop(release);
                machine.wait_for_search_worker_exit();
            }
            let acknowledged = import_selected(
                &machine,
                &project,
                "newest-one-refusal",
                HIGH,
                "1",
                &request_observer(&machine, "newest-one-acknowledgement"),
            );
            assert_eq!(acknowledged.code, 0, "{}", acknowledged.describe());
            assert_eq!(
                str_at(&acknowledged.json(), "/search/worker"),
                "not_started"
            );
            assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
            assert_selection_counts(&machine, 0);
            assert_ready(&machine, &project, "newest-one-refusal");
            let status = machine.run(&project, &["status", "--context", "newest-one-refusal"]);
            assert_eq!(
                u64_at(&status.json(), "/history/search/covered_events"),
                covered
            );
            assert_eq!(
                u64_at(&status.json(), "/history/search/outside_scope_events"),
                outside
            );
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            let rows = reader.execute(
                "SELECT session_ref, committed_sequence FROM event_search_cursors ORDER BY session_ref",
                &std::collections::HashMap::new(),
            ).unwrap();
            // Generation preparation may create a zero-progress cursor for unselected history.
            // It must never commit or publish an embedding from that session.
            assert!(
                rows.rows
                    .iter()
                    .filter(|row| { row[0] == contextdb_core::Value::Text("session-B".into()) })
                    .all(|row| row[1] == contextdb_core::Value::Int64(0))
            );
            let older_passages = reader.execute(
                "SELECT COUNT(*) AS total FROM event_search_passages WHERE session_ref = 'session-B'",
                &std::collections::HashMap::new(),
            ).unwrap();
            assert_eq!(older_passages.rows[0][0], contextdb_core::Value::Int64(0));
            let selected = rows
                .rows
                .iter()
                .find(|row| row[0] == contextdb_core::Value::Text("selected-newest-session".into()))
                .unwrap();
            assert_eq!(
                selected[1],
                contextdb_core::Value::Int64(2),
                "the original frontier completes without embedding its later append"
            );
        }
    }

    #[test]
    fn retired_refusal_remains_acknowledgeable_across_context_forget() {
        let machine = Machine::with_nomic();
        let project = machine.project("forget-retired-refusal");
        write_session(
            &machine.history_root(),
            &project,
            "session-A",
            &[("initial user prompt to be forgotten", "assistant reply A")],
        );
        std::fs::create_dir_all(&machine.data_root).unwrap();
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let observer = request_observer(&machine, "forget-refusal-obs");
        let accepted = import_selected(
            &machine,
            &project,
            "forget-retired-refusal",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let original =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_selection_row_counts(&machine, 1, 1);
        assert_completion_row_counts(&machine, 0, 0);

        event_support::write_test_file(
            observer.join("refuse-next-selected-memory"),
            LOW.to_string(),
        )
        .unwrap();
        drop(held);
        let worker = machine.spawn_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        let output = worker.wait_with_output().unwrap();
        assert!(output.status.success(), "{output:?}");

        let refused = worker_policy(&machine.data_root);
        let old = &refused["pending_background_request"]["write_requests"][0];
        assert_eq!(old["token"], original["token"]);
        assert_eq!(old["selected_work"], original["selected_work"]);
        assert_eq!(old["refused_memory"], true);
        assert_eq!(old["selection_retired"], true);
        assert_selection_row_counts(&machine, 0, 0);
        assert_completion_row_counts(&machine, 1, 1);

        let forget = machine.run(
            &project,
            &["forget", "--context", "forget-retired-refusal", "--yes"],
        );
        assert_eq!(forget.code, 0, "{}", forget.describe());
        assert_eq!(str_at(&forget.json(), "/search/status"), "nothing_to_index");

        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("reader after forget");
        for table in [
            "events",
            "event_search_passages",
            "event_search_refusal_frontiers",
        ] {
            let rows = reader
                .execute(
                    &format!("SELECT COUNT(*) AS total FROM {table}"),
                    &std::collections::HashMap::new(),
                )
                .expect("count");
            assert_eq!(
                rows.rows[0][0],
                contextdb_core::Value::Int64(0),
                "payload leaked in {table}"
            );
        }
        assert_completion_row_counts(&machine, 1, 0);
        assert_eq!(
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]["token"],
            original["token"]
        );

        let restart = machine.run(&project, &["__event-search-worker", &envelope()]);
        assert_eq!(restart.code, 0, "{}", restart.describe());
        assert_eq!(
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]["token"],
            original["token"]
        );
        assert_completion_row_counts(&machine, 1, 0);

        let reimport = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "forget-retired-refusal",
                "--search",
                "wait",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
        );
        assert_eq!(reimport.code, 0, "{}", reimport.describe());
        assert_eq!(str_at(&reimport.json(), "/search/status"), "ready");

        let caught_up = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "forget-retired-refusal",
                "--search",
                "wait",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
        );
        assert_eq!(caught_up.code, 0, "{}", caught_up.describe());
        assert_eq!(str_at(&caught_up.json(), "/search/status"), "ready");

        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_selection_counts(&machine, 0);
        assert_ready(&machine, &project, "forget-retired-refusal");
    }

    #[test]
    fn new_narrow_background_work_preserves_an_incomplete_broader_refusal() {
        for retired in [false, true] {
            let machine = Machine::with_nomic();
            let project = machine.project("new-narrow-broader-refusal");
            capture_b(&machine, &project);
            let history = write_session(
                &machine.history_root(),
                &project,
                "newest-session",
                &[("newest session completes first", "newest answer")],
            );
            let newest = machine.run(
                &project,
                &[
                    "import",
                    ".",
                    "--context",
                    "new-narrow-broader-refusal",
                    "--search",
                    "wait",
                    "--search-order",
                    "newest",
                    "--search-sessions",
                    "1",
                    "--write-memory-limit-bytes",
                    &HIGH.to_string(),
                ],
            );
            assert_eq!(newest.code, 0, "{}", newest.describe());
            let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
                .expect("hold broad request admission");
            let observer = request_observer(&machine, "new-narrow-broad-refusal");
            let accepted = import_selected(
                &machine,
                &project,
                "new-narrow-broader-refusal",
                LOW,
                "all",
                &observer,
            );
            assert_eq!(accepted.code, 0, "{}", accepted.describe());
            assert_eq!(
                str_at(&accepted.json(), "/search/worker"),
                "already_running"
            );
            let original =
                worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                    [0]
                .clone();
            let context: context_graph::ContextId =
                serde_json::from_value(original["context_id"].clone()).unwrap();
            let spec = context_graph::EventSearchWorkSpec {
                profile: context_graph::EventSearchProfile::default(),
                embedding_space_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
                encoding_profile_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
            };
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            // Retain the exact real accepted frontier, including after physical retirement.
            let old_frontier = reader
                .event_search_selected_work(
                    context,
                    serde_json::from_value(original["selected_work"].clone()).unwrap(),
                )
                .unwrap();
            assert_eq!(old_frontier.streams.len(), 2);
            for session in ["session-B", "newest-session"] {
                let stream = old_frontier
                    .streams
                    .iter()
                    .find(|stream| stream.session_ref == session)
                    .expect("original broad session");
                assert_eq!(stream.frontier, 2);
            }
            assert!(
                !reader
                    .event_search_selected_work_complete(context, &spec, &old_frontier)
                    .unwrap()
            );
            drop(reader);
            event_support::write_test_file(
                observer.join("refuse-next-selected-memory"),
                LOW.to_string(),
            )
            .unwrap();
            if !retired {
                event_support::write_test_file(
                    observer.join("pause-before-selection-cleanup"),
                    b"retain the accepted frontier",
                )
                .unwrap();
            }
            drop(held);
            let mut worker = machine.spawn_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            if retired {
                let output = worker.wait_with_output().unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            } else {
                await_marker(&observer, "selection-cleanup-entered");
                worker
                    .kill()
                    .expect("interrupt this owned refusal worker before retirement");
                worker.wait().expect("reap this owned refusal worker");
            }
            let refused = worker_policy(&machine.data_root);
            let old = &refused["pending_background_request"]["write_requests"][0];
            assert_eq!(old["token"], original["token"]);
            assert_eq!(old["selected_work"], original["selected_work"]);
            assert_eq!(old["work_policy"]["sessions"], "all");
            assert_eq!(old["memory_limit_bytes"], LOW);
            assert_eq!(old["completed"], true);
            assert_eq!(old["refused_memory"], true);
            assert_eq!(old["refusal_superseded"], false);
            assert_eq!(old["selection_retired"], retired);
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            assert!(
                !reader
                    .event_search_selected_work_complete(context, &spec, &old_frontier)
                    .unwrap()
            );
            drop(reader);

            // Append a genuinely new closed turn, keeping the captured source prefix immutable.
            let prefix = std::fs::read(&history).unwrap();
            write_session(
                &machine.history_root(),
                &project,
                "newest-session",
                &[("new narrow pending turn", "new narrow answer")],
            );
            let suffix = std::fs::read(&history).unwrap();
            let mut appended = prefix.clone();
            appended.extend(suffix);
            event_support::write_test_file(&history, appended).unwrap();
            assert!(std::fs::read(&history).unwrap().starts_with(&prefix));
            let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
                .expect("hold fresh narrow request admission");
            let new_observer = request_observer(&machine, "new-narrow-accepted-work");
            let narrow = import_selected(
                &machine,
                &project,
                "new-narrow-broader-refusal",
                HIGH,
                "1",
                &new_observer,
            );
            assert_eq!(narrow.code, 0, "{}", narrow.describe());
            assert_eq!(str_at(&narrow.json(), "/search/worker"), "already_running");
            let queued = worker_policy(&machine.data_root);
            let requests = queued["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap();
            assert_eq!(requests.len(), 2, "{queued}");
            let old = requests
                .iter()
                .find(|request| request["token"] == original["token"])
                .unwrap();
            assert_eq!(old["refusal_superseded"], false, "{queued}");
            assert_eq!(old["selected_work"], original["selected_work"]);
            let new = requests
                .iter()
                .find(|request| request["token"] != original["token"])
                .unwrap();
            assert_ne!(new["selected_work"], original["selected_work"]);
            assert_eq!(new["work_policy"]["sessions"]["recent"], 1);
            assert_eq!(new["memory_limit_bytes"], HIGH);
            assert!(
                queued["pending_background_request"]["write_memory_refusals"][context.to_string()]
                    .is_object()
            );
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            let new_frontier = reader
                .event_search_selected_work(
                    context,
                    serde_json::from_value(new["selected_work"].clone()).unwrap(),
                )
                .unwrap();
            assert_eq!(new_frontier.streams.len(), 1);
            assert_eq!(new_frontier.streams[0].session_ref, "newest-session");
            assert_eq!(new_frontier.streams[0].frontier, 4);
            assert!(
                !reader
                    .event_search_selected_work_complete(context, &spec, &old_frontier)
                    .unwrap()
            );
            drop(reader);
            drop(held);
            let worker = machine.spawn_with_env(
                &project,
                &[
                    "__event-search-worker",
                    &envelope(),
                    "--write-memory-limit-bytes",
                    &HIGH.to_string(),
                ],
                REQUEST_OBSERVER_ENV,
                &new_observer,
            );
            let output = worker.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let actual = actual_budgets(&new_observer);
            assert!(!actual.is_empty(), "new narrow work must commit");
            assert!(
                actual
                    .iter()
                    .all(|batch| batch["session"] == "newest-session" && batch["limit"] == HIGH),
                "{actual:?}"
            );
            assert!(
                actual
                    .iter()
                    .all(|batch| batch["committed"].as_u64().unwrap() >= 2),
                "completed prefix replayed: {actual:?}"
            );
            assert!(
                actual.iter().any(|batch| batch["through"] == 4),
                "new frontier not reached: {actual:?}"
            );
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            assert!(
                reader
                    .event_search_selected_work_complete(context, &spec, &new_frontier)
                    .unwrap()
            );
            assert!(
                !reader
                    .event_search_selected_work_complete(context, &spec, &old_frontier)
                    .unwrap()
            );
            drop(reader);
            let status = machine.run(
                &project,
                &["status", "--context", "new-narrow-broader-refusal"],
            );
            assert_eq!(status.code, 0, "{}", status.describe());
            assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
            assert_eq!(
                str_at(&status.json(), "/history/search/last_failure"),
                context_graph::CgError::WriteDidNotFitTheMemoryLimit {
                    requested_bytes: LOW + 1,
                    limit_bytes: LOW,
                }
                .to_string()
            );
            assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 4);
            assert_eq!(
                u64_at(&status.json(), "/history/search/outside_scope_events"),
                2
            );
            let queued = worker_policy(&machine.data_root);
            let requests = queued["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap();
            assert_eq!(requests.len(), 1, "{queued}");
            assert_eq!(requests[0]["token"], original["token"]);
            assert_eq!(requests[0]["selected_work"], original["selected_work"]);
            assert_eq!(requests[0]["work_policy"]["sessions"], "all");
            assert_eq!(requests[0]["memory_limit_bytes"], LOW);
            assert_eq!(requests[0]["refused_memory"], true);
            assert_eq!(requests[0]["refusal_superseded"], false);
            assert_eq!(requests[0]["selection_retired"], true);
            assert!(
                queued["pending_background_request"]["write_memory_refusals"][context.to_string()]
                    .is_object()
            );
            assert_selection_row_counts(&machine, 0, 0);
            assert_completion_row_counts(&machine, 1, 2);
        }

        // Two engine admission refusals keep their actual facts on distinct saved tokens.
        // Completing the newest frontier reveals the older refusal, then completing the final
        // old stream clears it even though its one-byte physical cleanup remains impossible.
        let machine = Machine::with_nomic();
        let project = machine.project("real-refusal-identities");
        capture_b(&machine, &project);
        write_session(
            &machine.history_root(),
            &project,
            "newest-session",
            &[("newest pending", "newest answer")],
        );
        let observer = request_observer(&machine, "real-refusal-identities");
        let mut originals = Vec::new();
        let mut facts = Vec::new();
        for (sessions, limit) in [("all", 1), ("1", 2)] {
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let accepted = import_selected(
                &machine,
                &project,
                "real-refusal-identities",
                HIGH,
                sessions,
                &observer,
            );
            assert_eq!(accepted.code, 0, "{}", accepted.describe());
            let mut queued = worker_policy(&machine.data_root);
            let request = queued["pending_background_request"]["write_requests"]
                .as_array_mut()
                .unwrap()
                .last_mut()
                .unwrap();
            request["memory_limit_bytes"] = limit.into();
            let original = request.clone();
            write_event_search_worker_policy(&machine.data_root, &queued.to_string()).unwrap();
            drop(held);
            let worker = machine.spawn_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            let output = worker.wait_with_output().unwrap();
            assert!(
                !output.status.success(),
                "impossible saved cleanup must remain refused"
            );
            let queued = worker_policy(&machine.data_root);
            let request = queued["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .iter()
                .find(|request| request["token"] == original["token"])
                .unwrap();
            assert_eq!(request["refused_memory"], true, "{queued}");
            assert_eq!(request["selection_retired"], false);
            let recorded = &request["memory_refusal"]["Recorded"];
            assert_eq!(recorded["refusal"]["limit_bytes"], limit);
            assert!(recorded["refusal"]["requested_bytes"].as_u64().unwrap() > limit);
            assert_eq!(recorded["ordinal"], originals.len() + 1);
            let context: context_graph::ContextId =
                serde_json::from_value(request["context_id"].clone()).unwrap();
            assert_eq!(
                queued["pending_background_request"]["write_memory_refusals"][context.to_string()],
                recorded["refusal"]
            );
            originals.push(original);
            facts.push(recorded["refusal"].clone());
        }
        assert_ne!(facts[0], facts[1]);
        let context: context_graph::ContextId =
            serde_json::from_value(originals[0]["context_id"].clone()).unwrap();
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let accepted = import_selected(
            &machine,
            &project,
            "real-refusal-identities",
            HIGH,
            "1",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        drop(held);
        let worker = machine.spawn_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        let output = worker.wait_with_output().unwrap();
        assert!(
            !output.status.success(),
            "old cleanup stays unaffordable after successful service"
        );
        let partial = worker_policy(&machine.data_root);
        let requests = partial["pending_background_request"]["write_requests"]
            .as_array()
            .unwrap();
        for (index, original) in originals.iter().enumerate() {
            let request = requests
                .iter()
                .find(|request| request["token"] == original["token"])
                .unwrap();
            assert_eq!(request["refusal_superseded"], index == 1, "{partial}");
            assert_eq!(request["selection_retired"], false);
            assert_eq!(
                request["memory_refusal"]["Recorded"]["refusal"],
                facts[index]
            );
            assert_eq!(request["memory_limit_bytes"], index + 1);
        }
        assert_eq!(
            partial["pending_background_request"]["write_memory_refusals"][context.to_string()],
            facts[0]
        );
        let status = machine.run(
            &project,
            &["status", "--context", "real-refusal-identities"],
        );
        assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
        assert!(
            str_at(&status.json(), "/history/search/last_failure").contains("allowed 1 bytes"),
            "{}",
            status.describe()
        );
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let accepted = machine.run_with_env(
            &project,
            &[
                "import",
                ".",
                "--context",
                "real-refusal-identities",
                "--search",
                "background",
                "--search-sessions",
                "1",
                "--search-order",
                "oldest",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        drop(held);
        let worker = machine.spawn_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        let output = worker.wait_with_output().unwrap();
        assert!(
            !output.status.success(),
            "physical cleanup still owns the original saved ceilings"
        );
        let completed = worker_policy(&machine.data_root);
        let requests = completed["pending_background_request"]["write_requests"]
            .as_array()
            .unwrap();
        for original in &originals {
            let request = requests
                .iter()
                .find(|request| request["token"] == original["token"])
                .unwrap();
            assert_eq!(request["refusal_superseded"], true, "{completed}");
            assert_eq!(request["selection_retired"], false);
        }
        assert!(
            completed["pending_background_request"]["write_memory_refusals"]
                .as_object()
                .unwrap()
                .is_empty(),
            "{completed}"
        );
        let status = machine.run(
            &project,
            &["status", "--context", "real-refusal-identities"],
        );
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "ready",
            "{}",
            status.describe()
        );
        // All indexed events remain covered after narrowing (crates/context-graph/README.md:364–368).
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 4);
        assert_eq!(
            u64_at(&status.json(), "/history/search/outside_scope_events"),
            2
        );
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .unwrap();
        let spec = context_graph::EventSearchWorkSpec {
            profile: context_graph::EventSearchProfile::default(),
            embedding_space_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
            encoding_profile_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
        };
        for original in &originals {
            let frontier = reader
                .event_search_selected_work(
                    context,
                    serde_json::from_value(original["selected_work"].clone()).unwrap(),
                )
                .unwrap();
            assert!(
                reader
                    .event_search_selected_work_complete(context, &spec, &frontier)
                    .unwrap(),
                "the original refused frontier must now be complete"
            );
        }
        drop(reader);
        assert_selection_row_counts(&machine, 2, 3);
    }

    #[test]
    fn older_completed_import_acknowledges_only_observed_refusal_identities() {
        // A real all-session refusal remains owed when only the newest session is complete.
        // Cover both a retained accepted frontier and the original completion-only frontier.
        for retired in [false, true] {
            let machine = Machine::with_nomic();
            let project = machine.project("mixed-refusal-frontier");
            capture_b(&machine, &project);
            write_session(
                &machine.history_root(),
                &project,
                "newest-session",
                &[("newest session completes first", "newest answer")],
            );
            let newest = machine.run(
                &project,
                &[
                    "import",
                    ".",
                    "--context",
                    "mixed-refusal-frontier",
                    "--search",
                    "wait",
                    "--search-order",
                    "newest",
                    "--search-sessions",
                    "1",
                    "--write-memory-limit-bytes",
                    &HIGH.to_string(),
                ],
            );
            assert_eq!(newest.code, 0, "{}", newest.describe());
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer = request_observer(&machine, "mixed-real-refusal");
            let accepted = import_selected(
                &machine,
                &project,
                "mixed-refusal-frontier",
                LOW,
                "all",
                &observer,
            );
            assert_eq!(accepted.code, 0, "{}", accepted.describe());
            let original =
                worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                    [0]
                .clone();
            assert_selection_row_counts(&machine, 1, 2);
            event_support::write_test_file(
                observer.join("refuse-next-selected-memory"),
                LOW.to_string(),
            )
            .unwrap();
            if !retired {
                event_support::write_test_file(
                    observer.join("pause-before-selection-cleanup"),
                    b"retain the accepted frontier",
                )
                .unwrap();
            }
            drop(held);
            let mut worker = machine.spawn_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            if retired {
                let output = worker.wait_with_output().unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            } else {
                await_marker(&observer, "selection-cleanup-entered");
                worker
                    .kill()
                    .expect("interrupt before accepted selection retirement");
                worker.wait().expect("reap the interrupted refusal worker");
            }
            let refused = worker_policy(&machine.data_root);
            let old = &refused["pending_background_request"]["write_requests"][0];
            assert_eq!(old["token"], original["token"]);
            assert_eq!(old["selected_work"], original["selected_work"]);
            assert_eq!(old["work_policy"]["sessions"], "all");
            assert_eq!(old["memory_limit_bytes"], LOW);
            assert_eq!(old["refused_memory"], true);
            assert_eq!(old["refusal_superseded"], false);
            assert_eq!(old["selection_retired"], retired);
            assert_selection_row_counts(&machine, i64::from(!retired), if retired { 0 } else { 2 });

            let narrow = import_selected(
                &machine,
                &project,
                "mixed-refusal-frontier",
                HIGH,
                "1",
                &request_observer(&machine, "mixed-narrow-ack"),
            );
            assert_eq!(narrow.code, 0, "{}", narrow.describe());
            assert_eq!(str_at(&narrow.json(), "/search/worker"), "not_started");
            assert_eq!(
                worker_policy(&machine.data_root)["pending_background_request"],
                refused["pending_background_request"]
            );
            let status = machine.run(&project, &["status", "--context", "mixed-refusal-frontier"]);
            assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
            assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
            assert_eq!(
                u64_at(&status.json(), "/history/search/outside_scope_events"),
                2
            );
            assert_selection_row_counts(&machine, i64::from(!retired), if retired { 0 } else { 2 });

            let context: context_graph::ContextId =
                serde_json::from_value(original["context_id"].clone()).unwrap();
            let spec = context_graph::EventSearchWorkSpec {
                profile: context_graph::EventSearchProfile::default(),
                embedding_space_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
                encoding_profile_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into(),
            };
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            let all_policy = context_graph::EventSearchWorkPolicy {
                sessions: context_graph::EventSearchSessionScope::All,
                ..Default::default()
            };
            let all_frontier = reader
                .select_event_search_work(context, all_policy)
                .unwrap();
            assert_eq!(all_frontier.streams.len(), 2);
            assert!(
                !reader
                    .event_search_selected_work_complete(context, &spec, &all_frontier)
                    .unwrap()
            );
            let narrow_frontier = reader
                .select_event_search_work(
                    context,
                    context_graph::EventSearchWorkPolicy {
                        sessions: context_graph::EventSearchSessionScope::Recent(
                            std::num::NonZeroUsize::new(1).unwrap(),
                        ),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                reader
                    .event_search_selected_work_complete(context, &spec, &narrow_frontier)
                    .unwrap()
            );
            let saved_frontier = reader.event_search_selected_work(
                context,
                serde_json::from_value(original["selected_work"].clone()).unwrap(),
            );
            if retired {
                assert!(
                    saved_frontier.is_err(),
                    "retirement removes the runnable selection"
                );
            } else {
                let saved_frontier = saved_frontier.unwrap();
                assert_eq!(saved_frontier.streams.len(), 2);
                assert!(
                    !reader
                        .event_search_selected_work_complete(context, &spec, &saved_frontier)
                        .unwrap()
                );
            }
            drop(reader);

            // Another admitted library request completes All without changing the CLI queue.
            // This keeps the exact refused identity available for the acknowledgement proof.
            let mut model_config = context_graph::EmbedderConfig::default_nomic_offline();
            model_config.cache_dir = Some(machine.model_cache.clone());
            let embedder = context_graph::embedder::nomic::EventHistoryTextEmbedder::registration(
                std::sync::Arc::new(context_graph::embedder::nomic::NomicTextEmbedder::new(
                    model_config,
                )),
            )
            .unwrap();
            let completed =
                context_graph::EventSearchProjectionRunner::catch_up_observed_with_policy(
                    context_graph::EventSearchRunRequest {
                        store: context_graph::StoreConfig {
                            db_path: machine.db_path(),
                            default_text_embedder: None,
                            extra_embedding_spaces: vec![
                                context_graph::embedding_space::event_history_embedding_space(),
                            ],
                            write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
                            ..Default::default()
                        },
                        contexts: context_graph::ContextSelector::one(
                            serde_json::from_value(original["context_id"].clone()).unwrap(),
                        ),
                        embedder,
                        profile: context_graph::EventSearchProfile::default(),
                        embedding_workers: std::num::NonZeroUsize::new(1).unwrap(),
                        max_events_per_batch: std::num::NonZeroU32::new(256).unwrap(),
                        max_passages_per_batch: std::num::NonZeroU32::new(32).unwrap(),
                        max_bytes_per_batch: std::num::NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
                        lease_duration_ms: std::num::NonZeroU64::new(120_000).unwrap(),
                    },
                    context_graph::EventSearchWorkPolicy {
                        sessions: context_graph::EventSearchSessionScope::All,
                        ..Default::default()
                    },
                );
            assert!(completed.result.is_ok(), "{:?}", completed.result);
            assert_eq!(
                worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                    [0]["token"],
                original["token"]
            );
            let acknowledged = import_selected(
                &machine,
                &project,
                "mixed-refusal-frontier",
                HIGH,
                "1",
                &request_observer(&machine, "mixed-complete-ack"),
            );
            assert_eq!(acknowledged.code, 0, "{}", acknowledged.describe());
            assert_eq!(
                str_at(&acknowledged.json(), "/search/worker"),
                "not_started"
            );
            assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
            assert_selection_counts(&machine, 0);
            assert_ready(&machine, &project, "mixed-refusal-frontier");
        }
        // A's observed refusal cleanup cannot acknowledge B's completed worker wake before
        // B releases its long turn, even though B's durable selection is already retired.
        {
            let machine = Machine::with_nomic();
            let project_a = machine.project("acknowledgement-completed-a");
            let project_b = machine.project("acknowledgement-paused-b");
            write_session(
                &machine.history_root(),
                &project_a,
                "session-a",
                &[("complete A before B's worker turn", "answer A")],
            );
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer_a = request_observer(&machine, "completed-a-observer");
            let accepted_a = import(
                &machine,
                &project_a,
                "acknowledgement-completed-a",
                HIGH,
                &observer_a,
            );
            assert_eq!(accepted_a.code, 0, "{}", accepted_a.describe());
            let mut a =
                worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                    [0]
                .clone();
            drop(held);
            let completed_a = machine.run_with_env(
                &project_a,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer_a,
            );
            assert_eq!(completed_a.code, 0, "{}", completed_a.describe());
            assert_ready(&machine, &project_a, "acknowledgement-completed-a");
            assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
            assert_selection_counts(&machine, 0);

            write_session(
                &machine.history_root(),
                &project_b,
                "session-b",
                &[("B wake remains until its worker unlocks", "answer B")],
            );
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer_b = request_observer(&machine, "paused-b-observer");
            let accepted_b = import(
                &machine,
                &project_b,
                "acknowledgement-paused-b",
                HIGH,
                &observer_b,
            );
            assert_eq!(accepted_b.code, 0, "{}", accepted_b.describe());
            std::fs::remove_file(observer_b.join("release")).unwrap();
            event_support::write_test_file(observer_b.join("numbered-passes"), b"1").unwrap();
            drop(held);
            let mut worker_b = machine.spawn_with_env(
                &project_b,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer_b,
            );
            await_marker(&observer_b, "entered.1");
            let mut queued = worker_policy(&machine.data_root);
            let b_pending = queued["pending_background_request"].clone();
            let b = b_pending["write_requests"][0].clone();
            assert_eq!(b_pending["write_requests"], serde_json::json!([b]));
            assert_eq!(b["completed"], true);
            assert_eq!(b["selection_retired"], true);
            assert_eq!(b["refused_memory"], false);
            assert!(worker_b.try_wait().unwrap().is_none());
            assert!(
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).is_err()
            );
            assert_selection_counts(&machine, 0);

            // Persist A's real completed selection only after B pauses, so B cannot retire
            // it first. Seed only the same explicit memory-refusal fixture used below.
            let store = context_graph::Store::open(context_graph::StoreConfig {
                db_path: machine.db_path(),
                default_text_embedder: None,
                write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
                ..Default::default()
            })
            .unwrap();
            let a_context: context_graph::ContextId =
                serde_json::from_value(a["context_id"].clone()).unwrap();
            let selection = store
                .select_event_search_work(
                    a_context,
                    serde_json::from_value(a["work_policy"].clone()).unwrap(),
                )
                .unwrap();
            a["selected_work"] = serde_json::to_value(
                store
                    .persist_event_search_selected_work(a_context, &selection)
                    .unwrap(),
            )
            .unwrap();
            drop(store);
            a["completed"] = true.into();
            a["refused_memory"] = true.into();
            a["memory_refusal"] = serde_json::json!({"Recorded": {"ordinal": 1, "refusal": {"requested_bytes": 42, "limit_bytes": HIGH}}});
            a["selection_retired"] = false.into();
            queued["pending_background_request"]["write_requests"]
                .as_array_mut()
                .unwrap()
                .push(a.clone());
            queued["pending_background_request"]["write_memory_refusals"] = serde_json::json!({
                a_context.to_string(): {"requested_bytes": 42, "limit_bytes": HIGH}
            });
            write_event_search_worker_policy(&machine.data_root, &queued.to_string()).unwrap();
            assert_selection_counts(&machine, 1);
            let acknowledged_a = import(
                &machine,
                &project_a,
                "acknowledgement-completed-a",
                HIGH,
                &observer_a,
            );
            assert_eq!(acknowledged_a.code, 0, "{}", acknowledged_a.describe());
            let retained = worker_policy(&machine.data_root);
            assert_eq!(retained["pending_background_request"], b_pending);
            assert!(
                retained["pending_background_request"]["write_requests"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|request| request["token"] != a["token"])
            );
            assert!(worker_b.try_wait().unwrap().is_none());
            assert!(
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).is_err()
            );
            assert_selection_counts(&machine, 0);
            event_support::write_test_file(
                observer_b.join("release.1"),
                b"acknowledge B after unlock",
            )
            .unwrap();
            let output = worker_b.wait_with_output().unwrap();
            assert!(output.status.success(), "{output:?}");
            assert_ready(&machine, &project_a, "acknowledgement-completed-a");
            assert_ready(&machine, &project_b, "acknowledgement-paused-b");
            assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
            assert_selection_counts(&machine, 0);
        }
        // A caught-up acknowledgement retires only its observed selections. An unrelated
        // terminal refusal cannot spend its one-byte ceiling on A's acknowledgement.
        {
            let machine = Machine::with_nomic();
            let project_a = machine.project("acknowledgement-affordable");
            let project_b = machine.project("acknowledgement-unaffordable");
            for (project, session) in [(&project_a, "session-a"), (&project_b, "session-b")] {
                write_session(
                    &machine.history_root(),
                    project,
                    session,
                    &[("complete before acknowledgement", "answer")],
                );
            }
            std::fs::create_dir_all(&machine.data_root).unwrap();
            let held =
                acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
            let observer = request_observer(&machine, "scoped-acknowledgement");
            for (project, context) in [
                (&project_a, "acknowledgement-affordable"),
                (&project_b, "acknowledgement-unaffordable"),
            ] {
                let imported = import(&machine, project, context, HIGH, &observer);
                assert_eq!(imported.code, 0, "{}", imported.describe());
            }
            let mut saved = worker_policy(&machine.data_root);
            drop(held);
            let completed = machine.run_with_env(
                &project_a,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &observer,
            );
            assert_eq!(completed.code, 0, "{}", completed.describe());
            assert_ready(&machine, &project_a, "acknowledgement-affordable");
            assert_ready(&machine, &project_b, "acknowledgement-unaffordable");
            let pending = &mut saved["pending_background_request"];
            let store = context_graph::Store::open(context_graph::StoreConfig {
                db_path: machine.db_path(),
                default_text_embedder: None,
                write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
                ..Default::default()
            })
            .unwrap();
            for request in pending["write_requests"].as_array_mut().unwrap() {
                let context = serde_json::from_value(request["context_id"].clone()).unwrap();
                let selection = store
                    .select_event_search_work(
                        context,
                        context_graph::EventSearchWorkPolicy::default(),
                    )
                    .unwrap();
                request["selected_work"] = serde_json::to_value(
                    store
                        .persist_event_search_selected_work(context, &selection)
                        .unwrap(),
                )
                .unwrap();
            }
            drop(store);
            assert_selection_counts(&machine, 2);
            let mut refusals = serde_json::Map::new();
            for (index, request) in pending["write_requests"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                let limit = if index == 0 { HIGH } else { 1 };
                request["completed"] = true.into();
                request["refused_memory"] = true.into();
                request["memory_refusal"] = serde_json::json!({"Recorded": {"ordinal": index + 1, "refusal": {"requested_bytes": 42, "limit_bytes": limit}}});
                request["selection_retired"] = false.into();
                request["memory_limit_bytes"] = limit.into();
                let context: context_graph::ContextId =
                    serde_json::from_value(request["context_id"].clone()).unwrap();
                refusals.insert(
                    context.to_string(),
                    serde_json::json!({"requested_bytes": 42, "limit_bytes": limit}),
                );
            }
            pending["write_memory_refusals"] = refusals.into();
            let b = pending["write_requests"][1].clone();
            let b_context: context_graph::ContextId =
                serde_json::from_value(b["context_id"].clone()).unwrap();
            write_event_search_worker_policy(&machine.data_root, &saved.to_string()).unwrap();
            let acknowledged = import(
                &machine,
                &project_a,
                "acknowledgement-affordable",
                HIGH,
                &observer,
            );
            assert_eq!(acknowledged.code, 0, "{}", acknowledged.describe());
            let retained = worker_policy(&machine.data_root);
            assert_eq!(
                retained["pending_background_request"]["write_requests"],
                serde_json::json!([b])
            );
            assert_eq!(
                retained["pending_background_request"]["write_memory_refusals"],
                serde_json::json!({b_context.to_string(): {"requested_bytes": 42, "limit_bytes": 1}})
            );
            assert_selection_counts(&machine, 1);
            let reader = context_graph::ConsumerReader::open(
                machine.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .unwrap();
            let retained_selection = reader
                .event_search_selected_work(
                    b_context,
                    serde_json::from_value(b["selected_work"].clone()).unwrap(),
                )
                .unwrap();
            assert_eq!(retained_selection.streams.len(), 1);
            assert_eq!(retained_selection.streams[0].session_ref, "session-b");
            drop(reader);
            let status_a = machine.run(
                &project_a,
                &["status", "--context", "acknowledgement-affordable"],
            );
            assert_eq!(str_at(&status_a.json(), "/history/search/status"), "ready");
            let status_b = machine.run(
                &project_b,
                &["status", "--context", "acknowledgement-unaffordable"],
            );
            assert_eq!(str_at(&status_b.json(), "/history/search/status"), "failed");
        }
        let machine = Machine::with_nomic();
        let project = machine.project("acknowledgement-race");
        write_session(
            &machine.history_root(),
            &project,
            "completed-session",
            &[("completed before newer refusal", "answer")],
        );
        std::fs::create_dir_all(&machine.data_root).unwrap();
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let observer = request_observer(&machine, "initial-completion");
        let imported = import(&machine, &project, "acknowledgement-race", HIGH, &observer);
        assert_eq!(imported.code, 0, "{}", imported.describe());
        let mut saved = worker_policy(&machine.data_root);
        drop(held);
        let completed = machine.run_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(completed.code, 0, "{}", completed.describe());
        assert_ready(&machine, &project, "acknowledgement-race");
        // Seed a refused identity with the current build's real retired completion evidence.
        // The race still observes only this completed identity before a later refusal arrives.
        let store = context_graph::Store::open(context_graph::StoreConfig {
            db_path: machine.db_path(),
            default_text_embedder: None,
            write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
            ..Default::default()
        })
        .unwrap();
        let old_context = serde_json::from_value(
            saved["pending_background_request"]["write_requests"][0]["context_id"].clone(),
        )
        .unwrap();
        let old_frontier = store
            .select_event_search_work(old_context, Default::default())
            .unwrap();
        let old_selection = store
            .persist_event_search_selected_work(old_context, &old_frontier)
            .unwrap();
        store
            .preserve_event_search_refusal_work(old_context, old_selection)
            .unwrap();
        store
            .release_event_search_selected_work(old_context, &[old_selection])
            .unwrap();
        drop(store);
        saved["pending_background_request"]["write_requests"][0]["selected_work"] =
            serde_json::to_value(old_selection).unwrap();
        saved["pending_background_request"]["write_requests"][0]["selection_retired"] = true.into();
        write_event_search_worker_policy(&machine.data_root, &saved.to_string()).unwrap();
        let old = refuse_queue(&machine, LOW);
        let barrier = request_observer(&machine, "older-import");
        event_support::write_test_file(
            barrier.join("pause-before-import-acknowledgement"),
            b"pause after assessment",
        )
        .unwrap();
        let older = machine.spawn_with_env(
            &project,
            &[
                "import",
                ".",
                "--context",
                "acknowledgement-race",
                "--search",
                "background",
            ],
            REQUEST_OBSERVER_ENV,
            &barrier,
        );
        await_marker(&barrier, "import-assessment-complete");
        write_session(
            &machine.history_root(),
            &project,
            "newer-session",
            &[("new work accepted after older assessment", "newer answer")],
        );
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let newer_observer = request_observer(&machine, "newer-refusal-publication");
        let accepted = import(
            &machine,
            &project,
            "acknowledgement-race",
            LOW,
            &newer_observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        assert_eq!(
            str_at(&accepted.json(), "/search/worker"),
            "already_running"
        );
        let newer_token =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()["token"]
                .clone();
        assert_ne!(
            newer_token,
            old["pending_background_request"]["write_requests"][0]["token"]
        );
        event_support::write_test_file(
            newer_observer.join("refuse-next-selected-memory"),
            LOW.to_string(),
        )
        .unwrap();
        drop(held);
        let published = machine.run_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &newer_observer,
        );
        assert_eq!(published.code, 0, "{}", published.describe());
        let newer = worker_policy(&machine.data_root);
        assert_eq!(
            newer["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            newer["pending_background_request"]["write_requests"][0]["token"],
            newer_token
        );
        assert_eq!(
            newer["pending_background_request"]["write_requests"][0]["refused_memory"],
            true
        );
        event_support::write_test_file(
            barrier.join("release-import-acknowledgement"),
            b"acknowledge only observed identities",
        )
        .unwrap();
        let output = older.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let retained = worker_policy(&machine.data_root);
        assert_eq!(
            retained["pending_background_request"]["write_memory_refusals"],
            newer["pending_background_request"]["write_memory_refusals"]
        );
        assert_eq!(
            retained["pending_background_request"]["write_requests"][0]["token"],
            newer_token
        );
        assert_eq!(
            retained["pending_background_request"]["write_requests"][0]["selection_retired"],
            true
        );
        let status = machine.run(&project, &["status", "--context", "acknowledgement-race"]);
        assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
    }

    #[test]
    fn forgetting_after_all_import_requests_are_acknowledged_rebuilds_surviving_history() {
        let machine = Machine::with_nomic();
        let project = machine.project("forget-acknowledged");
        capture_b(&machine, &project);
        write_session(
            &machine.history_root(),
            &project,
            "session-A",
            &[("erase this session", "erase answer")],
        );
        let observer = request_observer(&machine, "forget-acknowledged-observer");
        let imported = import(&machine, &project, "forget-acknowledged", HIGH, &observer);
        assert_eq!(imported.code, 0, "{}", imported.describe());
        machine.wait_for_search_ready(&project, "forget-acknowledged");
        await_acknowledgement(&machine);
        machine.wait_for_search_worker_exit();
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        std::fs::remove_file(observer.join("actual-write-memory-batches"))
            .expect("reset batch evidence");
        let forgotten = machine.run_with_env(
            &project,
            &[
                "forget",
                "--context",
                "forget-acknowledged",
                "--session",
                "session-A",
                "--yes",
                "--write-memory-limit-bytes",
                &LOW.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
        assert_eq!(
            str_at(&forgotten.json(), "/search/status"),
            "rebuild_scheduled",
            "{}",
            forgotten.describe()
        );
        machine.wait_for_search_ready(&project, "forget-acknowledged");
        await_acknowledgement(&machine);
        machine.wait_for_search_worker_exit();
        let actual = actual_budgets(&observer);
        assert!(!actual.is_empty(), "surviving B must rebuild: {actual:?}");
        assert!(
            actual.iter().all(|batch| batch["session"] == "session-B"
                && batch["limit"].as_u64() == Some(LOW as u64)),
            "forget wake lost its resolved ceiling: {actual:?}"
        );
        let status = machine.run(&project, &["status", "--context", "forget-acknowledged"]);
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
    }

    #[test]
    fn a_standalone_forget_rebuild_coexists_with_another_contexts_queued_import() {
        let machine = Machine::with_nomic();
        let old = machine.project("forget-coexists-old");
        capture_b(&machine, &old);
        write_session(
            &machine.history_root(),
            &old,
            "session-A",
            &[("erase A", "erase answer")],
        );
        let observer = request_observer(&machine, "forget-coexists-observer");
        let ready = import(&machine, &old, "old-context", HIGH, &observer);
        assert_eq!(ready.code, 0, "{}", ready.describe());
        machine.wait_for_search_ready(&old, "old-context");
        // The long worker lock is released before queue acknowledgement. The
        // initial request must be gone before this case constructs its two new owners.
        await_acknowledgement(&machine);
        machine.wait_for_search_worker_exit();
        std::fs::remove_file(observer.join("actual-write-memory-batches"))
            .expect("reset admissions");
        let new = machine.project("forget-coexists-new");
        write_session(
            &machine.history_root(),
            &new,
            "session-C",
            &[("queued C", "C answer")],
        );
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let queued = import(&machine, &new, "new-context", HIGH, &observer);
        assert_eq!(queued.code, 0, "{}", queued.describe());
        write_session(
            &machine.history_root(),
            &new,
            "session-D",
            &[("deferred D", "D answer")],
        );
        let deferred = machine.run(
            &new,
            &[
                "import",
                ".",
                "--context",
                "new-context",
                "--search",
                "defer",
            ],
        );
        assert_eq!(deferred.code, 0, "{}", deferred.describe());
        let forgotten = machine.run_with_env(
            &old,
            &[
                "forget",
                "--context",
                "old-context",
                "--session",
                "session-A",
                "--yes",
                "--write-memory-limit-bytes",
                &LOW.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
        assert_eq!(
            str_at(&forgotten.json(), "/search/status"),
            "rebuild_scheduled"
        );
        assert_eq!(
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        drop(held);
        let worker = machine.run_with_env(
            &new,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        let actual = actual_budgets(&observer);
        for (session, ceiling) in [("session-B", LOW), ("session-C", HIGH)] {
            let batches = actual
                .iter()
                .filter(|batch| batch["session"] == session)
                .collect::<Vec<_>>();
            assert!(!batches.is_empty(), "lost {session}: {actual:?}");
            assert!(
                batches
                    .iter()
                    .all(|batch| batch["limit"].as_u64() == Some(ceiling as u64)),
                "wrong ceiling: {actual:?}"
            );
        }
        assert!(
            actual.iter().all(|batch| batch["session"] != "session-D"),
            "queued import acquired deferred suffix: {actual:?}"
        );
        assert_ready(&machine, &old, "old-context");
        let status = machine.run(&new, &["status", "--context", "new-context"]);
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn profile_replacement_retains_an_earlier_queued_frontier_published_after_acceptance() {
        let machine = Machine::with_nomic();
        let project = machine.project("queued-profile-retention");
        capture_b(&machine, &project);
        let d = write_session(
            &machine.history_root(),
            &project,
            "session-D",
            &[("D selected in the running request", "D answer")],
        );
        let d_prefix = std::fs::read_to_string(&d).expect("D source");
        event_support::write_test_file(&d, d_prefix.replace("2026-09-01", "2026-08-01"))
            .expect("D follows B in newest order");
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold initial request");
        let observer = request_observer(&machine, "queued-profile-observer");
        let first = import_selected(
            &machine,
            &project,
            "queued-profile-retention",
            LOW,
            "2",
            &observer,
        );
        assert_eq!(first.code, 0, "{}", first.describe());
        let barrier = machine.root.join("queued-profile-embedding");
        std::fs::create_dir_all(&barrier).expect("embedding rendezvous");
        drop(held);
        let worker = machine.spawn_with_envs(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            &[
                (REQUEST_OBSERVER_ENV, &observer),
                ("CG_NOMIC_TEST_PREPARED_EMBEDDING_BARRIER", &barrier),
            ],
        );
        await_marker(&barrier, "reached");
        // P1 selected B+D. B is embedding; D will only be claimed after C acceptance.
        let source = machine
            .history_root()
            .join(event_support::claude_project_key(&project))
            .join("session-B.jsonl");
        let prefix = std::fs::read_to_string(&source).expect("immutable B prefix");
        write_session(
            &machine.history_root(),
            &project,
            "session-B",
            &[
                ("B request selected before C exists", "B answer"),
                ("unselected later B tail", "unselected tail answer"),
            ],
        );
        let expanded = std::fs::read_to_string(&source).expect("B suffix");
        let suffix = expanded.lines().skip(2).collect::<Vec<_>>().join("\n");
        event_support::write_test_file(&source, format!("{prefix}{suffix}\n"))
            .expect("append preserves B prefix");
        let c = write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("newest C request", "C answer")],
        );
        let c_text = std::fs::read_to_string(&c).expect("C source");
        event_support::write_test_file(&c, c_text.replace("2026-09-01", "2026-09-02"))
            .expect("C is newest");
        write_config(
            &machine,
            "[search.passages]\ntarget_tokens = 8\nhard_tokens = 64\n",
        );
        let next = import_selected(
            &machine,
            &project,
            "queued-profile-retention",
            HIGH,
            "1",
            &observer,
        );
        assert_eq!(next.code, 0, "{}", next.describe());
        let pending =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .clone();
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("selection reader");
        let context = reader
            .graph()
            .list_contexts()
            .expect("Contexts")
            .into_iter()
            .find(|context| context.name == "queued-profile-retention")
            .unwrap();
        let selection = reader
            .event_search_selected_work(
                context.id,
                uuid::Uuid::parse_str(pending["selected_work"].as_str().unwrap()).unwrap(),
            )
            .expect("saved next selection");
        assert_eq!(
            selection.streams.len(),
            3,
            "C acceptance must retain the whole running B+D request"
        );
        assert_eq!(
            selection
                .streams
                .iter()
                .find(|stream| stream.session_ref == "session-B")
                .unwrap()
                .frontier,
            2,
            "B tail was never selected"
        );
        assert_eq!(
            selection
                .streams
                .iter()
                .find(|stream| stream.session_ref == "session-D")
                .expect("unclaimed D must be retained")
                .frontier,
            2,
        );
        drop(reader);
        event_support::write_test_file(barrier.join("release"), b"resume P1 publication")
            .expect("release embedding");
        let result = worker
            .wait_with_output()
            .expect("worker completes replacement");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual = actual_budgets(&observer);
        let b = actual
            .iter()
            .filter(|batch| batch["session"] == "session-B")
            .collect::<Vec<_>>();
        assert_eq!(b.len(), 2, "B must publish P1 and rebuild P2: {actual:?}");
        assert_eq!(b[0]["limit"], LOW);
        assert_eq!(b[1]["limit"], HIGH);
        assert!(
            b.iter().all(|batch| batch["through"] == 2),
            "deferred tail must stay out: {actual:?}"
        );
        assert!(
            actual
                .iter()
                .any(|batch| batch["session"] == "session-C" && batch["limit"] == HIGH)
        );
        let d_batches = actual
            .iter()
            .filter(|batch| batch["session"] == "session-D")
            .collect::<Vec<_>>();
        assert_eq!(
            d_batches.len(),
            2,
            "D must publish after acceptance and survive P2: {actual:?}"
        );
        assert_eq!(d_batches[0]["limit"], LOW);
        assert_eq!(d_batches[1]["limit"], HIGH);
        let status = machine.run(
            &project,
            &["status", "--context", "queued-profile-retention"],
        );
        assert_eq!(status.code, 0, "{}", status.describe());
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            6,
            "{}",
            status.describe()
        );
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn profile_replacement_retains_published_frontiers_outside_the_newest_selection() {
        let machine = Machine::with_nomic();
        let project = machine.project("retained-profile-selection");
        capture_b(&machine, &project);
        let a = write_session(
            &machine.history_root(),
            &project,
            "session-A",
            &[("retained earlier searchable history", "retained answer")],
        );
        let text = std::fs::read_to_string(&a).expect("A source");
        event_support::write_test_file(&a, text.replace("2026-09-01", "2026-07-01"))
            .expect("A precedes B");
        let observer = request_observer(&machine, "retained-profile-observer");
        let imported = import(
            &machine,
            &project,
            "retained-profile-selection",
            HIGH,
            &observer,
        );
        assert_eq!(imported.code, 0, "{}", imported.describe());
        machine.wait_for_search_ready(&project, "retained-profile-selection");
        // Ready coverage precedes the initial worker's queue acknowledgment.
        // Retire that request before publishing a tail and replacing its profile.
        await_acknowledgement(&machine);
        machine.wait_for_search_worker_exit();
        let original = std::fs::read_to_string(&a).expect("captured A prefix");
        write_session(
            &machine.history_root(),
            &project,
            "session-A",
            &[
                ("retained earlier searchable history", "retained answer"),
                ("deferred tail must stay unselected", "deferred answer"),
            ],
        );
        let expanded = std::fs::read_to_string(&a).expect("A appended source");
        let suffix = expanded.lines().skip(2).collect::<Vec<_>>().join("\n");
        event_support::write_test_file(
            &a,
            format!("{original}{suffix}\n").replace("2026-09-01", "2026-07-01"),
        )
        .expect("preserve captured A prefix");
        let captured = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "retained-profile-selection",
                "--search",
                "defer",
            ],
        );
        assert_eq!(captured.code, 0, "{}", captured.describe());
        write_config(
            &machine,
            "[search.passages]\ntarget_tokens = 8\nhard_tokens = 64\n",
        );
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold replacement claims");
        let accepted = import_selected(
            &machine,
            &project,
            "retained-profile-selection",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let pending =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("acceptance selection reader");
        let context = reader
            .graph()
            .list_contexts()
            .expect("Contexts")
            .into_iter()
            .find(|context| context.name == "retained-profile-selection")
            .unwrap();
        let selection = reader
            .event_search_selected_work(
                context.id,
                uuid::Uuid::parse_str(pending["selected_work"].as_str().unwrap()).unwrap(),
            )
            .expect("frozen replacement selection");
        assert_eq!(
            selection.streams.len(),
            2,
            "replacement selects retained A and newest B"
        );
        assert_eq!(
            selection
                .streams
                .iter()
                .find(|stream| stream.session_ref == "session-A")
                .unwrap()
                .frontier,
            2,
            "only previously published A is retained"
        );
        drop(reader);
        std::fs::remove_file(observer.join("actual-write-memory-batches"))
            .expect("reset admissions");
        drop(held);
        let profile = context_graph::EventSearchProfile {
            target_passage_tokens: std::num::NonZeroU32::new(8).unwrap(),
            hard_passage_tokens: std::num::NonZeroU32::new(64).unwrap(),
            ..Default::default()
        };
        let request = serde_json::json!({"profile": profile, "embedding_workers": 1, "ignored_publication": null, "retry_worker_lock": false}).to_string();
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &request,
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        let actual = actual_budgets(&observer);
        for session in ["session-A", "session-B"] {
            assert!(
                actual.iter().any(|batch| batch["session"] == session),
                "replacement omitted {session}: {actual:?}"
            );
        }
        assert!(
            actual
                .iter()
                .all(|batch| batch["limit"].as_u64() == Some(LOW as u64)),
            "retained work lost accepted ceiling: {actual:?}"
        );
        let status = machine.run(
            &project,
            &["status", "--context", "retained-profile-selection"],
        );
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            4,
            "{}",
            status.describe()
        );
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        let caught_up = import_selected(
            &machine,
            &project,
            "retained-profile-selection",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(caught_up.code, 0, "{}", caught_up.describe());
        machine.wait_for_search_worker_exit();
        assert_eq!(
            actual_budgets(&observer).len(),
            actual.len(),
            "same-profile retained frontiers are no-ops"
        );
    }

    #[test]
    fn forgetting_an_unrelated_session_preserves_the_queued_selection_and_ceiling() {
        let machine = Machine::with_nomic();
        let project = machine.project("forget-unrelated-selection");
        capture_b(&machine, &project);
        let a = write_session(
            &machine.history_root(),
            &project,
            "session-A",
            &[("older unrelated request", "older answer")],
        );
        let text = std::fs::read_to_string(&a).expect("A source");
        event_support::write_test_file(a, text.replace("2026-09-01", "2026-07-01"))
            .expect("A precedes B");
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let observer = request_observer(&machine, "forget-selection-observer");
        let accepted = import_selected(
            &machine,
            &project,
            "forget-unrelated-selection",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let pending =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        let forgotten = machine.run(
            &project,
            &[
                "forget",
                "--context",
                "forget-unrelated-selection",
                "--session",
                "session-A",
                "--yes",
            ],
        );
        assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("selection survives forget");
        let context = reader
            .graph()
            .list_contexts()
            .expect("Context")
            .into_iter()
            .find(|context| context.name == "forget-unrelated-selection")
            .expect("selected Context");
        let selection = reader
            .event_search_selected_work(
                context.id,
                uuid::Uuid::parse_str(pending["selected_work"].as_str().unwrap()).unwrap(),
            )
            .expect("queued B selection");
        assert_eq!(selection.streams.len(), 1);
        assert_eq!(selection.streams[0].session_ref, "session-B");
        assert!(
            !reader
                .event_search_selected_work_complete(
                    context.id,
                    &context_graph::EventSearchWorkSpec {
                        profile: context_graph::EventSearchProfile::default(),
                        embedding_space_id: context_graph::embedding_space::event_history_embedding_space()
                            .embedding_space_id,
                        encoding_profile_id: context_graph::embedding_space::BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.into()
                    },
                    &selection
                )
                .expect("B remains pending")
        );
        drop(reader);
        drop(held);
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        let batches = actual_budgets(&observer);
        assert!(!batches.is_empty(), "B must receive a projection claim");
        assert!(
            batches.iter().all(|batch| batch["session"] == "session-B"
                && batch["limit"].as_u64() == Some(LOW as u64)),
            "B lost its ceiling: {batches:?}"
        );
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_ready(&machine, &project, "forget-unrelated-selection");
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn unpublished_selections_are_reclaimed_without_erasing_queued_work_after_reopen() {
        let machine = Machine::with_nomic();
        let project = machine.project("orphan-selection");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let observer = request_observer(&machine, "orphan-selection-observer");
        let accepted = import_selected(&machine, &project, "orphan-selection", LOW, "1", &observer);
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let add_unpublished = || {
            let store = context_graph::Store::open(context_graph::StoreConfig {
                db_path: machine.db_path(),
                default_text_embedder: None,
                write_memory_limit_bytes: std::num::NonZeroUsize::new(HIGH).unwrap(),
                ..Default::default()
            })
            .expect("reopen writable capture");
            let context = store
                .list_contexts()
                .expect("Context")
                .into_iter()
                .find(|context| context.name == "orphan-selection")
                .expect("selected Context");
            let selection = store
                .select_event_search_work(
                    context.id,
                    context_graph::EventSearchWorkPolicy::default(),
                )
                .expect("captured frontier");
            for _ in 0..70 {
                store
                    .persist_event_search_selected_work(context.id, &selection)
                    .expect("persist before interrupted publication");
            }
            // Dropping the Store models restart after the database commit but before queue publication.
        };
        add_unpublished();
        assert_selection_counts(&machine, 71);
        let retried = import_selected(&machine, &project, "orphan-selection", HIGH, "1", &observer);
        assert_eq!(retried.code, 0, "{}", retried.describe());
        assert_selection_counts(&machine, 2);
        add_unpublished();
        assert_selection_counts(&machine, 72);
        drop(held);
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        assert_selection_counts(&machine, 0);
        let batches = actual_budgets(&observer);
        assert!(!batches.is_empty());
        assert!(
            batches
                .iter()
                .all(|batch| batch["limit"].as_u64() == Some(LOW as u64)),
            "accepted B lost its low ceiling: {batches:?}"
        );
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_ready(&machine, &project, "orphan-selection");
    }

    #[test]
    fn an_active_time_refusal_keeps_its_original_frontier_until_a_complete_retry() {
        let machine = Machine::with_nomic();
        let project = machine.project("read-time-broad-refusal");
        capture_b(&machine, &project);
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("newer request", "newer answer")],
        );
        std::fs::create_dir_all(&machine.data_root).unwrap();
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let observer = request_observer(&machine, "read-time-broad-observer");
        let accepted = import_selected(
            &machine,
            &project,
            "read-time-broad-refusal",
            LOW,
            "all",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let original =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        event_support::write_test_file(observer.join("refuse-selected-active-time-count"), b"10")
            .unwrap();
        drop(held);
        let refused = machine.run_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(refused.code, 0, "{}", refused.describe());
        let saved =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(saved["token"], original["token"]);
        assert_eq!(saved["selected_work"], original["selected_work"]);
        assert_eq!(saved["selection_retired"], true);
        std::fs::remove_file(observer.join("refuse-selected-active-time-count")).unwrap();
        for sessions in ["1", "all"] {
            let retry = if sessions == "1" {
                machine.run(
                    &project,
                    &[
                        "import",
                        ".",
                        "--context",
                        "read-time-broad-refusal",
                        "--search",
                        "wait",
                        "--search-sessions",
                        sessions,
                    ],
                )
            } else {
                let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
                    .unwrap();
                let accepted = import_selected(
                    &machine,
                    &project,
                    "read-time-broad-refusal",
                    LOW,
                    sessions,
                    &observer,
                );
                assert_eq!(accepted.code, 0, "{}", accepted.describe());
                drop(held);
                machine.run_with_env(
                    &project,
                    &["__event-search-worker", &envelope()],
                    REQUEST_OBSERVER_ENV,
                    &observer,
                )
            };
            assert_eq!(retry.code, 0, "{}", retry.describe());
            let status = machine.run(
                &project,
                &["status", "--context", "read-time-broad-refusal"],
            );
            if sessions == "1" {
                assert_eq!(
                    str_at(&status.json(), "/history/search/status"),
                    "failed",
                    "{}",
                    status.describe()
                );
                assert!(
                    str_at(&status.json(), "/history/search/last_failure")
                        .contains("--search wait")
                );
                let active = worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0].clone();
                assert_eq!(active["token"], original["token"]);
                assert_eq!(active["refusal_superseded"], false);
                assert_eq!(u64_at(&retry.json(), "/search/local_embedding_calls"), 1);
                assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
            } else {
                assert_ready(&machine, &project, "read-time-broad-refusal");
                assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
                let batches = actual_budgets(&observer);
                assert_eq!(
                    batches.len(),
                    1,
                    "previously completed newest session is never reprojected: {batches:?}"
                );
                assert_eq!(batches[0]["session"], "session-B");
                assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 4);
            }
        }
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn a_refused_aggregate_charges_only_its_assessed_low_active_time_selections() {
        let machine = Machine::with_nomic();
        let low_project = machine.project("read-time-low");
        let high_project = machine.project("read-time-high");
        let arriving_project = machine.project("read-time-arrival");
        for project in [&low_project, &high_project, &arriving_project] {
            capture_b(&machine, project);
        }
        std::fs::create_dir_all(&machine.data_root).unwrap();
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let observer = request_observer(&machine, "read-time-budget-observer");
        let low = import_selected(&machine, &low_project, "read-time-low", LOW, "1", &observer);
        assert_eq!(low.code, 0, "{}", low.describe());
        write_config(&machine, "[read]\nactive_time_ms = 12000\n");
        let high = import_selected(
            &machine,
            &high_project,
            "read-time-high",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(high.code, 0, "{}", high.describe());
        write_config(&machine, "[read]\nactive_time_ms = 5000\n");
        for (marker, content) in [
            ("refuse-aggregate-active-time-count", "10"),
            ("refuse-aggregate-active-time-limit-ms", "5000"),
            ("pause-before-aggregate-active-time-refusal", "pause"),
            (
                "pause-after-read-timeout",
                "pause after saving the assessed identities",
            ),
        ] {
            event_support::write_test_file(observer.join(marker), content).unwrap();
        }
        drop(held);
        let worker = machine.spawn_with_env(
            &low_project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        await_marker(&observer, "aggregate-active-time-refusal-entered");
        let arrival = import_selected(
            &machine,
            &arriving_project,
            "read-time-arrival",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(arrival.code, 0, "{}", arrival.describe());
        let queued =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .clone();
        assert_eq!(queued.len(), 3);
        let identities = queued
            .iter()
            .map(|request| {
                (
                    request["token"].clone(),
                    request["read_active_time_ms"].clone(),
                )
            })
            .collect::<Vec<_>>();
        event_support::write_test_file(
            observer.join("release-aggregate-active-time-refusal"),
            b"refuse observed statement",
        )
        .unwrap();
        await_marker(&observer, "read-timeout-recovery-entered");
        let saved =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .clone();
        for (ordinal, (token, budget)) in identities.iter().enumerate() {
            let request = saved
                .iter()
                .find(|request| request["token"] == *token)
                .unwrap();
            assert_eq!(
                request["read_timeout_no_progress"],
                if ordinal == 0 { 1 } else { 0 },
                "only the original low-budget selection was assessed; high or later requests keep their allowances"
            );
            assert_eq!(request["read_active_time_ms"], *budget);
        }
        std::fs::remove_file(observer.join("pause-before-aggregate-active-time-refusal")).unwrap();
        std::fs::remove_file(observer.join("pause-after-read-timeout")).unwrap();
        event_support::write_test_file(
            observer.join("release-read-timeout-recovery"),
            b"continue remaining accepted work",
        )
        .unwrap();
        let finished = worker.wait_with_output().unwrap();
        assert_eq!(
            finished.status.code(),
            Some(1),
            "{}",
            String::from_utf8_lossy(&finished.stderr)
        );
        assert_ready(&machine, &high_project, "read-time-high");
        let remaining =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
                .as_array()
                .unwrap()
                .clone();
        assert_eq!(remaining.len(), 2);
        assert!(
            remaining
                .iter()
                .all(|request| request["read_timeout_no_progress"] == 3
                    && request["read_active_time_ms"] == 5000)
        );
        assert_eq!(
            std::fs::read_to_string(observer.join("refuse-aggregate-active-time-count")).unwrap(),
            "6"
        );
        let low_status = machine.run(&low_project, &["status", "--context", "read-time-low"]);
        let arrival_status = machine.run(
            &arriving_project,
            &["status", "--context", "read-time-arrival"],
        );
        for status in [low_status, arrival_status] {
            assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
            assert!(
                str_at(&status.json(), "/history/search/last_failure").contains("--search wait")
            );
        }
        assert_eq!(
            actual_budgets(&observer).len(),
            1,
            "higher-budget accepted work is served without starvation"
        );
    }

    #[test]
    fn assessment_active_time_refusals_resume_or_leave_a_retryable_failed_status() {
        for marker in [
            "refuse-aggregate-active-time-count",
            "refuse-selected-assessment-active-time-count",
        ] {
            for failures in [1, 10] {
                let machine = Machine::with_nomic();
                let project = machine.project("read-time-assessment");
                capture_b(&machine, &project);
                std::fs::create_dir_all(&machine.data_root).unwrap();
                let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
                    .unwrap();
                let observer = request_observer(&machine, "read-time-assessment-observer");
                let accepted = import_selected(
                    &machine,
                    &project,
                    "read-time-assessment",
                    LOW,
                    "1",
                    &observer,
                );
                assert_eq!(accepted.code, 0, "{}", accepted.describe());
                event_support::write_test_file(observer.join(marker), failures.to_string())
                    .unwrap();
                drop(held);
                let worker = machine.run_with_env(
                    &project,
                    &["__event-search-worker", &envelope()],
                    REQUEST_OBSERVER_ENV,
                    &observer,
                );
                if failures == 1 {
                    assert_eq!(worker.code, 0, "{}", worker.describe());
                    assert_ready(&machine, &project, "read-time-assessment");
                    assert_eq!(actual_budgets(&observer).len(), 1);
                    assert!(
                        worker_policy(&machine.data_root)["pending_background_request"].is_null(),
                        "marker={marker}, failures={failures}, queue={}",
                        worker_policy(&machine.data_root)
                    );
                } else {
                    assert_eq!(worker.code, 1, "{}", worker.describe());
                    let saved = worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0].clone();
                    assert_eq!(saved["read_timeout_no_progress"], 3);
                    assert_eq!(
                        saved["read_timeout_failure_recorded"], false,
                        "failure precedes an owned generation attempt"
                    );
                    assert!(actual_budgets(&observer).is_empty());
                    assert_eq!(std::fs::read_to_string(observer.join(marker)).unwrap(), "7");
                    let status =
                        machine.run(&project, &["status", "--context", "read-time-assessment"]);
                    assert_eq!(
                        str_at(&status.json(), "/history/search/status"),
                        "failed",
                        "{}",
                        status.describe()
                    );
                    assert!(
                        str_at(&status.json(), "/history/search/last_failure")
                            .contains("5000 ms active-time allowance")
                    );
                    assert!(
                        str_at(&status.json(), "/history/search/last_failure")
                            .contains("--search wait")
                    );
                    let restarted = machine.run_with_env(
                        &project,
                        &["__event-search-worker", &envelope()],
                        REQUEST_OBSERVER_ENV,
                        &observer,
                    );
                    assert_eq!(restarted.code, 1, "{}", restarted.describe());
                    assert_eq!(
                        std::fs::read_to_string(observer.join(marker)).unwrap(),
                        "7",
                        "a restart grants no further attempts to the exhausted identity"
                    );
                    std::fs::remove_file(observer.join(marker)).unwrap();
                    let retry_held =
                        acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
                            .expect("hold the explicit retry until its synchronous worker starts");
                    let retry = import_selected(
                        &machine,
                        &project,
                        "read-time-assessment",
                        LOW,
                        "1",
                        &observer,
                    );
                    assert_eq!(retry.code, 0, "{}", retry.describe());
                    drop(retry_held);
                    let finished = machine.run_with_env(
                        &project,
                        &["__event-search-worker", &envelope()],
                        REQUEST_OBSERVER_ENV,
                        &observer,
                    );
                    assert_eq!(finished.code, 0, "marker={marker}: {}", finished.describe());
                    assert_ready(&machine, &project, "read-time-assessment");
                    assert!(
                        worker_policy(&machine.data_root)["pending_background_request"].is_null(),
                        "marker={marker}, failures={failures}, queue={}",
                        worker_policy(&machine.data_root)
                    );
                }
                assert_selection_counts(&machine, 0);
            }
        }
    }

    #[test]
    fn a_transient_typed_active_time_refusal_resumes_selected_work_without_reprojecting_the_completed_prefix()
     {
        let machine = Machine::with_nomic();
        let project = machine.project("read-time-prefix");
        capture_b(&machine, &project);
        let observer = request_observer(&machine, "read-time-prefix-observer");
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold the accepted request");
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("C request", "C answer")],
        );
        let accepted = import_selected(
            &machine,
            &project,
            "read-time-prefix",
            LOW,
            "all",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        event_support::write_test_file(
            observer.join("refuse-active-time-after-committed-batch"),
            b"one typed refusal",
        )
        .expect("refuse after this request commits its first batch");
        event_support::write_test_file(
            observer.join("pause-after-read-timeout"),
            b"restart after committed progress",
        )
        .unwrap();
        drop(held);
        let mut interrupted = machine.spawn_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        await_marker(&observer, "read-timeout-recovery-entered");
        let saved =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(
            saved["read_timeout_no_progress"], 0,
            "committed progress restores the no-progress allowance"
        );
        assert_eq!(saved["read_timeout_covered_events"], 2);
        interrupted
            .kill()
            .expect("interrupt during recovery after a committed batch");
        let first = interrupted
            .wait_with_output()
            .expect("reap the interrupted worker");
        assert!(String::from_utf8_lossy(&first.stderr).contains("active-time allowance"));
        std::fs::remove_file(observer.join("pause-after-read-timeout")).unwrap();
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_ready(&machine, &project, "read-time-prefix");
        let actual = actual_budgets(&observer);
        assert_eq!(
            actual.len(),
            2,
            "each selected session must commit once: {actual:?}"
        );
        assert_eq!(
            actual
                .iter()
                .filter(|batch| batch["session"] == "session-B")
                .count(),
            1,
            "completed batch in this accepted request must not be reprojected: {actual:?}"
        );
        assert_eq!(
            actual
                .iter()
                .filter(|batch| batch["session"] == "session-C")
                .count(),
            1
        );
        assert!(actual.iter().all(|batch| batch["limit"] == LOW));
        let status = machine.run(&project, &["status", "--context", "read-time-prefix"]);
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            4,
            "{}",
            status.describe()
        );
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn foreground_assessment_active_time_refusals_recover_and_exhaust_with_durable_status() {
        for after_runs in [0, 1] {
            for failures in [1, 10] {
                let machine = Machine::with_nomic();
                let project = machine.project("foreground-assessment-recovery");
                capture_b(&machine, &project);
                let observer = request_observer(&machine, "foreground-assessment-observer");
                event_support::write_test_file(
                    observer.join("refuse-foreground-assessment-active-time-count"),
                    failures.to_string(),
                )
                .unwrap();
                event_support::write_test_file(
                    observer.join("foreground-assessment-refusal-after-runs"),
                    after_runs.to_string(),
                )
                .unwrap();
                if after_runs == 1 {
                    event_support::write_test_file(
                        observer.join("refuse-foreground-active-time-count"),
                        b"1",
                    )
                    .unwrap();
                }
                let args = [
                    "import",
                    ".",
                    "--context",
                    "foreground-assessment-recovery",
                    "--search",
                    "wait",
                    "--search-sessions",
                    "all",
                    "--embedding-workers",
                    "1",
                ];
                let foreground =
                    machine.run_with_env(&project, &args, REQUEST_OBSERVER_ENV, &observer);
                if failures == 1 {
                    assert_eq!(
                        foreground.code,
                        0,
                        "after_runs={after_runs}: {}",
                        foreground.describe()
                    );
                    assert_ready(&machine, &project, "foreground-assessment-recovery");
                    assert_eq!(observed_request_caps(&observer).len(), after_runs + 1);
                } else {
                    assert_eq!(
                        foreground.code,
                        1,
                        "after_runs={after_runs}: {}",
                        foreground.describe()
                    );
                    assert!(str_at(&foreground.json(), "/search/reason").contains("--search wait"));
                    assert_eq!(observed_request_caps(&observer).len(), after_runs);
                    assert_eq!(
                        std::fs::read_to_string(
                            observer.join("refuse-foreground-assessment-active-time-count")
                        )
                        .unwrap(),
                        (7 + after_runs).to_string(),
                        "assessment and runner refusals share one bounded allowance",
                    );
                    let status = machine.run(
                        &project,
                        &["status", "--context", "foreground-assessment-recovery"],
                    );
                    assert_eq!(
                        str_at(&status.json(), "/history/search/status"),
                        "failed",
                        "{}",
                        status.describe()
                    );
                    assert!(
                        str_at(&status.json(), "/history/search/last_failure")
                            .contains("5000 ms active-time allowance")
                    );
                    assert!(
                        str_at(&status.json(), "/history/search/last_failure")
                            .contains("--search wait")
                    );
                    std::fs::remove_file(
                        observer.join("refuse-foreground-assessment-active-time-count"),
                    )
                    .unwrap();
                    let saved =
                        worker_policy(&machine.data_root)["foreground_read_refusals"].clone();
                    assert_eq!(saved.as_array().unwrap().len(), 1);
                    assert_eq!(
                        saved[0]["selected_work"]["streams"][0]["session_ref"],
                        "session-B"
                    );
                    assert_eq!(saved[0]["selected_work"]["streams"][0]["frontier"], 2);
                    write_session(
                        &machine.history_root(),
                        &project,
                        "session-C",
                        &[("newer captured request", "newer answer")],
                    );
                    let narrow = machine.run(
                        &project,
                        &[
                            "import",
                            ".",
                            "--context",
                            "foreground-assessment-recovery",
                            "--search",
                            "wait",
                            "--search-sessions",
                            "1",
                            "--embedding-workers",
                            "1",
                        ],
                    );
                    assert_eq!(narrow.code, 0, "{}", narrow.describe());
                    assert_eq!(
                        worker_policy(&machine.data_root)["foreground_read_refusals"],
                        saved,
                        "a narrower completion cannot acknowledge the original incomplete frontier"
                    );
                    let status = machine.run(
                        &project,
                        &["status", "--context", "foreground-assessment-recovery"],
                    );
                    assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
                    let retry =
                        machine.run_with_env(&project, &args, REQUEST_OBSERVER_ENV, &observer);
                    assert_eq!(retry.code, 0, "{}", retry.describe());
                    assert_ready(&machine, &project, "foreground-assessment-recovery");
                }
                let status = machine.run(
                    &project,
                    &["status", "--context", "foreground-assessment-recovery"],
                );
                let expected_events = if failures == 1 { 2 } else { 4 };
                assert_eq!(
                    u64_at(&status.json(), "/history/search/covered_events"),
                    expected_events
                );
                let reader = context_graph::ConsumerReader::open(
                    machine.db_path(),
                    context_graph::event_history_reader_declaration(),
                )
                .unwrap();
                let passages = reader.execute("SELECT COUNT(*) AS total FROM event_search_passages WHERE is_current = true AND is_searchable = true", &std::collections::HashMap::new()).unwrap();
                assert_eq!(
                    passages.rows[0][0],
                    contextdb_core::Value::Int64(expected_events as i64 / 2),
                    "each completed session publishes exactly one current passage"
                );
                drop(reader);
                assert!(
                    worker_policy(&machine.data_root)["foreground_read_refusals"]
                        .as_array()
                        .unwrap()
                        .is_empty()
                );
                assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
                assert_selection_counts(&machine, 0);
            }
        }
    }

    #[test]
    fn large_foreground_selections_and_repeated_failures_keep_bounded_durable_status() {
        let machine = Machine::with_nomic();
        let other_project = machine.project("other-foreground-refusal");
        write_session(
            &machine.history_root(),
            &other_project,
            "other-session",
            &[("other request", "other answer")],
        );
        let observer = request_observer(&machine, "large-foreground-refusal-observer");
        event_support::write_test_file(
            observer.join("refuse-foreground-selection-active-time-count"),
            b"10",
        )
        .unwrap();
        let other = machine.run_with_env(
            &other_project,
            &[
                "import",
                ".",
                "--context",
                "other-foreground-refusal",
                "--search",
                "wait",
                "--search-sessions",
                "all",
                "--embedding-workers",
                "1",
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(other.code, 1, "{}", other.describe());
        std::fs::remove_file(observer.join("refuse-foreground-selection-active-time-count"))
            .unwrap();
        let other_refusal =
            worker_policy(&machine.data_root)["foreground_read_refusals"][0].clone();
        let project = machine.project("large-foreground-refusal");
        for index in 0..32 {
            write_session(
                &machine.history_root(),
                &project,
                &format!("session-{index:03}-{}", "long-source-frontier".repeat(3)),
                &[("selected request", "selected answer")],
            );
        }
        let args = [
            "import",
            ".",
            "--context",
            "large-foreground-refusal",
            "--search",
            "wait",
            "--search-sessions",
            "all",
            "--embedding-workers",
            "1",
        ];
        let mut last_ordinal = other_refusal["ordinal"].as_u64().unwrap();
        for _ in 0..3 {
            event_support::write_test_file(
                observer.join("refuse-foreground-assessment-active-time-count"),
                b"10",
            )
            .unwrap();
            let failed = machine.run_with_env(&project, &args, REQUEST_OBSERVER_ENV, &observer);
            assert_eq!(failed.code, 1, "{}", failed.describe());
            assert!(
                !str_at(&failed.json(), "/search/reason").contains("status could not be saved")
            );
            assert_eq!(
                std::fs::read_to_string(
                    observer.join("refuse-foreground-assessment-active-time-count")
                )
                .unwrap(),
                "7"
            );
            let policy = worker_policy(&machine.data_root);
            assert!(
                policy.to_string().len() as u64 <= cg_node_fs::EVENT_SEARCH_WORKER_POLICY_MAX_BYTES
            );
            let saved = policy["foreground_read_refusals"].as_array().unwrap();
            assert_eq!(
                saved.len(),
                2,
                "repeated failures collapse only the overflowing Context"
            );
            assert!(
                saved.contains(&other_refusal),
                "another Context's refusal stays intact"
            );
            let large = saved
                .iter()
                .find(|refusal| refusal["token"] != other_refusal["token"])
                .unwrap();
            assert!(
                large["selected_work"].is_null(),
                "the large frontier uses bounded full-Context completion evidence"
            );
            assert_eq!(large["limit_ms"], 5000);
            let ordinal = large["ordinal"].as_u64().unwrap();
            assert!(ordinal > last_ordinal);
            last_ordinal = ordinal;
            assert!(policy["pending_background_request"].is_null());
            let status = machine.run(
                &project,
                &["status", "--context", "large-foreground-refusal"],
            );
            assert_eq!(
                str_at(&status.json(), "/history/search/status"),
                "failed",
                "{}",
                status.describe()
            );
            assert!(
                str_at(&status.json(), "/history/search/last_failure").contains("--search wait")
            );
        }
        std::fs::remove_file(observer.join("refuse-foreground-assessment-active-time-count"))
            .unwrap();
        let retry = machine.run(&project, &args);
        assert_eq!(retry.code, 0, "{}", retry.describe());
        assert_ready(&machine, &project, "large-foreground-refusal");
        assert_eq!(
            worker_policy(&machine.data_root)["foreground_read_refusals"],
            serde_json::json!([other_refusal])
        );
        let status = machine.run(
            &project,
            &["status", "--context", "large-foreground-refusal"],
        );
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 64);
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn foreground_selection_timeout_exhaustion_stays_failed_until_full_context_completion() {
        let machine = Machine::with_nomic();
        let project = machine.project("foreground-selection-recovery");
        capture_b(&machine, &project);
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("newer selected request", "newer answer")],
        );
        let observer = request_observer(&machine, "foreground-selection-observer");
        event_support::write_test_file(
            observer.join("refuse-foreground-selection-active-time-count"),
            b"10",
        )
        .unwrap();
        let args = [
            "import",
            ".",
            "--context",
            "foreground-selection-recovery",
            "--search",
            "wait",
            "--search-sessions",
            "all",
            "--embedding-workers",
            "1",
        ];
        let failed = machine.run_with_env(&project, &args, REQUEST_OBSERVER_ENV, &observer);
        assert_eq!(failed.code, 1, "{}", failed.describe());
        assert!(str_at(&failed.json(), "/search/reason").contains("--search wait"));
        assert!(
            observed_request_caps(&observer).is_empty(),
            "selection refused before a projection attempt"
        );
        assert_eq!(
            std::fs::read_to_string(observer.join("refuse-foreground-selection-active-time-count"))
                .unwrap(),
            "7"
        );
        let saved = worker_policy(&machine.data_root)["foreground_read_refusals"].clone();
        assert_eq!(saved.as_array().unwrap().len(), 1);
        assert!(saved[0]["selected_work"].is_null());
        let status = machine.run(
            &project,
            &["status", "--context", "foreground-selection-recovery"],
        );
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "failed",
            "{}",
            status.describe()
        );
        assert!(str_at(&status.json(), "/history/search/last_failure").contains("--search wait"));
        std::fs::remove_file(observer.join("refuse-foreground-selection-active-time-count"))
            .unwrap();
        let narrow = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "foreground-selection-recovery",
                "--search",
                "wait",
                "--search-sessions",
                "1",
                "--embedding-workers",
                "1",
            ],
        );
        assert_eq!(narrow.code, 0, "{}", narrow.describe());
        assert_eq!(
            worker_policy(&machine.data_root)["foreground_read_refusals"],
            saved
        );
        let status = machine.run(
            &project,
            &["status", "--context", "foreground-selection-recovery"],
        );
        assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
        let retry = machine.run(&project, &args);
        assert_eq!(retry.code, 0, "{}", retry.describe());
        assert_ready(&machine, &project, "foreground-selection-recovery");
        assert!(
            worker_policy(&machine.data_root)["foreground_read_refusals"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn foreground_active_time_refusals_stop_when_capture_changes_without_coverage_progress() {
        let machine = Machine::with_nomic();
        let project = machine.project("foreground-read-time-bound");
        capture_b(&machine, &project);
        let observer = request_observer(&machine, "foreground-read-time-bound-observer");
        event_support::write_test_file(observer.join("refuse-foreground-active-time-count"), b"10")
            .unwrap();
        event_support::write_test_file(
            observer.join("pause-after-foreground-read-timeout"),
            b"pause",
        )
        .unwrap();
        let mut foreground = machine.spawn_with_env(
            &project,
            &[
                "import",
                ".",
                "--context",
                "foreground-read-time-bound",
                "--search",
                "wait",
                "--search-sessions",
                "all",
                "--embedding-workers",
                "1",
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        for remaining in [10, 9, 8] {
            await_marker(&observer, &format!("foreground-read-timeout-{remaining}"));
            write_session(
                &machine.history_root(),
                &project,
                &format!("arrival-{remaining}"),
                &[("captured during recovery", "no projection progress")],
            );
            let capture = machine.run(
                &project,
                &[
                    "import",
                    ".",
                    "--context",
                    "foreground-read-time-bound",
                    "--search",
                    "defer",
                    "--search-sessions",
                    "all",
                ],
            );
            assert_eq!(capture.code, 0, "{}", capture.describe());
            assert_eq!(u64_at(&capture.json(), "/search/covered_events"), 0);
            event_support::write_test_file(
                observer.join(format!("release-foreground-read-timeout-{remaining}")),
                b"capture changed without coverage progress",
            )
            .unwrap();
        }
        foreground
            .wait()
            .expect("bounded foreground exit after three refusals");
        let output = foreground
            .wait_with_output()
            .expect("foreground terminates at its bound");
        assert_eq!(
            output.status.code(),
            Some(1),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(str_at(&receipt, "/search/status"), "failed");
        assert_eq!(u64_at(&receipt, "/search/covered_events"), 0);
        assert!(
            output
                .stdout
                .windows(b"cg status".len())
                .any(|text| text == b"cg status")
        );
        assert!(
            output
                .stdout
                .windows(b"--search wait".len())
                .any(|text| text == b"--search wait")
        );
        assert_eq!(observed_request_caps(&observer), vec![1, 1, 1]);
        assert_eq!(
            std::fs::read_to_string(observer.join("refuse-foreground-active-time-count")).unwrap(),
            "7"
        );
        let status = machine.run(
            &project,
            &["status", "--context", "foreground-read-time-bound"],
        );
        assert_eq!(u64_at(&status.json(), "/history/captured_events"), 8);
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 0);
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "failed",
            "{}",
            status.describe()
        );
        assert!(str_at(&status.json(), "/history/search/last_failure").contains("--search wait"));
        std::fs::remove_file(observer.join("refuse-foreground-active-time-count")).unwrap();
        let retry = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "foreground-read-time-bound",
                "--search",
                "wait",
                "--search-sessions",
                "all",
                "--embedding-workers",
                "1",
            ],
        );
        assert_eq!(retry.code, 0, "{}", retry.describe());
        assert_ready(&machine, &project, "foreground-read-time-bound");
    }

    #[test]
    fn prior_committed_coverage_does_not_extend_a_new_requests_active_time_allowance() {
        let machine = Machine::with_nomic();
        let project = machine.project("read-time-existing-coverage");
        capture_b(&machine, &project);
        let first = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "read-time-existing-coverage",
                "--search",
                "wait",
                "--search-sessions",
                "all",
            ],
        );
        assert_eq!(first.code, 0, "{}", first.describe());
        assert_eq!(u64_at(&first.json(), "/search/covered_events"), 2);
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("new request", "new answer")],
        );
        let held =
            acquire_worker_lock(&machine.data_root.join("event-search-worker.lock")).unwrap();
        let observer = request_observer(&machine, "read-time-existing-coverage-observer");
        let accepted = import_selected(
            &machine,
            &project,
            "read-time-existing-coverage",
            LOW,
            "all",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let saved =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(saved["read_timeout_covered_events"], 2);
        event_support::write_test_file(observer.join("refuse-selected-active-time-count"), b"10")
            .unwrap();
        drop(held);
        let refused = machine.run_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(refused.code, 0, "{}", refused.describe());
        assert_eq!(
            refused.stderr.matches("active-time allowance").count(),
            3,
            "{}",
            refused.describe()
        );
        assert_eq!(
            std::fs::read_to_string(observer.join("refuse-selected-active-time-count")).unwrap(),
            "7"
        );
        let saved =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(saved["read_timeout_no_progress"], 3);
        assert_eq!(saved["read_timeout_covered_events"], 2);
        let status = machine.run(
            &project,
            &["status", "--context", "read-time-existing-coverage"],
        );
        assert_eq!(str_at(&status.json(), "/history/search/status"), "failed");
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 2);
        assert!(str_at(&status.json(), "/history/search/last_failure").contains("--search wait"));
        assert!(
            actual_budgets(&observer).is_empty(),
            "the completed prefix must not be reprojected"
        );
    }

    #[test]
    fn consecutive_typed_active_time_refusals_exhaust_one_selected_request_allowance() {
        let machine = Machine::with_nomic();
        let project = machine.project("read-time-bound");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold request");
        let observer = request_observer(&machine, "read-time-bound-observer");
        let accepted = import_selected(&machine, &project, "read-time-bound", LOW, "1", &observer);
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        event_support::write_test_file(observer.join("refuse-selected-active-time-count"), b"10")
            .expect("more refusals than allowance");
        drop(held);
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        assert_eq!(
            worker.stderr.matches("active-time allowance").count(),
            3,
            "{}",
            worker.describe()
        );
        assert_eq!(
            std::fs::read_to_string(observer.join("refuse-selected-active-time-count"))
                .expect("remaining refusals"),
            "7"
        );
        let refused =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(refused["completed"], true);
        assert_eq!(refused["selection_retired"], true);
        assert_eq!(refused["read_timeout_no_progress"], 3);
        assert_eq!(refused["refusal_superseded"], false);
        assert!(
            actual_budgets(&observer).is_empty(),
            "no failed claim may publish passages"
        );
        let status = machine.run(&project, &["status", "--context", "read-time-bound"]);
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "failed",
            "{}",
            status.describe()
        );
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            0,
            "{}",
            status.describe()
        );
        assert!(
            str_at(&status.json(), "/history/search/last_failure").contains("--search wait"),
            "the failed status supplies a useful retry instruction: {}",
            status.describe()
        );
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn active_time_recovery_restart_keeps_the_selected_allowance_and_no_progress_count() {
        let machine = Machine::with_nomic();
        let project = machine.project("read-time-restart");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold accepted work");
        let observer = request_observer(&machine, "read-time-restart-observer");
        let accepted =
            import_selected(&machine, &project, "read-time-restart", LOW, "1", &observer);
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let accepted_request =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(accepted_request["read_active_time_ms"], 5000);
        event_support::write_test_file(observer.join("refuse-selected-active-time-count"), b"10")
            .unwrap();
        event_support::write_test_file(
            observer.join("pause-after-read-timeout"),
            b"pause after durable recovery count",
        )
        .unwrap();
        drop(held);
        let mut first = machine.spawn_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        await_marker(&observer, "read-timeout-recovery-entered");
        let interrupted =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(
            interrupted["selected_work"],
            accepted_request["selected_work"]
        );
        assert_eq!(interrupted["read_timeout_no_progress"], 1);
        assert_eq!(interrupted["read_active_time_ms"], 5000);
        first.kill().expect("interrupt during bounded recovery");
        first.wait().expect("reap interrupted worker");
        assert!(actual_budgets(&observer).is_empty());
        std::fs::remove_file(observer.join("pause-after-read-timeout")).unwrap();
        write_config(&machine, "[read]\nactive_time_ms = 12000\n");
        let resumed = machine.run_with_env(
            &project,
            &["__event-search-worker", &envelope()],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(resumed.code, 0, "{}", resumed.describe());
        assert_eq!(
            resumed
                .stderr
                .matches("5000 ms active-time allowance")
                .count(),
            2,
            "restart must retain the accepted allowance and only its two remaining attempts: {}",
            resumed.describe()
        );
        assert_eq!(
            std::fs::read_to_string(observer.join("refuse-selected-active-time-count")).unwrap(),
            "7"
        );
        let exhausted =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        assert_eq!(exhausted["completed"], true);
        assert_eq!(exhausted["selection_retired"], true);
        assert_eq!(
            std::fs::read_to_string(observer.join("cleanup-read-active-time-ms")).unwrap(),
            "5000",
            "cleanup retains the accepted allowance after configuration changes"
        );
        let status = machine.run(&project, &["status", "--context", "read-time-restart"]);
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "failed",
            "{}",
            status.describe()
        );
        assert!(str_at(&status.json(), "/history/search/last_failure").contains("--search wait"));
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 0);
        std::fs::remove_file(observer.join("refuse-selected-active-time-count")).unwrap();
        let retried = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "read-time-restart",
                "--search",
                "wait",
                "--search-sessions",
                "1",
            ],
        );
        assert_eq!(retried.code, 0, "{}", retried.describe());
        assert_eq!(
            str_at(&retried.json(), "/search/status"),
            "ready",
            "{}",
            retried.describe()
        );
        assert_ready(&machine, &project, "read-time-restart");
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn a_transient_typed_coordination_refusal_resumes_selected_work_without_reprojecting_the_completed_prefix()
     {
        let machine = Machine::with_nomic();
        let project = machine.project("coordination-prefix");
        capture_b(&machine, &project);
        let observer = request_observer(&machine, "coordination-prefix-observer");
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold the accepted request");
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("C request", "C answer")],
        );
        let accepted = import_selected(
            &machine,
            &project,
            "coordination-prefix",
            LOW,
            "all",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        event_support::write_test_file(
            observer.join("refuse-coordination-after-committed-batch"),
            b"one typed refusal",
        )
        .expect("refuse after this request commits its first batch");
        drop(held);
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        assert!(
            worker
                .stderr
                .contains("deterministic selected-work coordination refusal"),
            "{}",
            worker.describe()
        );
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert_ready(&machine, &project, "coordination-prefix");
        let actual = actual_budgets(&observer);
        assert_eq!(
            actual.len(),
            2,
            "each selected session must commit once: {actual:?}"
        );
        assert_eq!(
            actual
                .iter()
                .filter(|batch| batch["session"] == "session-B")
                .count(),
            1,
            "completed batch in this accepted request must not be reprojected: {actual:?}"
        );
        assert_eq!(
            actual
                .iter()
                .filter(|batch| batch["session"] == "session-C")
                .count(),
            1
        );
        assert!(actual.iter().all(|batch| batch["limit"] == LOW));
        let status = machine.run(&project, &["status", "--context", "coordination-prefix"]);
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            4,
            "{}",
            status.describe()
        );
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn consecutive_typed_coordination_refusals_exhaust_one_selected_request_allowance() {
        let machine = Machine::with_nomic();
        let project = machine.project("coordination-bound");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold request");
        let observer = request_observer(&machine, "coordination-bound-observer");
        let accepted = import_selected(
            &machine,
            &project,
            "coordination-bound",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        event_support::write_test_file(observer.join("refuse-selected-coordination-count"), b"10")
            .expect("more refusals than allowance");
        drop(held);
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        assert_eq!(
            worker
                .stderr
                .matches("deterministic selected-work coordination refusal")
                .count(),
            3,
            "{}",
            worker.describe()
        );
        assert_eq!(
            std::fs::read_to_string(observer.join("refuse-selected-coordination-count"))
                .expect("remaining refusals"),
            "7"
        );
        assert!(
            worker_policy(&machine.data_root)["pending_background_request"].is_null(),
            "recorded refusal acknowledges exactly its selection"
        );
        assert!(
            actual_budgets(&observer).is_empty(),
            "no failed claim may publish passages"
        );
        let status = machine.run(&project, &["status", "--context", "coordination-bound"]);
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "failed",
            "{}",
            status.describe()
        );
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            0,
            "{}",
            status.describe()
        );
        assert_selection_counts(&machine, 0);
    }

    #[test]
    fn a_recorded_refusal_acknowledges_only_its_selection_and_serves_later_disjoint_work() {
        let machine = Machine::with_nomic();
        let project = machine.project("refused-selection");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let observer = request_observer(&machine, "refused-selection-observer");
        let first = import_selected(&machine, &project, "refused-selection", LOW, "1", &observer);
        assert_eq!(first.code, 0, "{}", first.describe());
        let old = worker_policy(&machine.data_root)["pending_background_request"]["write_requests"]
            [0]
        .clone();
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("C request", "C answer")],
        );
        let later = import_selected(
            &machine,
            &project,
            "refused-selection",
            HIGH,
            "1",
            &observer,
        );
        assert_eq!(later.code, 0, "{}", later.describe());
        std::fs::remove_file(observer.join("release")).expect("pause after refusal");
        event_support::write_test_file(observer.join("numbered-passes"), b"1")
            .expect("observe first pass");
        event_support::write_test_file(
            observer.join("refuse-next-selected-work"),
            b"typed refusal",
        )
        .expect("refuse B once");
        drop(held);
        let worker = machine.spawn_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        await_marker(&observer, "entered.1");
        let queued = worker_policy(&machine.data_root);
        let requests = queued["pending_background_request"]["write_requests"]
            .as_array()
            .expect("requests");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["token"], old["token"]);
        assert_eq!(requests[0]["completed"], true, "{queued}");
        assert_eq!(requests[1]["completed"], false, "{queued}");
        let status = machine.run(&project, &["status", "--context", "refused-selection"]);
        assert_eq!(
            str_at(&status.json(), "/history/search/status"),
            "failed",
            "{}",
            status.describe()
        );
        assert!(
            str_at(&status.json(), "/history/search/last_failure")
                .contains("explicit test refusal")
        );
        event_support::write_test_file(observer.join("release.1"), b"serve C")
            .expect("release worker");
        let output = worker.wait_with_output().expect("worker completes");
        assert!(output.status.success(), "{:?}", output);
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        let actual = actual_budgets(&observer);
        assert!(!actual.is_empty(), "C must progress after B refusal");
        assert!(
            actual
                .iter()
                .all(|batch| batch["session"] == "session-C" && batch["limit"] == HIGH),
            "{actual:?}"
        );
        let status = machine.run(&project, &["status", "--context", "refused-selection"]);
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            2,
            "{}",
            status.describe()
        );
    }

    #[test]
    fn a_rejected_queue_publication_retires_its_unaccepted_selection() {
        let machine = Machine::with_nomic();
        let project = machine.project("rejected-selection");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let observer = request_observer(&machine, "rejected-selection-observer");
        let accepted = import_selected(
            &machine,
            &project,
            "rejected-selection",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let mut policy = worker_policy(&machine.data_root);
        policy["pending_background_request"]["write_requests"][0]["token"] =
            serde_json::Value::String("x".repeat(3000));
        let padded_token = 3000 + 4000 - policy.to_string().len();
        policy["pending_background_request"]["write_requests"][0]["token"] =
            serde_json::Value::String("x".repeat(padded_token));
        let existing = policy.to_string();
        assert_eq!(existing.len(), 4000);
        assert!(existing.len() < cg_node_fs::EVENT_SEARCH_WORKER_POLICY_MAX_BYTES as usize);
        write_event_search_worker_policy(&machine.data_root, &existing)
            .expect("valid queue near its byte ceiling");
        let output = import_selected(
            &machine,
            &project,
            "rejected-selection",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(output.code, 1, "{}", output.describe());
        assert_eq!(
            str_at(&output.json(), "/search/worker"),
            "spawn_failed",
            "{}",
            output.describe()
        );
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("read accepted capture");
        for table in [
            "event_search_request_headers",
            "event_search_request_selections",
        ] {
            let rows = reader
                .execute(
                    &format!("SELECT COUNT(*) AS total FROM {table}"),
                    &std::collections::HashMap::new(),
                )
                .expect("read selection count");
            assert_eq!(
                rows.rows[0][0],
                contextdb_core::Value::Int64(1),
                "unaccepted rows in {table}"
            );
        }
        drop(held);
        let status = reader
            .event_history_status(context_graph::ContextSelector::one(
                reader
                    .graph()
                    .list_contexts()
                    .expect("Context")
                    .into_iter()
                    .find(|context| context.name == "rejected-selection")
                    .expect("selected Context")
                    .id,
            ))
            .expect("capture remains");
        assert_eq!(status.contexts[0].captured_events, 2);
        assert_eq!(
            read_event_search_worker_policy(&machine.data_root)
                .expect("policy unchanged")
                .as_deref(),
            Some(existing.as_str())
        );
    }

    #[test]
    fn a_deferred_append_stays_outside_the_accepted_frontier_even_when_it_would_fail() {
        let machine = Machine::with_nomic();
        let project = machine.project("deferred-frontier");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold first claim");
        let observer = request_observer(&machine, "deferred-frontier-observer");
        let accepted =
            import_selected(&machine, &project, "deferred-frontier", LOW, "1", &observer);
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let source = machine
            .history_root()
            .join(event_support::claude_project_key(&project))
            .join("session-B.jsonl");
        // Preserve the committed UUIDs and bytes before generating only the new suffix.
        let original = std::fs::read_to_string(&source).expect("captured prefix");
        let oversized = "unselected suffix ".repeat(65_000);
        write_session(
            &machine.history_root(),
            &project,
            "session-B",
            &[
                ("B request selected before C exists", "B answer"),
                (&oversized, "unselected answer"),
            ],
        );
        let expanded = std::fs::read_to_string(&source).expect("expanded source");
        let suffix = expanded.lines().skip(2).collect::<Vec<_>>().join("\n");
        event_support::write_test_file(&source, format!("{original}{suffix}\n"))
            .expect("deferred append");
        let deferred = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "deferred-frontier",
                "--search",
                "defer",
            ],
        );
        assert_eq!(deferred.code, 0, "{}", deferred.describe());
        drop(held);
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        let actual = actual_budgets(&observer);
        assert!(!actual.is_empty(), "prefix must be projected");
        assert!(
            actual.iter().all(|batch| batch["session"] == "session-B"
                && batch["committed"] == 0
                && batch["through"] == 2
                && batch["limit"] == LOW),
            "{actual:?}"
        );
        assert!(
            worker_policy(&machine.data_root)["pending_background_request"].is_null(),
            "selected prefix must be acknowledged"
        );
        assert_selection_counts(&machine, 0);
        let status = machine.run(&project, &["status", "--context", "deferred-frontier"]);
        assert_eq!(status.code, 0, "{}", status.describe());
        assert_eq!(
            u64_at(&status.json(), "/history/search/covered_events"),
            2,
            "{}",
            status.describe()
        );
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("projection reader");
        let rows = reader
            .execute(
                "SELECT embedded_event_count, sanitized_failure FROM event_search_generations",
                &std::collections::HashMap::new(),
            )
            .expect("projection counters");
        assert_eq!(rows.rows.len(), 1);
        assert_eq!(rows.rows[0][0], contextdb_core::Value::Int64(2));
        assert_eq!(
            rows.rows[0][1],
            contextdb_core::Value::Null,
            "unselected suffix cannot refuse the selected request"
        );
        let context = reader
            .graph()
            .list_contexts()
            .expect("Contexts")
            .into_iter()
            .find(|context| context.name == "deferred-frontier")
            .expect("selected Context");
        let mut passages = reader
            .open_cursor(
                "SELECT last_event_sequence, rendered_text FROM event_search_passages WHERE context_id = $context_id AND is_current = true AND is_searchable = true ORDER BY last_event_sequence DESC LIMIT 1",
                &std::collections::HashMap::from([("context_id".into(), contextdb_core::Value::Uuid(context.id.as_uuid()))]),
            )
            .expect("published prefix cursor");
        let prefix = passages
            .first_page()
            .expect("published prefix page")
            .rows
            .clone();
        passages.close().expect("close prefix cursor");
        assert_eq!(prefix.len(), 1, "one published prefix passage");
        assert_eq!(prefix[0][0], contextdb_core::Value::Int64(2));
        let contextdb_core::Value::Text(text) = &prefix[0][1] else {
            panic!("prefix text");
        };
        assert!(text.contains("B request selected before C exists") && text.contains("B answer"));
        assert!(!text.contains("unselected suffix"));
    }

    #[test]
    fn later_capture_keeps_the_earlier_ceiling_on_its_selected_session() {
        let machine = Machine::with_nomic();
        let project = machine.project("durable-selection");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let observer = request_observer(&machine, "durable-selection-observer");
        let low = import_selected(&machine, &project, "durable-selection", LOW, "1", &observer);
        assert_eq!(low.code, 0, "{}", low.describe());
        let first =
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        let reader = context_graph::ConsumerReader::open(
            machine.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("read durable selection");
        let context = reader
            .graph()
            .list_contexts()
            .expect("Context")
            .into_iter()
            .find(|context| context.name == "durable-selection")
            .expect("selected Context");
        let selected = reader
            .event_search_selected_work(
                context.id,
                uuid::Uuid::parse_str(first["selected_work"].as_str().expect("selection identity"))
                    .expect("UUID"),
            )
            .expect("accepted selection");
        assert_eq!(selected.streams[0].session_ref, "session-B");
        assert_eq!(selected.streams[0].frontier, 2);
        drop(reader);
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("newer C request", "C answer")],
        );
        let high = import_selected(
            &machine,
            &project,
            "durable-selection",
            HIGH,
            "all",
            &observer,
        );
        assert_eq!(high.code, 0, "{}", high.describe());
        assert_eq!(
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0],
            first,
            "later captures must preserve the earlier accepted selection"
        );
        drop(held);
        let worker = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        assert_selected_b_and_c_completed(&machine, &project, &observer, LOW, HIGH);

        // The same session can grow too: split its next claim at the accepted source frontier
        // so the earlier low ceiling does not govern a later request's exclusive appended work.
        let appended = Machine::with_nomic();
        let appended_project = appended.project("appended-frontier");
        let growing = write_session(
            &appended.history_root(),
            &appended_project,
            "growing-session",
            &[("first request", "first answer")],
        );
        let prefix = std::fs::read_to_string(&growing).expect("original immutable prefix");
        std::fs::create_dir_all(&appended.data_root).expect("append queue root");
        let held = acquire_worker_lock(&appended.data_root.join("event-search-worker.lock"))
            .expect("hold append claims");
        let append_observer = request_observer(&appended, "appended-frontier-observer");
        let first = import_selected(
            &appended,
            &appended_project,
            "appended-frontier",
            LOW,
            "1",
            &append_observer,
        );
        assert_eq!(first.code, 0, "{}", first.describe());
        write_session(
            &appended.history_root(),
            &appended_project,
            "growing-session",
            &[
                ("first request", "first answer"),
                ("later request", "later answer"),
            ],
        );
        let expanded = std::fs::read_to_string(&growing).expect("expanded session");
        let suffix = expanded.lines().skip(2).collect::<Vec<_>>().join("\n");
        event_support::write_test_file(&growing, format!("{prefix}{suffix}\n"))
            .expect("append while preserving source prefix");
        let later = import_selected(
            &appended,
            &appended_project,
            "appended-frontier",
            HIGH,
            "all",
            &append_observer,
        );
        assert_eq!(later.code, 0, "{}", later.describe());
        drop(held);
        let worker = appended.run_with_env(
            &appended_project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &append_observer,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        let actual = actual_budgets(&append_observer);
        assert!(
            actual.iter().any(|batch| batch["committed"] == 0
                && batch["limit"] == LOW
                && batch["through"] == 2),
            "{actual:?}"
        );
        assert!(
            actual.iter().any(|batch| batch["committed"] == 2
                && batch["limit"] == HIGH
                && batch["through"] == 4),
            "{actual:?}"
        );
        assert!(worker_policy(&appended.data_root)["pending_background_request"].is_null());
        assert_ready(&appended, &appended_project, "appended-frontier");
        {
            let reader = context_graph::ConsumerReader::open(
                appended.db_path(),
                context_graph::event_history_reader_declaration(),
            )
            .expect("widened passage reader");
            let mut cursor = reader.open_cursor(
                "SELECT last_event_sequence, rendered_text, is_current, is_searchable FROM event_search_passages ORDER BY last_event_sequence ASC LIMIT 3",
                &std::collections::HashMap::new()).expect("bounded widened passage cursor");
            let passages = cursor
                .first_page()
                .expect("widened passage page")
                .rows
                .clone();
            cursor.close().expect("close widened passage cursor");
            assert_eq!(
                passages.len(),
                2,
                "prefix is published then replaced once: {passages:?}"
            );
            assert_eq!(passages[0][0], contextdb_core::Value::Int64(2));
            assert_eq!(passages[0][2], contextdb_core::Value::Bool(false));
            assert_eq!(passages[1][0], contextdb_core::Value::Int64(4));
            assert_eq!(passages[1][2], contextdb_core::Value::Bool(true));
            assert_eq!(passages[1][3], contextdb_core::Value::Bool(true));
            let contextdb_core::Value::Text(text) = &passages[1][1] else {
                panic!("widened text");
            };
            for content in [
                "first request",
                "first answer",
                "later request",
                "later answer",
            ] {
                assert!(text.contains(content), "missing {content}: {text}");
            }
        }

        // Selected work must not consume the bounded queue's byte allowance as sessions grow.
        let many = Machine::with_nomic();
        let many_project = many.project("many-selected-sessions");
        for index in 0..40 {
            write_session(
                &many.history_root(),
                &many_project,
                &format!("selected-session-{index:02}"),
                &[("selected request", "selected answer")],
            );
        }
        std::fs::create_dir_all(&many.data_root).expect("large selection root");
        let held = acquire_worker_lock(&many.data_root.join("event-search-worker.lock"))
            .expect("hold large selection claims");
        let many_observer = request_observer(&many, "many-selection-observer");
        let accepted = import_selected(
            &many,
            &many_project,
            "many-selected-sessions",
            HIGH,
            "all",
            &many_observer,
        );
        assert_eq!(accepted.code, 0, "{}", accepted.describe());
        let policy = read_event_search_worker_policy(&many.data_root)
            .expect("bounded queue")
            .expect("accepted queue");
        assert!(policy.len() as u64 <= cg_node_fs::EVENT_SEARCH_WORKER_POLICY_MAX_BYTES);
        let request =
            worker_policy(&many.data_root)["pending_background_request"]["write_requests"][0]
                .clone();
        let reader = context_graph::ConsumerReader::open(
            many.db_path(),
            context_graph::event_history_reader_declaration(),
        )
        .expect("large durable selection reader");
        let context = reader
            .graph()
            .list_contexts()
            .expect("large selection Context")
            .into_iter()
            .find(|context| context.name == "many-selected-sessions")
            .expect("selected Context");
        let selection = reader
            .event_search_selected_work(
                context.id,
                uuid::Uuid::parse_str(
                    request["selected_work"]
                        .as_str()
                        .expect("selection identity"),
                )
                .expect("UUID"),
            )
            .expect("large durable selection");
        assert_eq!(selection.streams.len(), 40);
        drop(reader);
        drop(held);
    }

    #[test]
    fn selected_session_ceiling_survives_restart_before_the_first_claim() {
        let machine = Machine::with_nomic();
        let project = machine.project("durable-selection");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let observer = request_observer(&machine, "restart-selection-observer");
        let low = import_selected(&machine, &project, "durable-selection", LOW, "1", &observer);
        assert_eq!(low.code, 0, "{}", low.describe());
        drop(held);
        let pause = machine.root.join("before-first-claim");
        std::fs::create_dir_all(&pause).expect("pre-claim barrier");
        let mut first_worker = machine.spawn_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            "CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER",
            &pause,
        );
        await_marker(&pause, "pre-catch-up-entered");
        write_session(
            &machine.history_root(),
            &project,
            "session-C",
            &[("newer C request", "C answer")],
        );
        let high = import_selected(
            &machine,
            &project,
            "durable-selection",
            HIGH,
            "all",
            &observer,
        );
        assert_eq!(high.code, 0, "{}", high.describe());
        assert_eq!(str_at(&high.json(), "/search/worker"), "already_running");
        let durable = worker_policy(&machine.data_root)["pending_background_request"].clone();
        first_worker.kill().expect("interrupt pre-claim worker");
        first_worker.wait().expect("reap interrupted worker");
        machine.wait_for_search_worker_exit();
        assert_eq!(
            worker_policy(&machine.data_root)["pending_background_request"],
            durable
        );
        assert!(actual_budgets(&observer).is_empty());
        let resumed = machine.run_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(resumed.code, 0, "{}", resumed.describe());
        assert_selected_b_and_c_completed(&machine, &project, &observer, LOW, HIGH);
    }

    #[test]
    fn work_completed_by_another_request_is_acknowledged_without_reprojection() {
        let machine = Machine::with_nomic();
        let project = machine.project("already-completed-selection");
        capture_b(&machine, &project);
        std::fs::create_dir_all(&machine.data_root).expect("queue root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold claims");
        let observer = request_observer(&machine, "completed-selection-observer");
        let queued = import_selected(
            &machine,
            &project,
            "completed-selection",
            LOW,
            "1",
            &observer,
        );
        assert_eq!(queued.code, 0, "{}", queued.describe());
        assert_eq!(
            worker_policy(&machine.data_root)["pending_background_request"]["write_requests"][0]["completed"],
            false
        );
        drop(held);
        // The foreground request commits the queued request's whole frontier before its worker
        // takes a claim. Its ordinary handoff must acknowledge the existing durable request.
        let foreground = machine.run_with_env(
            &project,
            &[
                "import",
                ".",
                "--context",
                "completed-selection",
                "--search",
                "wait",
                "--search-sessions",
                "all",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        assert_eq!(foreground.code, 0, "{}", foreground.describe());
        await_acknowledgement(&machine);
        machine.wait_for_search_worker_exit();
        assert!(worker_policy(&machine.data_root)["pending_background_request"].is_null());
        assert!(
            actual_budgets(&observer).is_empty(),
            "a covered frontier must not take another projection claim"
        );
        assert_eq!(
            observed_request_caps(&observer),
            vec![1],
            "only the foreground projection run is needed"
        );
        assert_ready(&machine, &project, "completed-selection");

        // Distinct queued scopes complete from one shared pass too: the all-sessions request
        // commits the newest request's frontier, and that second request takes no further claim.
        let queued = Machine::with_nomic();
        let queued_project = queued.project("durable-selection");
        capture_b(&queued, &queued_project);
        write_session(
            &queued.history_root(),
            &queued_project,
            "session-C",
            &[("shared newest request", "shared newest answer")],
        );
        std::fs::create_dir_all(&queued.data_root).expect("completed overlap root");
        let held = acquire_worker_lock(&queued.data_root.join("event-search-worker.lock"))
            .expect("hold overlap claims");
        let observed = request_observer(&queued, "completed-overlap-observer");
        for (limit, sessions) in [(HIGH, "all"), (LOW, "1")] {
            let accepted = import_selected(
                &queued,
                &queued_project,
                "durable-selection",
                limit,
                sessions,
                &observed,
            );
            assert_eq!(accepted.code, 0, "{}", accepted.describe());
        }
        drop(held);
        let worker = queued.run_with_env(
            &queued_project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observed,
        );
        assert_eq!(worker.code, 0, "{}", worker.describe());
        assert_selected_b_and_c_completed(&queued, &queued_project, &observed, HIGH, LOW);
        let actual = actual_budgets(&observed);
        assert_eq!(
            actual
                .iter()
                .filter(|batch| batch["session"] == "session-C")
                .count(),
            1,
            "completed shared work must not be projected twice: {actual:?}"
        );
        assert_eq!(
            observed_request_caps(&observed),
            vec![1],
            "one pass must complete both accepted scopes"
        );
    }

    #[test]
    fn overlapping_imports_keep_each_contexts_write_ceiling() {
        let machine = Machine::with_nomic();
        let first_project = machine.project("first-budget");
        let second_project = machine.project("second-budget");
        write_session(
            &machine.history_root(),
            &first_project,
            "first-session",
            &[("first request", "first answer")],
        );
        write_session(
            &machine.history_root(),
            &second_project,
            "second-session",
            &[("second request", "second answer")],
        );
        let observer = batch_observer(&machine, "overlapping-budgets");
        let first = import(&machine, &first_project, "first-budget", HIGH, &observer);
        assert_eq!(first.code, 0, "{}", first.describe());
        await_marker(&observer, "entered");
        // The active batch has a claim, but embedding retains no writable Store. A second
        // independent capture must still succeed before releasing the first batch.
        let second = import(&machine, &second_project, "second-budget", LOW, &observer);
        event_support::write_test_file(observer.join("release"), b"continue\n")
            .expect("release active batch");
        await_acknowledgement(&machine);
        machine.wait_for_search_worker_exit();
        assert_eq!(second.code, 0, "{}", second.describe());
        assert_eq!(str_at(&second.json(), "/search/worker"), "already_running");
        let seen = budgets(&observer);
        let first_context = &seen.first().expect("first budget admission").0;
        assert!(
            seen.iter()
                .filter(|(context, _)| context == first_context)
                .all(|(_, limit)| *limit == HIGH),
            "{seen:?}"
        );
        assert!(
            seen.iter()
                .any(|(context, limit)| context != first_context && *limit == LOW),
            "{seen:?}"
        );
        assert!(
            seen.iter()
                .filter(|(context, _)| context != first_context)
                .all(|(_, limit)| *limit == LOW),
            "{seen:?}"
        );
        assert_ready(&machine, &first_project, "first-budget");
        assert_ready(&machine, &second_project, "second-budget");

        // Requests selecting different sessions in one Context share only the selected overlap.
        let scoped = Machine::with_nomic();
        let scoped_project = scoped.project("scoped-budgets");
        let old = write_session(
            &scoped.history_root(),
            &scoped_project,
            "old-session",
            &[("older request", "older answer")],
        );
        let old_source = std::fs::read_to_string(&old).expect("read old fixture");
        event_support::write_test_file(old, old_source.replace("2026-09-01", "2026-08-01"))
            .expect("order old session first");
        let middle = write_session(
            &scoped.history_root(),
            &scoped_project,
            "middle-session",
            &[("middle request", "middle answer")],
        );
        let middle_source = std::fs::read_to_string(&middle).expect("read middle fixture");
        event_support::write_test_file(middle, middle_source.replace("2026-09-01", "2026-08-15"))
            .expect("keep exclusive work after the first claim");
        write_session(
            &scoped.history_root(),
            &scoped_project,
            "new-session",
            &[("newer request", "newer answer")],
        );
        let scope_observer = batch_observer(&scoped, "scoped-admission");
        let older = scoped.run_with_env(
            &scoped_project,
            &[
                "import",
                ".",
                "--context",
                "scoped-budgets",
                "--search",
                "background",
                "--search-sessions",
                "all",
                "--search-order",
                "oldest",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &scope_observer,
        );
        assert_eq!(older.code, 0, "{}", older.describe());
        await_marker(&scope_observer, "entered");
        let already_seen = budgets(&scope_observer).len();
        let newer = scoped.run_with_env(
            &scoped_project,
            &[
                "import",
                ".",
                "--context",
                "scoped-budgets",
                "--search",
                "background",
                "--search-sessions",
                "1",
                "--search-order",
                "newest",
                "--embedding-workers",
                "1",
                "--write-memory-limit-bytes",
                &LOW.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &scope_observer,
        );
        event_support::write_test_file(scope_observer.join("release"), b"serve both selectors")
            .expect("release scoped claims");
        await_acknowledgement(&scoped);
        scoped.wait_for_search_worker_exit();
        assert_eq!(newer.code, 0, "{}", newer.describe());
        let after_overlap = budgets(&scope_observer);
        let after_overlap = &after_overlap[already_seen..];
        assert!(
            after_overlap.iter().any(|(_, limit)| *limit == HIGH),
            "exclusive older work lost its ceiling: {after_overlap:?}"
        );
        assert!(
            after_overlap.iter().any(|(_, limit)| *limit == LOW),
            "shared newer work missed the lower ceiling: {after_overlap:?}"
        );
        assert_ready(&scoped, &scoped_project, "scoped-budgets");
    }

    #[test]
    fn shared_pending_work_resumes_under_the_lower_persisted_ceiling_without_reprojection() {
        // Both imports need the same unfinished events before any projection claim exists.
        // They must share one batch at the lower ceiling rather than build it twice.
        let queued = Machine::with_nomic();
        let queued_project = queued.project("queued-shared-budget");
        write_session(
            &queued.history_root(),
            &queued_project,
            "queued-session",
            &[("shared pending source", "shared pending answer")],
        );
        std::fs::create_dir_all(&queued.data_root).expect("create queued worker root");
        let held = acquire_worker_lock(&queued.data_root.join("event-search-worker.lock"))
            .expect("hold queued writer admission");
        let queued_observer = request_observer(&queued, "queued-shared-observer");
        for limit in [HIGH, LOW] {
            let imported = import(
                &queued,
                &queued_project,
                "queued-shared-budget",
                limit,
                &queued_observer,
            );
            assert_eq!(imported.code, 0, "{}", imported.describe());
            assert_eq!(
                str_at(&imported.json(), "/search/worker"),
                "already_running"
            );
            assert_eq!(u64_at(&imported.json(), "/search/pending_events"), 2);
        }
        drop(held);
        let completed_shared = queued.run_with_env(
            &queued_project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &queued_observer,
        );
        assert_eq!(completed_shared.code, 0, "{}", completed_shared.describe());
        assert!(!budgets(&queued_observer).is_empty());
        assert!(
            budgets(&queued_observer)
                .iter()
                .all(|(_, limit)| *limit == LOW)
        );
        assert_ready(&queued, &queued_project, "queued-shared-budget");

        // A killed writer leaves a live durable lease. The replacement must keep the request
        // until that lease is released, then resume under its persisted ceiling.
        let orphan = Machine::with_nomic();
        let orphan_project = orphan.project("orphan-budget");
        write_session(
            &orphan.history_root(),
            &orphan_project,
            "orphan-session",
            &[("unfinished claim", "resume at the selected ceiling")],
        );
        let orphan_pause = batch_observer(&orphan, "orphan-claim");
        let captured = import(
            &orphan,
            &orphan_project,
            "orphan-budget",
            LOW,
            &orphan_pause,
        );
        assert_eq!(captured.code, 0, "{}", captured.describe());
        await_marker(&orphan_pause, "entered");
        terminate_observed_background_worker(&orphan_pause);
        orphan.wait_for_search_worker_exit();
        let lease_wait = batch_observer(&orphan, "orphan-lease-wait");
        event_support::write_test_file(lease_wait.join("numbered-passes"), b"1")
            .expect("observe the lease-only pass");
        let resumed = orphan.spawn_with_env(
            &orphan_project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &lease_wait,
        );
        await_marker(&lease_wait, "entered.1");
        assert_eq!(
            worker_policy(&orphan.data_root)["pending_background_request"]["write_requests"][0]["completed"],
            false
        );
        {
            let db = contextdb_engine::Database::open(orphan.db_path())
                .expect("open the closed lease fixture");
            let changed = db
                .execute(
                    "UPDATE event_search_leases SET expires_at = $expired",
                    &std::collections::HashMap::from([(
                        "expired".into(),
                        contextdb_core::Value::Timestamp(0),
                    )]),
                )
                .expect("expire the orphan claim deterministically");
            assert_eq!(changed.rows_affected, 1);
        }
        event_support::write_test_file(lease_wait.join("release"), b"resume expired claim")
            .expect("release replacement");
        let output = resumed.wait_with_output().expect("replacement completes");
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!budgets(&lease_wait).is_empty());
        assert!(budgets(&lease_wait).iter().all(|(_, limit)| *limit == LOW));
        assert_ready(&orphan, &orphan_project, "orphan-budget");

        for (completed_limit, next_limit, startup_limit) in [(HIGH, LOW, HIGH), (LOW, HIGH, LOW)] {
            let machine = Machine::with_nomic();
            let project = machine.project("shared-budget");
            let first_turn = ("the original request", "the original answer");
            let original = write_session(
                &machine.history_root(),
                &project,
                "shared-session",
                &[first_turn],
            );
            let observer = batch_observer(&machine, "shared-budget-pause");
            event_support::write_test_file(observer.join("numbered-passes"), b"2")
                .expect("pause after completed pass");
            event_support::write_test_file(observer.join("release.1"), b"complete first batch\n")
                .expect("release first claim");
            let first = import(
                &machine,
                &project,
                "shared-budget",
                completed_limit,
                &observer,
            );
            assert_eq!(first.code, 0, "{}", first.describe());
            await_marker(&observer, "entered.2");
            // The first projection finished before the pause, so restart has no orphan lease.
            assert_ready(&machine, &project, "shared-budget");
            let additional = write_session(
                &machine.history_root(),
                &project,
                "additional-session",
                &[("new request", "new answer")],
            );
            let mut prefix =
                std::fs::read_to_string(&original).expect("read unchanged captured prefix");
            let suffix = std::fs::read_to_string(&additional).expect("read new source records");
            prefix.push_str(&suffix.replace("additional-session", "shared-session"));
            event_support::write_test_file(&original, prefix)
                .expect("append without changing captured source identity");
            std::fs::remove_file(additional).expect("remove temporary source fixture");
            let second = import(&machine, &project, "shared-budget", next_limit, &observer);
            assert_eq!(second.code, 0, "{}", second.describe());
            assert_eq!(str_at(&second.json(), "/search/worker"), "already_running");
            let pending = worker_policy(&machine.data_root);
            let requests = pending["pending_background_request"]["write_requests"]
                .as_array()
                .expect("persisted request ceilings");
            // The completed first selection has already retired before entered.2.
            // Only the newly accepted frontier remains an owner of resumed work.
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0]["completed"], false);
            assert_eq!(requests[0]["selection_retired"], false);
            assert_eq!(
                requests[0]["memory_limit_bytes"].as_u64(),
                Some(next_limit as u64)
            );
            assert_selection_counts(&machine, 1);
            assert!(
                std::fs::read_to_string(observer.join("cleanup-write-memory-limits"))
                    .unwrap()
                    .lines()
                    .any(|limit| limit == completed_limit.to_string()),
                "the completed prefix must retire under its original ceiling"
            );
            terminate_observed_background_worker(&observer);
            machine.wait_for_search_worker_exit();
            let resumed_observer = request_observer(&machine, "resumed-budget");
            let resumed = machine.run_with_env(
                &project,
                &[
                    "__event-search-worker",
                    &envelope(),
                    "--write-memory-limit-bytes",
                    &startup_limit.to_string(),
                ],
                REQUEST_OBSERVER_ENV,
                &resumed_observer,
            );
            assert_eq!(resumed.code, 0, "{}", resumed.describe());
            assert!(!budgets(&resumed_observer).is_empty());
            assert!(
                budgets(&resumed_observer)
                    .iter()
                    .all(|(_, limit)| *limit == next_limit)
            );
            let actual = actual_budgets(&resumed_observer);
            assert!(!actual.is_empty(), "new frontier was never admitted");
            assert!(
                actual
                    .iter()
                    .all(|batch| batch["committed"].as_u64().unwrap() >= 2),
                "completed prefix was reprojected: {actual:?}"
            );
            assert!(
                actual.iter().any(|batch| batch["through"] == 4),
                "{actual:?}"
            );
            assert_ready(&machine, &project, "shared-budget");
            let completed = fingerprint(&machine.data_root);
            let repeated_observer = request_observer(&machine, "completed-budget");
            let repeated = machine.run_with_env(
                &project,
                &["__event-search-worker", &envelope()],
                REQUEST_OBSERVER_ENV,
                &repeated_observer,
            );
            assert_eq!(repeated.code, 0, "{}", repeated.describe());
            assert!(
                budgets(&repeated_observer).is_empty(),
                "completed projection must not run again"
            );
            assert_eq!(
                fingerprint(&machine.data_root),
                completed,
                "no completed work is duplicated"
            );
        }
    }

    #[test]
    fn the_next_batch_refuses_an_insufficient_persisted_ceiling_and_keeps_its_request() {
        let machine = Machine::with_nomic();
        let project = machine.project("insufficient-budget");
        let history = write_session(
            &machine.history_root(),
            &project,
            "bounded-session",
            &[("durable capture", "keep the capture after refusal")],
        );
        std::fs::create_dir_all(&machine.data_root).expect("create isolated worker root");
        let held = acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold worker admission");
        let observer = batch_observer(&machine, "insufficient-budget-pause");
        let imported = import(&machine, &project, "insufficient-budget", HIGH, &observer);
        assert_eq!(imported.code, 0, "{}", imported.describe());
        drop(held);
        let worker = machine.spawn_with_env(
            &project,
            &[
                "__event-search-worker",
                &envelope(),
                "--write-memory-limit-bytes",
                &HIGH.to_string(),
            ],
            REQUEST_OBSERVER_ENV,
            &observer,
        );
        await_marker(&observer, "entered");
        let mut pending = worker_policy(&machine.data_root);
        let token = pending["pending_background_request"]["token"].clone();
        let original = pending["pending_background_request"]["write_requests"][0].clone();
        // Supply a persisted request too small even to reopen the existing writer state. This
        // fixture isolates background admission from foreground capture's own admission.
        for value in pending["pending_background_request"]["write_requests"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
        {
            value["memory_limit_bytes"] = serde_json::json!(1);
        }
        write_event_search_worker_policy(&machine.data_root, &pending.to_string())
            .expect("deliver next request ceiling");
        event_support::write_test_file(
            observer.join("release"),
            b"finish admitted batch and reopen\n",
        )
        .expect("release worker");
        let output = worker
            .wait_with_output()
            .expect("worker refuses bounded reopen");
        assert_eq!(output.status.code(), Some(1));
        let refusal = String::from_utf8_lossy(&output.stderr);
        assert!(
            refusal.contains("memory") && refusal.contains('1'),
            "{refusal}"
        );
        let seen = budgets(&observer);
        assert_eq!(seen.first().unwrap().1, HIGH);
        assert_eq!(seen.last().unwrap().1, 1);
        assert_eq!(
            worker_policy(&machine.data_root)["pending_background_request"]["token"],
            token
        );
        assert_selection_counts(&machine, 1);
        let retained = worker_policy(&machine.data_root);
        let refused = &retained["pending_background_request"]["write_requests"][0];
        assert_eq!(refused["token"], original["token"]);
        assert_eq!(refused["selected_work"], original["selected_work"]);
        assert_eq!(refused["memory_limit_bytes"], 1);
        assert_eq!(refused["completed"], true);
        assert_eq!(refused["refused_memory"], true);
        assert_eq!(refused["selection_retired"], false);
        let readable = machine.run(&project, &["status", "--context", "insufficient-budget"]);
        assert_eq!(readable.code, 0, "{}", readable.describe());
        assert_eq!(str_at(&readable.json(), "/history/search/status"), "failed");
        let message = str_at(&readable.json(), "/history/search/last_failure").to_owned();
        assert!(
            message.contains("allowed 1 bytes") && message.contains("--write-memory-limit-bytes"),
            "{message}"
        );
        let caught_up_observer = request_observer(&machine, "caught-up-retry");
        let caught_up = import(
            &machine,
            &project,
            "insufficient-budget",
            HIGH,
            &caught_up_observer,
        );
        assert_eq!(caught_up.code, 1, "{}", caught_up.describe());
        assert_eq!(
            str_at(&caught_up.json(), "/condition"),
            "WriteDidNotFitTheMemoryLimit"
        );
        assert!(
            budgets(&caught_up_observer).is_empty(),
            "completed prefix was reprojected"
        );
        let prefix = std::fs::read(&history).unwrap();
        write_session(
            &machine.history_root(),
            &project,
            "bounded-session",
            &[("new pending closed turn", "higher budget serves new work")],
        );
        let suffix = std::fs::read(&history).unwrap();
        let mut appended = prefix.clone();
        appended.extend(suffix);
        event_support::write_test_file(&history, &appended).unwrap();
        assert!(std::fs::read(&history).unwrap().starts_with(&prefix));
        let retry_observer = request_observer(&machine, "affordable-retry");
        let resumed = import(
            &machine,
            &project,
            "insufficient-budget",
            LOW,
            &retry_observer,
        );
        assert_eq!(resumed.code, 0, "{}", resumed.describe());
        await_marker(&retry_observer, "worker-store-write-memory-limit-bytes");
        machine.wait_for_search_worker_exit();
        assert_ready(&machine, &project, "insufficient-budget");
        // New work retires independently while the old impossible cleanup stays inspectable.
        assert_selection_counts(&machine, 1);
        let retained = worker_policy(&machine.data_root);
        let requests = retained["pending_background_request"]["write_requests"]
            .as_array()
            .unwrap();
        assert_eq!(requests.len(), 1, "{retained}");
        assert_eq!(requests[0]["token"], original["token"]);
        assert_eq!(requests[0]["selected_work"], original["selected_work"]);
        assert_eq!(requests[0]["memory_limit_bytes"], 1);
        assert_eq!(requests[0]["completed"], true);
        assert_eq!(requests[0]["refused_memory"], true);
        assert_eq!(requests[0]["selection_retired"], false);
        let seen = budgets(&retry_observer);
        assert!(!seen.is_empty(), "new closed turn was not projected");
        assert!(seen.iter().all(|(_, limit)| *limit == LOW));
        let actual = actual_budgets(&retry_observer);
        assert!(!actual.is_empty(), "new frontier was never admitted");
        assert!(
            actual
                .iter()
                .all(|batch| batch["committed"].as_u64().unwrap() >= 2),
            "completed prefix was reprojected: {actual:?}"
        );
        assert!(
            actual.iter().any(|batch| batch["through"] == 4),
            "retry did not accept its distinct later frontier: {actual:?}"
        );
        let status = machine.run(&project, &["status", "--context", "insufficient-budget"]);
        assert_eq!(u64_at(&status.json(), "/history/search/covered_events"), 4);
    }
}
