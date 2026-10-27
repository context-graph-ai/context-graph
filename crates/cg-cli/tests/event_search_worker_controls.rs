//! Actual CLI journeys for the private event-search worker-policy lifetime.

mod event_support;
#[path = "support/store_contention.rs"]
mod store_contention;

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
use std::time::Duration;

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
use event_support::{Machine, str_at, wait_for_state, write_session};
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
use store_contention::HoldingChild;

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
const OBSERVER_ENV: &str = "CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER";

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
const PRE_CATCH_UP_ENV: &str = "CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER";

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn worker_writer_holder_child() {
    store_contention::run_writer_holder_child_if_asked();
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn worker_envelope() -> String {
    serde_json::json!({
        "profile": context_graph::EventSearchProfile::default(),
        "embedding_workers": 1,
        "ignored_publication": null,
        "retry_worker_lock": false,
    })
    .to_string()
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observer(machine: &Machine, name: &str, pause_batches: bool) -> std::path::PathBuf {
    let path = machine.root.join(name);
    std::fs::create_dir_all(&path).expect("create worker observer directory");
    if pause_batches {
        event_support::write_test_file(
            path.join("observe-embedding-worker-batches"),
            b"observe and pause\n",
        )
        .expect("enable existing batch observer");
    } else {
        event_support::write_test_file(path.join("release"), b"do not pause\n")
            .expect("pre-release the existing handoff barrier");
    }
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
        .map(|line| line.parse().expect("observed batch cap is numeric"))
        .collect()
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observed_request_caps(observer: &std::path::Path) -> Vec<usize> {
    std::fs::read_to_string(observer.join("embedding-worker-requests"))
        .unwrap_or_default()
        .lines()
        .map(|line| line.parse().expect("observed request cap is numeric"))
        .collect()
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn policy(root: &std::path::Path) -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(root.join(cg_node_fs::EVENT_SEARCH_WORKER_POLICY))
            .expect("read delivered worker policy"),
    )
    .expect("worker policy is valid JSON")
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn terminate_observed_worker(observer: &std::path::Path) {
    let group = std::fs::read_to_string(observer.join("embedding-worker-process-group"))
        .expect("read owned worker group")
        .trim()
        .parse::<i32>()
        .expect("worker group is a pid");
    // SAFETY: this fixture owns the detached worker group recorded by the existing test observer.
    assert_eq!(unsafe { libc::kill(-group, libc::SIGTERM) }, 0);
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn queued_forget_uses_configured_cap_and_finishes_after_foreground_releases() {
    let machine = Machine::with_nomic();
    let forget_project = machine.project("queued-forget");
    write_session(
        &machine.history_root(),
        &forget_project,
        "remove-me",
        &[("remove this queued history", "after the foreground turn")],
    );
    write_session(
        &machine.history_root(),
        &forget_project,
        "keep-me",
        &[("keep this queued history", "and rebuild it at cap two")],
    );
    let config_dir = machine.data_root.join("cg");
    std::fs::create_dir_all(&config_dir).expect("create config directory");
    event_support::write_test_file(
        config_dir.join("fabric.toml"),
        "[search]\nembedding_workers = 2\n",
    )
    .expect("write configured cap");
    let seeded = machine.run(
        &forget_project,
        &[
            "import",
            ".",
            "--context",
            "queued-forget",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(seeded.code, 0, "{}", seeded.describe());

    let foreground_project = machine.project("queued-forget-foreground");
    write_session(
        &machine.history_root(),
        &foreground_project,
        "foreground-session",
        &[("hold the foreground turn", "at the one-off cap one")],
    );
    let observer = observer(&machine, "queued-forget-observer", true);
    let foreground = machine.spawn_with_env(
        &foreground_project,
        &[
            "import",
            ".",
            "--context",
            "queued-forget-foreground",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
        OBSERVER_ENV,
        &observer,
    );
    let paused = wait_for_state(Duration::from_secs(10), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered").exists().then_some(()))
    })
    .expect("foreground pause observation cannot fail");
    assert_eq!(paused, Some(()), "foreground did not reach its batch");

    let forgotten = machine.run(
        &forget_project,
        &[
            "forget",
            "--session",
            "remove-me",
            "--context",
            "queued-forget",
            "--yes",
        ],
    );
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    assert_eq!(
        str_at(&forgotten.json(), "/search/worker"),
        "already_running"
    );

    event_support::write_test_file(observer.join("release"), b"continue\n")
        .expect("release foreground");
    let output = foreground
        .wait_with_output()
        .expect("wait for foreground cg");
    assert!(output.status.success(), "foreground failed: {output:?}");
    let ready = machine.wait_for_search_ready(&forget_project, "queued-forget");
    assert_eq!(
        ready.pointer("/history/search/pending_events"),
        Some(&serde_json::json!(0))
    );
    machine.wait_for_search_worker_exit();

    let caps = observed_batch_caps(&observer);
    assert_eq!(caps.first(), Some(&1), "foreground cap changed: {caps:?}");
    assert_eq!(
        caps.get(1),
        Some(&2),
        "queued forget's first effective claim did not use configured cap two: {caps:?}"
    );
    let found = machine.run(&forget_project, &["find", "keep this queued history"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    assert!(!found.json()["hits"].as_array().unwrap().is_empty());
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn forget_during_background_work_keeps_the_live_worker_cap() {
    let machine = Machine::with_nomic();
    let project = machine.project("forget-live-cap");
    let turns = [(
        "one two three four five six seven eight",
        "same words again",
    )];
    write_session(&machine.history_root(), &project, "remove-me", &turns);
    write_session(&machine.history_root(), &project, "keep-me", &turns);
    let observer = observer(&machine, "forget-live-observer", true);
    let imported = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "forget-live-cap",
            "--search",
            "background",
            "--embedding-workers",
            "1",
        ],
        OBSERVER_ENV,
        &observer,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let paused = wait_for_state(Duration::from_secs(10), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered").exists().then_some(()))
    })
    .expect("worker pause observation cannot fail");
    assert_eq!(
        paused,
        Some(()),
        "background worker did not reach its batch boundary"
    );
    let before = policy(&machine.data_root);
    assert_eq!(before["embedding_workers"], 1);

    let forgotten = machine.run(
        &project,
        &[
            "forget",
            "--session",
            "remove-me",
            "--context",
            "forget-live-cap",
            "--yes",
        ],
    );
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    assert_eq!(
        str_at(&forgotten.json(), "/search/worker"),
        "already_running"
    );
    let after = policy(&machine.data_root);
    assert_eq!(
        after["publication"], before["publication"],
        "forget replaced the live worker choice"
    );
    assert_eq!(
        after["embedding_workers"], before["embedding_workers"],
        "forget replaced the live worker cap"
    );
    assert!(
        after["pending_background_request"]["token"].is_string(),
        "forget did not leave a background handoff request"
    );

    event_support::write_test_file(observer.join("release"), b"continue\n")
        .expect("release worker");
    terminate_observed_worker(&observer);
    machine.wait_for_search_worker_exit();
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn fresh_forget_worker_uses_config_not_an_old_one_off_import_cap() {
    let machine = Machine::with_nomic();
    let project = machine.project("forget-fresh-cap");
    write_session(
        &machine.history_root(),
        &project,
        "remove-me",
        &[("old one-off cap", "must not survive a fresh forget worker")],
    );
    write_session(
        &machine.history_root(),
        &project,
        "keep-me",
        &[("configured cap", "must serve the remaining session")],
    );
    let config_dir = machine.data_root.join("cg");
    std::fs::create_dir_all(&config_dir).expect("create config directory");
    event_support::write_test_file(
        config_dir.join("fabric.toml"),
        "[search]\nembedding_workers = 2\n",
    )
    .expect("write configured cap");
    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "forget-fresh-cap",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(str_at(&imported.json(), "/search/worker"), "foreground");
    assert_eq!(policy(&machine.data_root)["embedding_workers"], 1);
    assert!(policy(&machine.data_root)["pending_background_request"].is_null());

    let observer = observer(&machine, "forget-fresh-observer", false);
    let forgotten = machine.run_with_env(
        &project,
        &[
            "forget",
            "--session",
            "remove-me",
            "--context",
            "forget-fresh-cap",
            "--yes",
        ],
        OBSERVER_ENV,
        &observer,
    );
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    assert_eq!(str_at(&forgotten.json(), "/search/worker"), "spawned");
    let fresh_request = wait_for_state(Duration::from_secs(10), Duration::from_millis(20), || {
        Ok::<_, ()>(
            (!observed_request_caps(&observer).is_empty())
                .then(|| observed_request_caps(&observer)),
        )
    })
    .expect("fresh worker request observation cannot fail");
    assert_eq!(fresh_request, Some(vec![2]));
    machine.wait_for_search_worker_exit();
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn foreground_wait_keeps_its_cap_when_another_import_publishes() {
    let machine = Machine::with_nomic();
    let foreground_project = machine.project("foreground-fixed-cap");
    let turns = [(
        "one two three four five six seven eight",
        "same words again",
    )];
    write_session(
        &machine.history_root(),
        &foreground_project,
        "foreground-session",
        &turns,
    );
    let other_project = machine.project("background-publication");
    write_session(
        &machine.history_root(),
        &other_project,
        "background-session",
        &[("publish cap two", "while foreground is at cap one")],
    );
    let config_dir = machine.data_root.join("cg");
    std::fs::create_dir_all(&config_dir).expect("create config directory");
    event_support::write_test_file(
        config_dir.join("fabric.toml"),
        "[search.passages]\ntarget_tokens = 8\nhard_tokens = 64\n",
    )
    .expect("write narrow passages profile");
    let observer = observer(&machine, "foreground-fixed-observer", true);
    let foreground = machine.spawn_with_env(
        &foreground_project,
        &[
            "import",
            ".",
            "--context",
            "foreground-fixed-cap",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
        OBSERVER_ENV,
        &observer,
    );
    let paused = wait_for_state(Duration::from_secs(30), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered").exists().then_some(()))
    })
    .expect("foreground pause observation cannot fail");
    assert_eq!(
        paused,
        Some(()),
        "foreground projection did not reach a batch boundary"
    );

    let published = machine.run(
        &other_project,
        &[
            "import",
            ".",
            "--context",
            "background-publication",
            "--search",
            "background",
            "--embedding-workers",
            "2",
        ],
    );
    assert_eq!(published.code, 0, "{}", published.describe());
    assert_eq!(
        str_at(&published.json(), "/search/worker"),
        "already_running"
    );
    assert_eq!(policy(&machine.data_root)["embedding_workers"], 2);

    event_support::write_test_file(observer.join("release"), b"continue\n")
        .expect("release foreground");
    let output = foreground
        .wait_with_output()
        .expect("wait for foreground cg");
    assert!(output.status.success(), "foreground failed: {:?}", output);
    let caps = observed_batch_caps(&observer);
    assert!(
        !caps.is_empty(),
        "foreground did not reach its production batch: {caps:?}"
    );
    assert!(
        caps.iter().all(|cap| *cap == 1),
        "foreground adopted another import cap: {caps:?}"
    );

    let ready = machine.wait_for_search_ready(&other_project, "background-publication");
    assert_eq!(
        ready.pointer("/history/search/pending_events"),
        Some(&serde_json::json!(0))
    );
    assert_eq!(
        ready.pointer("/history/search/covered_events"),
        Some(&serde_json::json!(2))
    );
    let found = machine.run(&other_project, &["find", "publish cap two"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    let passage_id = str_at(&found.json(), "/hits/0/passage_id").to_owned();
    let shown = machine.run(&other_project, &["show", &passage_id]);
    assert_eq!(shown.code, 0, "{}", shown.describe());
    assert_eq!(str_at(&shown.json(), "/passage/passage_id"), passage_id);
    machine.wait_for_search_worker_exit();
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn queued_import_selection_survives_forget_and_foreground_handoff() {
    let machine = Machine::with_nomic();
    let config_dir = machine.data_root.join("cg");
    std::fs::create_dir_all(&config_dir).expect("create config directory");
    event_support::write_test_file(
        config_dir.join("fabric.toml"),
        "[search]\nembedding_workers = 2\n",
    )
    .expect("write configured cap");

    let forget_project = machine.project("queued-import-forget");
    write_session(
        &machine.history_root(),
        &forget_project,
        "remove-me",
        &[("erase before queued rebuild", "remove this session")],
    );
    write_session(
        &machine.history_root(),
        &forget_project,
        "keep-me",
        &[("queued rebuild survives", "with the normal import cap")],
    );
    let seeded = machine.run(
        &forget_project,
        &[
            "import",
            ".",
            "--context",
            "queued-import-forget",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(seeded.code, 0, "{}", seeded.describe());

    let foreground_project = machine.project("queued-import-foreground");
    write_session(
        &machine.history_root(),
        &foreground_project,
        "foreground-session",
        &[("foreground remains fixed", "at cap one")],
    );
    let observer = observer(&machine, "queued-import-observer", true);
    let foreground = machine.spawn_with_env(
        &foreground_project,
        &[
            "import",
            ".",
            "--context",
            "queued-import-foreground",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
        OBSERVER_ENV,
        &observer,
    );
    let paused = wait_for_state(Duration::from_secs(10), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered").exists().then_some(()))
    })
    .expect("foreground pause observation cannot fail");
    assert_eq!(paused, Some(()), "foreground did not reach its batch");

    let import_project = machine.project("queued-normal-import");
    write_session(
        &machine.history_root(),
        &import_project,
        "normal-background-session",
        &[(
            "normal import queued first",
            "its cap three must survive forget",
        )],
    );
    let imported = machine.run(
        &import_project,
        &[
            "import",
            ".",
            "--context",
            "queued-normal-import",
            "--search",
            "background",
            "--embedding-workers",
            "3",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(
        str_at(&imported.json(), "/search/worker"),
        "already_running"
    );

    let forgotten = machine.run(
        &forget_project,
        &[
            "forget",
            "--session",
            "remove-me",
            "--context",
            "queued-import-forget",
            "--yes",
        ],
    );
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    assert_eq!(
        str_at(&forgotten.json(), "/search/worker"),
        "already_running"
    );
    assert_eq!(
        policy(&machine.data_root)["pending_background_request"]["embedding_workers"],
        3
    );

    event_support::write_test_file(observer.join("release"), b"continue\n")
        .expect("release foreground");
    let output = foreground
        .wait_with_output()
        .expect("wait for foreground cg");
    assert!(output.status.success(), "foreground failed: {output:?}");
    machine.wait_for_search_ready(&import_project, "queued-normal-import");
    machine.wait_for_search_ready(&forget_project, "queued-import-forget");
    machine.wait_for_search_worker_exit();

    let caps = observed_batch_caps(&observer);
    assert_eq!(caps.first(), Some(&1), "foreground cap changed: {caps:?}");
    assert!(
        caps.len() >= 3 && caps[1..].iter().all(|cap| *cap == 3),
        "forget replaced the queued normal import's effective cap: {caps:?}"
    );
    for (project, query) in [
        (&import_project, "normal import queued first"),
        (&forget_project, "queued rebuild survives"),
    ] {
        let found = machine.run(project, &["find", query]);
        assert_eq!(found.code, 0, "{}", found.describe());
        assert!(!found.json()["hits"].as_array().unwrap().is_empty());
    }
}

// SIGINT remains owned through the final handoff/status lifetime. The first rendezvous queues real
// work while foreground owns the worker turn; the second fires after that request has been handed
// off, proving Ctrl-C still exits promptly without abandoning or duplicating the queued service.
#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn sigint_foreground_hands_the_queued_request_to_one_background_worker() {
    let machine = Machine::with_nomic();
    let foreground_project = machine.project("sigint-foreground");
    write_session(
        &machine.history_root(),
        &foreground_project,
        "foreground-session",
        &[(
            "hold foreground work",
            "until SIGINT releases its worker turn",
        )],
    );
    let observer = observer(&machine, "sigint-observer", true);
    event_support::write_test_file(
        observer.join("pause-after-foreground-handoff"),
        b"pause before final status refresh\n",
    )
    .expect("arm final foreground handoff rendezvous");
    let mut foreground = machine.spawn_with_env(
        &foreground_project,
        &[
            "import",
            ".",
            "--context",
            "sigint-foreground",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
        OBSERVER_ENV,
        &observer,
    );
    let paused = wait_for_state(Duration::from_secs(20), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered").exists().then_some(()))
    })
    .expect("foreground observation cannot fail");
    assert_eq!(paused, Some(()), "foreground did not hold its worker turn");

    let queued_project = machine.project("sigint-queued");
    write_session(
        &machine.history_root(),
        &queued_project,
        "queued-session",
        &[(
            "the queued request",
            "must become searchable without another import",
        )],
    );
    let queued = machine.run(
        &queued_project,
        &[
            "import",
            ".",
            "--context",
            "sigint-queued",
            "--embedding-workers",
            "2",
        ],
    );
    assert_eq!(queued.code, 0, "{}", queued.describe());
    assert_eq!(str_at(&queued.json(), "/search/worker"), "already_running");

    event_support::write_test_file(
        observer.join("release"),
        b"release foreground worker turn\n",
    )
    .expect("release foreground into queued handoff");
    let handed_off = wait_for_state(Duration::from_secs(30), Duration::from_millis(20), || {
        Ok::<_, ()>(
            observer
                .join("foreground-handoff-complete")
                .exists()
                .then_some(()),
        )
    })
    .expect("final handoff observation cannot fail");
    assert_eq!(
        handed_off,
        Some(()),
        "foreground never reached its final status refresh"
    );

    // Tokio's installed process handler remains active here even though foreground projection and
    // queued handoff are complete. This signal must still own the command and return 130 promptly.
    assert_eq!(
        unsafe { libc::kill(foreground.id() as i32, libc::SIGINT) },
        0
    );
    let exited = foreground.wait().ok();
    assert_eq!(
        exited.and_then(|status| status.code()),
        Some(130),
        "SIGINT foreground must exit through the interrupt status"
    );

    let ready = machine.wait_for_search_ready(&queued_project, "sigint-queued");
    assert_eq!(
        ready.pointer("/history/search/pending_events"),
        Some(&serde_json::json!(0))
    );
    machine.wait_for_search_worker_exit();
    let no_retrigger = wait_for_state(Duration::from_secs(10), Duration::from_millis(20), || {
        Ok::<_, ()>(
            policy(&machine.data_root)["pending_background_request"]
                .is_null()
                .then_some(()),
        )
    })
    .expect("consumed request observation cannot fail");
    assert_eq!(
        no_retrigger,
        Some(()),
        "the consumed queued request must not retrigger a second worker"
    );
    // The worker releases its long lock before compare-and-clear. Observe the durable queue
    // acknowledgement above before reading the final policy; lock release alone is too early.
    let delivered = policy(&machine.data_root);
    assert_eq!(
        delivered["pending_background_request"],
        serde_json::Value::Null
    );
}

// This reaches the Nomic prepared-document forward-call boundary. The foreground process
// must exit promptly while that call is
// still held. Its queued child inherits the private test environment, so the fixture releases it
// only after the foreground exit and then proves the queued project becomes ready.
#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn sigint_during_prepared_embedding_hands_off_the_queued_request_once() {
    const PREPARED_EMBEDDING_BARRIER_ENV: &str = "CG_NOMIC_TEST_PREPARED_EMBEDDING_BARRIER";

    let machine = Machine::with_nomic();
    let foreground_project = machine.project("sigint-prepared-foreground");
    write_session(
        &machine.history_root(),
        &foreground_project,
        "foreground-session",
        &[(
            "hold the prepared embedding",
            "until SIGINT exits the foreground",
        )],
    );
    let barrier = machine.root.join("sigint-prepared-embedding");
    std::fs::create_dir_all(&barrier).expect("create prepared embedding barrier");
    let mut foreground = machine.spawn_with_env(
        &foreground_project,
        &[
            "import",
            ".",
            "--context",
            "sigint-prepared-foreground",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
        PREPARED_EMBEDDING_BARRIER_ENV,
        &barrier,
    );
    let reached = wait_for_state(Duration::from_secs(60), Duration::from_millis(20), || {
        Ok::<_, ()>(barrier.join("reached").exists().then_some(()))
    })
    .expect("prepared embedding observation cannot fail");
    assert_eq!(
        reached,
        Some(()),
        "foreground never reached Nomic forward admission"
    );

    let queued_project = machine.project("sigint-prepared-queued");
    write_session(
        &machine.history_root(),
        &queued_project,
        "queued-session",
        &[(
            "the queued request",
            "must become ready after foreground SIGINT",
        )],
    );
    let queued = machine.run(
        &queued_project,
        &[
            "import",
            ".",
            "--context",
            "sigint-prepared-queued",
            "--embedding-workers",
            "2",
        ],
    );
    assert_eq!(queued.code, 0, "{}", queued.describe());
    assert_eq!(str_at(&queued.json(), "/search/worker"), "already_running");

    assert_eq!(
        unsafe { libc::kill(foreground.id() as i32, libc::SIGINT) },
        0
    );
    let exited = foreground.wait().ok();
    assert_eq!(
        exited.and_then(|status| status.code()),
        Some(130),
        "SIGINT must exit the foreground while prepared embedding remains blocked"
    );

    // The handoff child inherits this test-only barrier. Releasing only after exit proves that the
    // foreground did not need the forward call to finish before it freed the worker turn.
    event_support::write_test_file(barrier.join("release"), b"release handed-off child\n")
        .expect("release inherited prepared embedding barrier");
    let ready = machine.wait_for_search_ready(&queued_project, "sigint-prepared-queued");
    assert_eq!(
        ready.pointer("/history/search/pending_events"),
        Some(&serde_json::json!(0))
    );
    machine.wait_for_search_worker_exit();
    assert!(
        policy(&machine.data_root)["pending_background_request"].is_null(),
        "the delivered request must not leave a second worker trigger"
    );
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn sigint_foreground_without_a_queue_exits_and_releases_its_worker_turn() {
    let machine = Machine::with_nomic();
    let project = machine.project("sigint-no-queue");
    write_session(
        &machine.history_root(),
        &project,
        "foreground-session",
        &[(
            "hold foreground work",
            "until SIGINT releases its worker turn",
        )],
    );
    let observer = observer(&machine, "sigint-no-queue-observer", true);
    let mut foreground = machine.spawn_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "sigint-no-queue",
            "--search",
            "wait",
            "--embedding-workers",
            "1",
        ],
        OBSERVER_ENV,
        &observer,
    );
    let paused = wait_for_state(Duration::from_secs(20), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered").exists().then_some(()))
    })
    .expect("foreground observation cannot fail");
    assert_eq!(paused, Some(()), "foreground did not hold its worker turn");

    assert_eq!(
        unsafe { libc::kill(foreground.id() as i32, libc::SIGINT) },
        0
    );
    let exited = foreground.wait().ok();
    assert_eq!(exited.and_then(|status| status.code()), Some(130));
    machine.wait_for_search_worker_exit();
    assert!(policy(&machine.data_root)["pending_background_request"].is_null());
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn newer_wake_survives_older_final_clear_and_is_serviced() {
    let machine = Machine::with_nomic();
    let project = machine.project("newer-wake-final-clear");
    write_session(
        &machine.history_root(),
        &project,
        "first-session",
        &[("older worker pass", "finishes before the newer wake")],
    );
    let observer = machine.root.join("newer-wake-observer");
    std::fs::create_dir_all(&observer).expect("create worker observer directory");
    event_support::write_test_file(observer.join("numbered-passes"), b"2\n")
        .expect("pause two completed worker passes");
    let first = machine.run_with_env(
        &project,
        &[
            "import",
            ".",
            "--context",
            "newer-wake-final-clear",
            "--search",
            "background",
            "--embedding-workers",
            "1",
        ],
        OBSERVER_ENV,
        &observer,
    );
    assert_eq!(first.code, 0, "{}", first.describe());
    let first_pass = wait_for_state(Duration::from_secs(30), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered.1").exists().then_some(()))
    })
    .expect("first completed-pass observation cannot fail");
    assert_eq!(first_pass, Some(()), "older worker did not finish its pass");

    write_session(
        &machine.history_root(),
        &project,
        "second-session",
        &[("newer wake history", "must survive the older final clear")],
    );
    let second = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "newer-wake-final-clear",
            "--search",
            "background",
            "--embedding-workers",
            "2",
        ],
    );
    assert_eq!(second.code, 0, "{}", second.describe());
    assert_eq!(str_at(&second.json(), "/search/worker"), "already_running");
    let newer_token = policy(&machine.data_root)["pending_background_request"]["token"]
        .as_str()
        .expect("newer request token")
        .to_owned();

    event_support::write_test_file(observer.join("release.1"), b"allow older clear\n")
        .expect("release older pass");
    let second_pass = wait_for_state(Duration::from_secs(30), Duration::from_millis(20), || {
        Ok::<_, ()>(observer.join("entered.2").exists().then_some(()))
    })
    .expect("second completed-pass observation cannot fail");
    let token_after_older_clear = policy(&machine.data_root)["pending_background_request"]["token"]
        .as_str()
        .map(str::to_owned);
    event_support::write_test_file(observer.join("release.2"), b"allow newer clear\n")
        .expect("release newer pass");
    machine.wait_for_search_ready(&project, "newer-wake-final-clear");
    machine.wait_for_search_worker_exit();

    assert_eq!(second_pass, Some(()), "newer request was not serviced");
    assert_eq!(
        token_after_older_clear.as_deref(),
        Some(newer_token.as_str())
    );
    assert!(policy(&machine.data_root)["pending_background_request"].is_null());
    let found = machine.run(&project, &["find", "newer wake history"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    assert!(!found.json()["hits"].as_array().unwrap().is_empty());
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn exhausted_writer_refusal_keeps_the_request_until_a_new_same_profile_wake_reaches_ready() {
    let machine = Machine::with_nomic();
    let project = machine.project("writer-refusal-revival");
    write_session(
        &machine.history_root(),
        &project,
        "writer-refusal-session",
        &[(
            "a writer blocks the first projection",
            "the next request finishes the same profile",
        )],
    );

    std::fs::create_dir_all(&machine.data_root).expect("create data root for worker lock");
    let held_worker_lock =
        cg_node_fs::acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold the worker turn while import publishes its request");
    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "writer-refusal-revival",
            "--search",
            "background",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(
        str_at(&imported.json(), "/search/worker"),
        "already_running"
    );
    let first_token = policy(&machine.data_root)["pending_background_request"]["token"]
        .as_str()
        .expect("first pending request token")
        .to_owned();
    drop(held_worker_lock);

    let barrier = machine.root.join("writer-refusal-barrier");
    std::fs::create_dir_all(&barrier).expect("create writer-refusal barrier");
    event_support::write_test_file(barrier.join("pause-after-work-assessment"), b"pause\n")
        .expect("arm post-assessment rendezvous");
    event_support::write_test_file(barrier.join("exhaust-store-open-retry"), b"exhaust\n")
        .expect("use the core's test-only allowance override");
    let envelope = worker_envelope();
    let mut worker = machine.spawn_with_env(
        &project,
        &["__event-search-worker", &envelope],
        PRE_CATCH_UP_ENV,
        &barrier,
    );
    let assessed = wait_for_state(Duration::from_secs(20), Duration::from_millis(20), || {
        Ok::<_, ()>(
            barrier
                .join("work-assessment-complete")
                .exists()
                .then_some(()),
        )
    })
    .expect("worker assessment observation cannot fail");
    assert_eq!(assessed, Some(()), "worker did not assess pending work");

    let runtime = machine.root.join("writer-runtime");
    std::fs::create_dir_all(&runtime).expect("create writer runtime directory");
    let mut writer =
        HoldingChild::hold_as_writer(&machine.db_path(), &runtime, "worker_writer_holder_child");
    writer.wait_until_holding_as_writer();
    event_support::write_test_file(
        barrier.join("work-assessment-release"),
        b"attempt projection\n",
    )
    .expect("release projection into writer contention");
    let exited = worker.wait().ok();
    if exited.is_none() {
        let _ = worker.kill();
        let _ = worker.wait();
    }
    assert_eq!(
        exited.and_then(|status| status.code()),
        Some(1),
        "an unrecordable writer refusal must fail the worker"
    );
    assert_eq!(
        policy(&machine.data_root)["pending_background_request"]["token"],
        first_token,
        "the failed worker must retain the pending request"
    );
    writer.release();

    let held_worker_lock =
        cg_node_fs::acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .expect("hold the worker turn while the new request is observed");
    let retried = machine.run(
        &project,
        &[
            "import",
            ".",
            "--search",
            "background",
            "--embedding-workers",
            "1",
        ],
    );
    assert_eq!(retried.code, 0, "{}", retried.describe());
    let second_token = policy(&machine.data_root)["pending_background_request"]["token"]
        .as_str()
        .expect("replacement request token")
        .to_owned();
    assert_ne!(second_token, first_token, "a new wake gets a new token");
    drop(held_worker_lock);

    let completed = machine.run(&project, &["__event-search-worker", &envelope]);
    assert_eq!(completed.code, 0, "{}", completed.describe());
    let status = machine.run(&project, &["status", "--context", "writer-refusal-revival"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(str_at(&status, "/history/search/status"), "ready");
    assert!(status["history"]["search"]["last_failure"].is_null());
    assert_eq!(
        policy(&machine.data_root)["pending_background_request"],
        serde_json::Value::Null
    );
}
