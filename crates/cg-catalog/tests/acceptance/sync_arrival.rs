use cg_catalog::{CG_CATALOG_CONTEXT_ID, CgCatalogError, FetchResult, OutputMode};
use context_graph::Observation;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

use crate::common::{
    CANONICAL_LICENSE_GATE_MESSAGE, GEMMA_ID, NOMIC_BYTES, NOMIC_ID, NOMIC_REVISION, catalog_at,
    fixture_dir, hold_model_lock, separate_store, stub_base_url, write_nomic_cache,
    write_source_override, write_valid_mini_gemma_acceptance,
};

fn comparable_observation(observation: &Observation) -> serde_json::Value {
    let mut comparable = serde_json::json!({
        "observation_type": observation.observation_type,
        "evidence": observation.evidence,
        "observed_properties": observation.observed_properties,
        "state_delta": observation.state_delta,
        "properties": observation.properties,
        "source": observation.source,
        "entity_id": observation.entity_id,
        "context_id": observation.context_id,
    });
    if let Some(evidence_items) = comparable
        .get_mut("evidence")
        .and_then(serde_json::Value::as_array_mut)
    {
        for evidence in evidence_items {
            if let Some(evidence) = evidence.as_object_mut() {
                evidence.remove("id");
                evidence.remove("observation_id");
            }
        }
    }
    comparable
}

#[derive(Clone)]
struct ChannelLogWriter {
    sender: mpsc::Sender<String>,
}

struct ChannelLogLine {
    sender: mpsc::Sender<String>,
    buffer: Vec<u8>,
}

impl<'writer> tracing_subscriber::fmt::MakeWriter<'writer> for ChannelLogWriter {
    type Writer = ChannelLogLine;

    fn make_writer(&'writer self) -> Self::Writer {
        ChannelLogLine {
            sender: self.sender.clone(),
            buffer: Vec::new(),
        }
    }
}

impl ChannelLogLine {
    fn send_complete_lines(&mut self) {
        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = String::from_utf8_lossy(&self.buffer[..position]).to_string();
            self.buffer.drain(..=position);
            if !line.is_empty() {
                let _ = self.sender.send(line);
            }
        }
    }
}

impl Write for ChannelLogLine {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(bytes);
        self.send_complete_lines();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.send_complete_lines();
        Ok(())
    }
}

impl Drop for ChannelLogLine {
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&self.buffer).to_string();
            if !line.is_empty() {
                let _ = self.sender.send(line);
            }
        }
    }
}

fn sync_fetch_child_if_requested() -> bool {
    let Some(root) = std::env::var_os("CG_CATALOG_SYNC_FETCH_ROOT") else {
        return false;
    };
    if std::env::var_os("CG_CATALOG_SYNC_FETCH_LOG_CHILD").is_some() {
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_ansi(false)
            .with_writer(std::io::stdout)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("test tracing subscriber");
    }
    let model_id = std::env::var("CG_CATALOG_SYNC_FETCH_MODEL").expect("model env");
    let root = std::path::PathBuf::from(root);
    let (_store_temp, store) = separate_store();
    let mut catalog = catalog_at(root, "host-A");
    if let Some(progress_file) = std::env::var_os("CG_CATALOG_SYNC_FETCH_PROGRESS_FILE") {
        let progress_file = std::path::PathBuf::from(progress_file);
        catalog = catalog.with_progress_observer(move |_| {
            std::fs::write(&progress_file, b"fetch-starting")
                .expect("sync fetch progress file written");
        });
    }
    let result = catalog.resolve_for_sync_arrival(&store, model_id, OutputMode::Headless);
    match result {
        Ok(result) => {
            if let Ok(expected) = std::env::var("CG_CATALOG_SYNC_EXPECT_RESULT") {
                assert_eq!(format!("{result:?}"), expected);
            }
        }
        Err(err) => {
            eprintln!("{err:?}");
            std::process::exit(1);
        }
    }
    true
}

fn spawn_sync_fetch_child(
    test_name: &str,
    root: &std::path::Path,
    model_id: &str,
    expected_result: &str,
) -> Child {
    spawn_sync_fetch_child_inner(test_name, root, model_id, expected_result, None)
}

fn spawn_sync_fetch_child_with_progress_file(
    test_name: &str,
    root: &std::path::Path,
    model_id: &str,
    expected_result: &str,
    progress_file: &std::path::Path,
) -> Child {
    spawn_sync_fetch_child_inner(
        test_name,
        root,
        model_id,
        expected_result,
        Some(progress_file),
    )
}

fn spawn_sync_fetch_child_inner(
    test_name: &str,
    root: &std::path::Path,
    model_id: &str,
    expected_result: &str,
    progress_file: Option<&std::path::Path>,
) -> Child {
    let mut command = Command::new(std::env::current_exe().expect("current test binary"));
    command
        .arg("--exact")
        .arg(test_name)
        .env("CG_CATALOG_SYNC_FETCH_ROOT", root)
        .env("CG_CATALOG_SYNC_FETCH_MODEL", model_id)
        .env("CG_CATALOG_SYNC_EXPECT_RESULT", expected_result);
    if let Some(progress_file) = progress_file {
        command.env("CG_CATALOG_SYNC_FETCH_PROGRESS_FILE", progress_file);
    }
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("sync fetch child starts")
}

fn assert_child_success(child: Child, label: &str) {
    let output = child
        .wait_with_output()
        .unwrap_or_else(|err| panic!("{label} output collected: {err:?}"));
    assert!(
        output.status.success(),
        "{label} exited unsuccessfully\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn wait_for_child_progress_file(child: &mut Child, progress_file: &std::path::Path, label: &str) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while !progress_file.exists() {
        if let Some(status) = child.try_wait().expect("child status readable") {
            panic!("{label} exited before reaching fetch progress path: {status}");
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{label} did not reach fetch progress path before timeout"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn respond_with_nomic_fixture(request: tiny_http::Request) {
    let url = request.url().to_string();
    let filename = ["tokenizer.json", "config.json", "REVISION"]
        .into_iter()
        .find(|candidate| url.ends_with(candidate))
        .unwrap_or("model.safetensors");
    let bytes = std::fs::read(fixture_dir("nomic").join(filename)).expect("fixture bytes");
    let _ = request.respond(tiny_http::Response::from_data(bytes));
}

struct FirstResponseGatedNomicServer {
    first_request_rx: mpsc::Receiver<()>,
    release_tx: mpsc::Sender<()>,
    shutdown: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
}

impl FirstResponseGatedNomicServer {
    fn stop_and_join(&mut self) -> std::thread::Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.release_tx.send(());
        match self.server_thread.take() {
            Some(server_thread) => server_thread.join(),
            None => Ok(()),
        }
    }

    fn shutdown_and_join(mut self) {
        self.stop_and_join().expect("gated source server joins");
    }
}

impl Drop for FirstResponseGatedNomicServer {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn start_first_response_gated_nomic_server(
    root: &std::path::Path,
) -> FirstResponseGatedNomicServer {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(root, NOMIC_ID, &base_url, NOMIC_REVISION);
    let (first_request_tx, first_request_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let server_thread = std::thread::spawn(move || {
        let mut first_release = Some(release_rx);
        let mut responder_threads = Vec::new();
        while !server_shutdown.load(Ordering::SeqCst) {
            match server.recv_timeout(Duration::from_millis(50)) {
                Ok(Some(request)) => {
                    if let Some(release_rx) = first_release.take() {
                        let first_request_tx = first_request_tx.clone();
                        responder_threads.push(std::thread::spawn(move || {
                            first_request_tx.send(()).expect("first request signalled");
                            release_rx
                                .recv_timeout(Duration::from_secs(30))
                                .expect("first response release signal");
                            respond_with_nomic_fixture(request);
                        }));
                    } else {
                        responder_threads.push(std::thread::spawn(move || {
                            respond_with_nomic_fixture(request)
                        }));
                    }
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
        for responder_thread in responder_threads {
            responder_thread
                .join()
                .expect("gated source responder joins");
        }
    });
    FirstResponseGatedNomicServer {
        first_request_rx,
        release_tx,
        shutdown,
        server_thread: Some(server_thread),
    }
}

struct CountedNomicSourceOverride {
    request_count: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    server_thread: std::thread::JoinHandle<()>,
}

impl CountedNomicSourceOverride {
    fn shutdown_and_join(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.server_thread
            .join()
            .expect("counted source server joins");
    }
}

fn start_counted_nomic_source_override(roots: &[&std::path::Path]) -> CountedNomicSourceOverride {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    for root in roots {
        write_source_override(root, NOMIC_ID, &base_url, NOMIC_REVISION);
    }
    let request_count = Arc::new(AtomicUsize::new(0));
    let shutdown = Arc::new(AtomicBool::new(false));
    let counter = Arc::clone(&request_count);
    let server_shutdown = Arc::clone(&shutdown);
    let server_thread = std::thread::spawn(move || {
        while !server_shutdown.load(Ordering::SeqCst) {
            match server.recv_timeout(Duration::from_millis(50)) {
                Ok(Some(request)) => {
                    counter.fetch_add(1, Ordering::SeqCst);
                    respond_with_nomic_fixture(request);
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    });
    CountedNomicSourceOverride {
        request_count,
        shutdown,
        server_thread,
    }
}

fn assert_headless_progress_payload(line: &str, target_path: &std::path::Path) {
    let event = serde_json::from_str::<serde_json::Value>(line);
    assert!(
        event.is_ok(),
        "headless progress line should be structured JSON: line={line:?}, error={:?}",
        event.as_ref().err()
    );
    let event = event.expect("asserted structured log JSON");
    let fields = event.get("fields");
    assert!(
        fields.is_some_and(serde_json::Value::is_object),
        "headless progress must be emitted by tracing subscriber fields object: {event:?}"
    );
    let fields = fields.expect("asserted tracing fields object");
    assert_eq!(
        fields.get("model_id").and_then(serde_json::Value::as_str),
        Some(NOMIC_ID)
    );
    let source_kind = fields
        .get("source.kind")
        .and_then(serde_json::Value::as_str);
    let source_repo = fields
        .get("source.repo")
        .and_then(serde_json::Value::as_str);
    let revision = fields
        .get("source.revision")
        .and_then(serde_json::Value::as_str);
    assert_eq!(source_kind, Some("HuggingFace"));
    assert_eq!(source_repo, Some("nomic-ai/nomic-embed-text-v1.5"));
    assert_eq!(revision, Some(NOMIC_REVISION));
    assert_eq!(
        fields
            .get("expected_bytes")
            .and_then(serde_json::Value::as_u64),
        Some(NOMIC_BYTES)
    );
    assert_eq!(
        fields
            .get("target_path")
            .and_then(serde_json::Value::as_str)
            .map(std::path::PathBuf::from),
        Some(target_path.to_path_buf())
    );
}

#[test]
fn sync_arrival_headless_mode_produces_observation_identical_to_interactive_mode_observation_single_producer_lockfile_serialises_writers()
 {
    // Interactive CLI fetches and headless sync arrivals should share one observation producer.
    if sync_fetch_child_if_requested() {
        return;
    }
    let interactive_temp = tempfile::tempdir().expect("interactive tempdir");
    let headless_temp = tempfile::tempdir().expect("headless tempdir");
    let source_server =
        start_counted_nomic_source_override(&[interactive_temp.path(), headless_temp.path()]);
    let request_count = Arc::clone(&source_server.request_count);
    let (_interactive_store_temp, interactive_store) = separate_store();
    let interactive = catalog_at(interactive_temp.path(), "host-A")
        .fetch(Some(&interactive_store), NOMIC_ID, OutputMode::Interactive)
        .expect("interactive fetch returns");
    let (_headless_store_temp, headless_store) = separate_store();
    let headless = catalog_at(headless_temp.path(), "host-A")
        .resolve_for_sync_arrival(&headless_store, NOMIC_ID, OutputMode::Headless)
        .expect("headless resolve returns");
    source_server.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) >= 2,
        "interactive and headless cold paths should both use the configured local source override"
    );
    assert_eq!(interactive, FetchResult::Completed);
    assert_eq!(headless, FetchResult::Completed);
    let interactive_rows = interactive_store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("interactive observations readable");
    let headless_rows = headless_store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("headless observations readable");
    assert_eq!(interactive_rows.len(), 1);
    assert_eq!(headless_rows.len(), 1);
    assert_eq!(
        interactive_rows[0]
            .properties
            .get("output_mode")
            .and_then(serde_json::Value::as_str),
        Some("interactive")
    );
    assert_eq!(
        headless_rows[0]
            .properties
            .get("output_mode")
            .and_then(serde_json::Value::as_str),
        Some("headless")
    );
    let mut interactive_comparable = comparable_observation(&interactive_rows[0]);
    let mut headless_comparable = comparable_observation(&headless_rows[0]);
    for comparable in [&mut interactive_comparable, &mut headless_comparable] {
        comparable
            .get_mut("properties")
            .and_then(serde_json::Value::as_object_mut)
            .expect("properties object")
            .remove("output_mode");
    }
    assert_eq!(
        interactive_comparable, headless_comparable,
        "interactive and headless rows should be byte-identical after excluding ids/timestamps/txids"
    );

    let lock_temp = tempfile::tempdir().expect("lock tempdir");
    let gated_server = start_first_response_gated_nomic_server(lock_temp.path());
    let winner = spawn_sync_fetch_child(
        "sync_arrival::sync_arrival_headless_mode_produces_observation_identical_to_interactive_mode_observation_single_producer_lockfile_serialises_writers",
        lock_temp.path(),
        NOMIC_ID,
        "Completed",
    );
    assert!(
        gated_server
            .first_request_rx
            .recv_timeout(Duration::from_secs(30))
            .is_ok(),
        "first headless resolver should reach the configured source before publishing artifacts"
    );
    let blocked_progress_file = lock_temp.path().join(".blocked-sync-fetch-started");
    let mut blocked = spawn_sync_fetch_child_with_progress_file(
        "sync_arrival::sync_arrival_headless_mode_produces_observation_identical_to_interactive_mode_observation_single_producer_lockfile_serialises_writers",
        lock_temp.path(),
        NOMIC_ID,
        "AlreadyCachedByConcurrentWinner",
        &blocked_progress_file,
    );
    wait_for_child_progress_file(&mut blocked, &blocked_progress_file, "blocked sync child");
    assert!(
        blocked.try_wait().expect("child status readable").is_none(),
        "headless sync-arrival should block on the same per-model lockfile as interactive fetch"
    );
    gated_server
        .release_tx
        .send(())
        .expect("release first source response");
    assert_child_success(winner, "winner sync child");
    assert_child_success(blocked, "blocked sync child");
    gated_server.shutdown_and_join();
}

#[test]
fn sync_arrival_headless_for_license_gated_model_defers_at_receiver_emits_observation_does_not_fetch()
 {
    // A receiver should defer a gated model instead of fetching bytes in headless mode.
    let temp = tempfile::tempdir().expect("tempdir");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(
        temp.path(),
        GEMMA_ID,
        &base_url,
        "embeddinggemma-fixture-revision",
    );
    let outbound_request_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&outbound_request_count);
    let server_thread = std::thread::spawn(move || {
        if let Ok(Some(request)) = server.recv_timeout(Duration::from_millis(250)) {
            counter.fetch_add(1, Ordering::SeqCst);
            let _ = request.respond(tiny_http::Response::from_string("unexpected gated fetch"));
        }
    });
    let (_store_temp, store) = separate_store();
    let result = catalog_at(temp.path(), "host-A").resolve_for_sync_arrival(
        &store,
        GEMMA_ID,
        OutputMode::Headless,
    );
    assert!(matches!(
        result,
        Err(CgCatalogError::ManualAcceptanceRequiredAtReceiver {
            canonical_message,
            ..
        }) if canonical_message == CANONICAL_LICENSE_GATE_MESSAGE
    ));
    let rows = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    assert!(
        rows.iter().any(|observation| {
            observation.observation_type == "license_gate_deferred"
                && observation.state_delta.get("model_id")
                    == Some(&Some(serde_json::json!(GEMMA_ID)))
                && observation.state_delta.get("canonical_message")
                    == Some(&Some(serde_json::json!(CANONICAL_LICENSE_GATE_MESSAGE)))
        }),
        "sync-arrival deferral must emit a license_gate_deferred Observation naming the model and canonical message"
    );
    server_thread.join().expect("server thread joins");
    assert_eq!(
        outbound_request_count.load(Ordering::SeqCst),
        0,
        "license-gated sync arrival should defer before any outbound request"
    );
    let model_dir = temp.path().join(GEMMA_ID);
    assert!(
        !model_dir.join("model.safetensors").exists() && !model_dir.join("catalog.toml").exists(),
        "license-gated deferral must not leave fetched artifacts or a cache manifest"
    );
}

#[test]
fn sync_arrival_headless_for_apache_model_on_pre_populated_cache_returns_already_cached_with_zero_network_and_zero_new_observations()
 {
    // Warm-cache sync arrivals should return without outbound network or no-op observations.
    let temp = tempfile::tempdir().expect("tempdir");
    write_nomic_cache(temp.path());
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let outbound_request_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&outbound_request_count);
    let server_thread = std::thread::spawn(move || {
        if let Ok(Some(request)) = server.recv_timeout(Duration::from_millis(250)) {
            counter.fetch_add(1, Ordering::SeqCst);
            let _ = request.respond(tiny_http::Response::from_string("unexpected warm fetch"));
        }
    });
    let (_store_temp, store) = separate_store();
    let result = catalog_at(temp.path(), "host-A")
        .resolve_for_sync_arrival(&store, NOMIC_ID, OutputMode::Headless)
        .expect("resolve returns");
    assert_eq!(result, FetchResult::AlreadyCached);
    assert_eq!(
        store
            .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
            .expect("observations readable"),
        Vec::new()
    );
    server_thread.join().expect("server thread joins");
    assert_eq!(
        outbound_request_count.load(Ordering::SeqCst),
        0,
        "warm-cache sync arrival should not perform a HEAD or GET"
    );
}

#[test]
fn sync_arrival_after_acceptance_recorded_on_receiver_succeeds_and_makes_rows_readable() {
    // Once the receiver records acceptance, the sync-arrived rows should become readable.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_model_dir, record) = write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let catalog = catalog_at(temp.path(), "host-A");
    let _record = catalog
        .record_acceptance(GEMMA_ID, record)
        .expect("acceptance returns");
    let (_store_temp, store) = separate_store();
    let result = catalog.resolve_for_sync_arrival(&store, GEMMA_ID, OutputMode::Headless);
    assert_eq!(result, Ok(FetchResult::AlreadyCachedAndAccepted));
}

#[test]
fn headless_mode_log_sink_receives_model_source_byte_size_payload_within_two_hundred_fifty_milliseconds()
 {
    // Headless operators should see the same model/source/size payload in logs, not stderr.
    if sync_fetch_child_if_requested() {
        return;
    }
    if std::env::var_os("CG_CATALOG_HEADLESS_STDERR_CHILD").is_some() {
        let root = std::env::var_os("CG_CATALOG_HEADLESS_STDERR_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let temp = tempfile::tempdir().expect("tempdir");
                let root = temp.path().to_path_buf();
                std::mem::forget(temp);
                root
            });
        let (_store_temp, store) = separate_store();
        let _ = catalog_at(root, "host-A").resolve_for_sync_arrival(
            &store,
            NOMIC_ID,
            OutputMode::Headless,
        );
        return;
    }
    let lock_temp = tempfile::tempdir().expect("lock tempdir");
    let log_source_server = start_counted_nomic_source_override(&[lock_temp.path()]);
    let log_request_count = Arc::clone(&log_source_server.request_count);
    let held_lock = hold_model_lock(lock_temp.path(), NOMIC_ID);
    let (line_tx, line_rx) = mpsc::channel();
    let (call_tx, call_rx) = mpsc::channel();
    let writer = ChannelLogWriter { sender: line_tx };
    let root = lock_temp.path().to_path_buf();
    let (thread_store_temp, thread_store) = separate_store();
    let fetch = std::thread::spawn(move || {
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_ansi(false)
            .with_writer(writer)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            let _thread_store_temp = thread_store_temp;
            let catalog = catalog_at(root, "host-A");
            let _ = call_tx.send(());
            catalog.resolve_for_sync_arrival(&thread_store, NOMIC_ID, OutputMode::Headless)
        })
    });
    call_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("fetch thread reaches resolve_for_sync_arrival call");
    let line = line_rx.recv_timeout(Duration::from_millis(250));
    assert!(
        line.is_ok(),
        "headless tracing line should arrive before fetch completion"
    );
    let line = line.expect("asserted headless tracing line is present");
    assert_headless_progress_payload(&line, &lock_temp.path().join(NOMIC_ID));
    assert!(
        !fetch.is_finished(),
        "headless progress must be produced before a lock-blocked fetch can complete"
    );
    drop(held_lock);
    assert_eq!(
        fetch.join().expect("headless fetch thread joins"),
        Ok(FetchResult::Completed)
    );
    log_source_server.shutdown_and_join();
    assert!(
        log_request_count.load(Ordering::SeqCst) > 0,
        "headless log child should complete through the local source override after lock release"
    );

    let temp = tempfile::tempdir().expect("tempdir");
    let source_server = start_counted_nomic_source_override(&[temp.path()]);
    let request_count = Arc::clone(&source_server.request_count);
    let (_store_temp, store) = separate_store();
    let _ = catalog_at(temp.path(), "host-A")
        .resolve_for_sync_arrival(&store, NOMIC_ID, OutputMode::Headless)
        .expect("resolve returns");
    source_server.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "headless observation path should use the configured local source override"
    );
    let rows = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    assert!(rows.iter().any(|observation| {
        observation.properties.get("log_payload")
            == Some(&serde_json::json!({
                "model_id": NOMIC_ID,
                "source": {"kind": "HuggingFace", "repo": "nomic-ai/nomic-embed-text-v1.5"},
                "expected_bytes": NOMIC_BYTES
            }))
    }));
    let stderr_temp = tempfile::tempdir().expect("stderr child tempdir");
    let stderr_source_server = start_counted_nomic_source_override(&[stderr_temp.path()]);
    let stderr_request_count = Arc::clone(&stderr_source_server.request_count);
    let output = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("sync_arrival::headless_mode_log_sink_receives_model_source_byte_size_payload_within_two_hundred_fifty_milliseconds")
        .env("CG_CATALOG_HEADLESS_STDERR_CHILD", "1")
        .env("CG_CATALOG_HEADLESS_STDERR_ROOT", stderr_temp.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("headless stderr child runs");
    stderr_source_server.shutdown_and_join();
    assert!(
        stderr_request_count.load(Ordering::SeqCst) > 0,
        "headless stderr probe should resolve through the local source override"
    );
    assert!(
        output.stderr.is_empty(),
        "headless mode should route progress to the log sink, not stderr"
    );
}

#[test]
fn sync_arrival_for_unknown_model_id_defers_with_typed_error_and_emits_observation_naming_the_unknown_id()
 {
    // Unknown model ids from another machine should defer with a typed error and diagnostic observation.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store();
    let result = catalog_at(temp.path(), "host-A").resolve_for_sync_arrival(
        &store,
        "custom-fine-tuned-v2",
        OutputMode::Headless,
    );
    assert!(matches!(
        result,
        Err(CgCatalogError::UnknownModelId { id }) if id == "custom-fine-tuned-v2"
    ));
    let rows = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    assert!(rows.iter().any(|observation| {
        observation.observation_type == "model_id_unknown_at_receiver"
            && observation.state_delta.get("unknown_model_id")
                == Some(&Some(serde_json::json!("custom-fine-tuned-v2")))
    }));
    assert!(
        !temp.path().join("custom-fine-tuned-v2").exists(),
        "unknown sync-arrival ids must not create cache directories"
    );
}
