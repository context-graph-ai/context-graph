use crate::common::{
    FixtureServer, LockCheckpoint, catalog_with_child_lock_checkpoint, release_lock_checkpoint,
};
use std::fs;
use std::process::{Command, Stdio};
use std::sync::{Arc, Barrier, Condvar, Mutex, atomic::Ordering, mpsc};

use cg_catalog::{FetchResult, OutputMode};
use context_graph::{Principal, TxId};

use crate::common::{
    FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA, GEMMA_ID, NOMIC_ID, NOMIC_REVISION, acceptance,
    acceptance_records, catalog_at, fixture_dir, fixture_sha, hold_model_lock, model_lock_path,
    stub_base_url, try_acquire_model_lock, write_source_override,
    write_source_override_with_actual_shas, write_valid_mini_gemma_acceptance,
    write_valid_mini_gemma_cache,
};

fn child_fetch_if_requested() -> bool {
    let Some(root) = std::env::var_os("CG_CATALOG_CONCURRENCY_FETCH_ROOT") else {
        return false;
    };
    let root = std::path::PathBuf::from(root);
    let model_id = std::env::var("CG_CATALOG_CONCURRENCY_FETCH_MODEL").expect("model env");
    let catalog = catalog_with_child_lock_checkpoint(catalog_at(root, "host-A"));
    let result = catalog.fetch(None, &model_id, OutputMode::Interactive);
    match result {
        Ok(result) => {
            if let Ok(expected) = std::env::var("CG_CATALOG_CONCURRENCY_EXPECT_RESULT") {
                assert_eq!(format!("{result:?}"), expected);
            } else if let Some(path) = std::env::var_os("CG_CATALOG_CONCURRENCY_RESULT_FILE") {
                fs::write(path, format!("{result:?}")).expect("child result file written");
            } else {
                println!("{result:?}");
            }
        }
        Err(err) => {
            eprintln!("{err:?}");
            std::process::exit(1);
        }
    }
    true
}

fn spawn_fetch_child(
    test_name: &str,
    root: &std::path::Path,
    model_id: &str,
    expected_result: &str,
) -> std::process::Child {
    Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg(test_name)
        .env("CG_CATALOG_CONCURRENCY_FETCH_ROOT", root)
        .env("CG_CATALOG_CONCURRENCY_FETCH_MODEL", model_id)
        .env("CG_CATALOG_CONCURRENCY_EXPECT_RESULT", expected_result)
        .spawn()
        .expect("fetch child starts")
}

fn spawn_race_child(
    test_name: &str,
    root: &std::path::Path,
) -> (std::process::Child, std::path::PathBuf, LockCheckpoint) {
    static NEXT_RESULT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let result_file = root.join(format!(
        ".race-result-{}.txt",
        NEXT_RESULT.fetch_add(1, Ordering::SeqCst)
    ));
    let checkpoint = LockCheckpoint::new();
    let child = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg(test_name)
        .env("CG_CATALOG_CONCURRENCY_FETCH_ROOT", root)
        .env("CG_CATALOG_CONCURRENCY_FETCH_MODEL", NOMIC_ID)
        .env("CG_CATALOG_CONCURRENCY_RESULT_FILE", &result_file)
        .env("CG_CATALOG_TEST_LOCK_CHECKPOINT", checkpoint.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("race child starts");
    (child, result_file, checkpoint)
}

fn wait_for_fetch_result(
    child: (std::process::Child, std::path::PathBuf, LockCheckpoint),
    label: &str,
) -> String {
    let (child, result_file, _checkpoint) = child;
    let output = child.wait_with_output().expect("child output collected");
    assert!(
        output.status.success(),
        "{label} exited unsuccessfully\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    fs::read_to_string(&result_file)
        .unwrap_or_else(|_| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn acceptance_record_for<'a>(records: &'a [toml::Value], scope: &str) -> &'a toml::Value {
    let position = records
        .iter()
        .position(|record| record.get("scope_key").and_then(toml::Value::as_str) == Some(scope))
        .unwrap_or(usize::MAX);
    assert!(
        position < records.len(),
        "durable manifest should contain the {scope} acceptance record"
    );
    &records[position]
}

fn accepted_by_human(record: &toml::Value) -> Option<&str> {
    record
        .get("accepted_by")
        .and_then(toml::Value::as_table)
        .and_then(|accepted_by| accepted_by.get("Human"))
        .and_then(toml::Value::as_str)
}

fn assert_child_blocked(
    child: &mut std::process::Child,
    label: &str,
    checkpoint: &LockCheckpoint,
) -> std::os::unix::net::UnixStream {
    let held = checkpoint.entered();
    assert!(
        child.try_wait().expect("child status readable").is_none(),
        "{label} should still be blocked on the per-model lockfile"
    );
    held
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

fn respond_with_fixture_for_model(request: tiny_http::Request, fixture_name: &str) {
    let url = request.url().to_string();
    let filename = ["tokenizer.json", "config.json", "REVISION"]
        .into_iter()
        .find(|candidate| url.ends_with(candidate))
        .unwrap_or("model.safetensors");
    let bytes = std::fs::read(fixture_dir(fixture_name).join(filename)).expect("fixture bytes");
    let _ = request.respond(tiny_http::Response::from_data(bytes));
}

fn respond_with_model_dir(request: tiny_http::Request, model_dir: &std::path::Path) {
    let url = request.url().to_string();
    let filename = ["tokenizer.json", "config.json", "REVISION"]
        .into_iter()
        .find(|candidate| url.ends_with(candidate))
        .unwrap_or("model.safetensors");
    let bytes = std::fs::read(model_dir.join(filename)).expect("model source bytes");
    let _ = request.respond(tiny_http::Response::from_data(bytes));
}

fn start_first_response_gated_nomic_server(
    root: &std::path::Path,
) -> (
    mpsc::Receiver<()>,
    mpsc::Sender<()>,
    Arc<std::sync::atomic::AtomicUsize>,
    FixtureServer,
) {
    let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("stub server"));
    let base_url = stub_base_url(&server);
    write_source_override(root, NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&request_count);
    let (first_request_tx, first_request_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let server_thread = FixtureServer::from_server(server, move |server| {
        let mut first_release = Some(release_rx);
        while let Ok(request) = server.recv() {
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(release_rx) = first_release.take() {
                let first_request_tx = first_request_tx.clone();
                std::thread::spawn(move || {
                    first_request_tx.send(()).expect("first request signalled");
                    release_rx.recv().expect("first response release signal");
                    respond_with_nomic_fixture(request);
                });
            } else {
                std::thread::spawn(move || respond_with_nomic_fixture(request));
            }
        }
    });
    (first_request_rx, release_tx, request_count, server_thread)
}

struct OverlapSourceProbe {
    nomic_model_requests: Arc<std::sync::atomic::AtomicUsize>,
    gemma_model_requests: Arc<std::sync::atomic::AtomicUsize>,
    overlap_gate: Arc<(Mutex<ModelRequestOverlapState>, Condvar)>,
    server_thread: FixtureServer,
}

#[derive(Default)]
struct ModelRequestOverlapState {
    active: usize,
    max_active: usize,
    overlap_observed: bool,
}

fn wait_for_model_request_overlap(gate: &Arc<(Mutex<ModelRequestOverlapState>, Condvar)>) {
    let (lock, cvar) = &**gate;
    let mut state = lock.lock().expect("overlap state lock");
    state.active += 1;
    state.max_active = state.max_active.max(state.active);
    if state.active >= 2 {
        state.overlap_observed = true;
        cvar.notify_all();
    }
    while !state.overlap_observed {
        state = cvar.wait(state).expect("overlap cvar wait");
    }
    state.active -= 1;
    cvar.notify_all();
}

fn start_overlap_gated_nomic_and_gemma_source_server(root: &std::path::Path) -> OverlapSourceProbe {
    let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("stub server"));
    let base_url = stub_base_url(&server);
    let (gemma_source_dir, _unique_token) =
        write_valid_mini_gemma_cache(&root.join(".valid-gemma-source"));
    write_source_override(
        root,
        NOMIC_ID,
        &format!("{base_url}/{NOMIC_ID}"),
        NOMIC_REVISION,
    );
    write_source_override_with_actual_shas(
        root,
        GEMMA_ID,
        &format!("{base_url}/{GEMMA_ID}"),
        "embeddinggemma-300m-matryoshka-v1",
        &gemma_source_dir,
    );
    let nomic_model_requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let gemma_model_requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let overlap_gate = Arc::new((
        Mutex::new(ModelRequestOverlapState::default()),
        Condvar::new(),
    ));
    let nomic_counter = Arc::clone(&nomic_model_requests);
    let gemma_counter = Arc::clone(&gemma_model_requests);
    let server_overlap_gate = Arc::clone(&overlap_gate);
    let server_thread = FixtureServer::from_server(server, move |server| {
        while let Ok(request) = server.recv() {
            let url = request.url().to_string();
            let is_model = url.ends_with("model.safetensors");
            if url.contains(GEMMA_ID) {
                if is_model {
                    gemma_counter.fetch_add(1, Ordering::SeqCst);
                }
                let gemma_source_dir = gemma_source_dir.clone();
                let overlap_gate = Arc::clone(&server_overlap_gate);
                std::thread::spawn(move || {
                    if is_model {
                        wait_for_model_request_overlap(&overlap_gate);
                    }
                    respond_with_model_dir(request, &gemma_source_dir);
                });
            } else {
                if is_model {
                    assert!(url.contains(NOMIC_ID), "unexpected source URL {url}");
                    nomic_counter.fetch_add(1, Ordering::SeqCst);
                }
                let overlap_gate = Arc::clone(&server_overlap_gate);
                std::thread::spawn(move || {
                    if is_model {
                        wait_for_model_request_overlap(&overlap_gate);
                    }
                    respond_with_fixture_for_model(request, "nomic");
                });
            }
        }
    });
    OverlapSourceProbe {
        nomic_model_requests,
        gemma_model_requests,
        overlap_gate,
        server_thread,
    }
}

fn prepare_preaccepted_gemma_for_fetch(root: &std::path::Path) {
    let (_model_dir, record) = write_valid_mini_gemma_acceptance(root, "host-A");
    let catalog = catalog_at(root, "host-A");
    let _record = catalog
        .record_acceptance(GEMMA_ID, record)
        .expect("pre-acceptance returns");
    for file in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        fs::remove_file(root.join(GEMMA_ID).join(file))
            .expect("preaccepted Gemma artifact evicted");
    }
}

#[test]
fn concurrent_writers_to_same_model_dir_via_lockfile_serialize_one_winner_other_observes_already_cached_barrier_inside_critical_section()
 {
    // Two writers for the same model should serialize through the per-model lockfile.
    if child_fetch_if_requested() {
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let test_name = "concurrency::concurrent_writers_to_same_model_dir_via_lockfile_serialize_one_winner_other_observes_already_cached_barrier_inside_critical_section";
    let mut completed_results = 0;
    let mut already_cached_results = 0;
    let mut first_child_won = false;
    let mut second_child_won = false;

    for iteration in 0..10 {
        let race_root = temp.path().join(format!("same-model-race-{iteration}"));
        fs::create_dir_all(&race_root).expect("race root created");
        let (first_request_rx, release_tx, request_count, server_thread) =
            start_first_response_gated_nomic_server(&race_root);
        let held_lock = hold_model_lock(&race_root, NOMIC_ID);
        let mut first = spawn_race_child(test_name, &race_root);
        let mut second = spawn_race_child(test_name, &race_root);
        let first_held = assert_child_blocked(&mut first.0, "first writer", &first.2);
        let second_held = assert_child_blocked(&mut second.0, "second writer", &second.2);
        let (winner, loser) = if iteration % 2 == 0 {
            (first_held, second_held)
        } else {
            (second_held, first_held)
        };
        release_lock_checkpoint(winner);
        drop(held_lock);
        first_request_rx
            .recv()
            .expect("selected winner enters its critical section");
        release_lock_checkpoint(loser);
        assert!(
            !race_root.join(NOMIC_ID).join("catalog.toml").exists(),
            "a writer blocked inside its first source response must not publish a manifest"
        );
        assert_eq!(
            request_count.load(Ordering::SeqCst),
            1,
            "a second same-model writer must not enter network fetch while the first writer is in the critical section"
        );
        release_tx.send(()).expect("release first source response");
        let first_result = wait_for_fetch_result(first, "first same-model race child");
        let second_result = wait_for_fetch_result(second, "second same-model race child");
        first_child_won |= first_result == "Completed";
        second_child_won |= second_result == "Completed";
        let results = [first_result.as_str(), second_result.as_str()];
        assert_eq!(
            results
                .iter()
                .filter(|result| **result == "Completed")
                .count(),
            1,
            "exactly one writer should publish the cold cache"
        );
        assert_eq!(
            results
                .iter()
                .filter(|result| **result == "AlreadyCachedByConcurrentWinner")
                .count(),
            1,
            "the serialized loser should observe the winner's completed cache"
        );
        completed_results += results
            .iter()
            .filter(|result| **result == "Completed")
            .count();
        already_cached_results += results
            .iter()
            .filter(|result| **result == "AlreadyCachedByConcurrentWinner")
            .count();
        server_thread.join().expect("server thread joins");
        let manifest_path = race_root.join(NOMIC_ID).join("catalog.toml");
        assert!(
            manifest_path.exists(),
            "same-model manifest should be written by the winning process"
        );
        let manifest = fs::read_to_string(manifest_path).expect("manifest readable");
        assert_eq!(
            manifest
                .matches("model_id = \"nomic-embed-text-v1.5\"")
                .count(),
            1,
            "per-model lockfile should leave one complete manifest, not interleaved duplicate writes"
        );
        assert_eq!(
            fixture_sha(&race_root.join(NOMIC_ID).join("model.safetensors")),
            FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA,
            "the manifest winner's artifact bytes should match the pinned SHA"
        );
    }

    assert_eq!(completed_results, 10);
    assert_eq!(already_cached_results, 10);
    assert!(
        first_child_won && second_child_won,
        "alternating spawn order should exercise both winner orderings"
    );
}

#[test]
fn concurrent_writers_to_different_model_dirs_do_not_block_each_other() {
    // Different model ids should not be serialized behind a single global lock.
    if child_fetch_if_requested() {
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let parallel_root = temp.path().join("parallel");
    fs::create_dir_all(&parallel_root).expect("parallel root created");
    prepare_preaccepted_gemma_for_fetch(&parallel_root);
    let parallel_probe = start_overlap_gated_nomic_and_gemma_source_server(&parallel_root);
    let barrier = Arc::new(Barrier::new(2));
    let nomic_barrier = Arc::clone(&barrier);
    let nomic_root = parallel_root.clone();
    let nomic = std::thread::spawn(move || {
        nomic_barrier.wait();
        catalog_at(nomic_root, "host-A").fetch(None, NOMIC_ID, OutputMode::Headless)
    });
    let gemma_barrier = Arc::clone(&barrier);
    let gemma_root = parallel_root.clone();
    let gemma = std::thread::spawn(move || {
        gemma_barrier.wait();
        catalog_at(gemma_root, "host-A").fetch(None, GEMMA_ID, OutputMode::Headless)
    });
    let nomic_result = nomic.join().expect("nomic fetch thread joins");
    let gemma_result = gemma.join().expect("gemma fetch thread joins");
    parallel_probe
        .server_thread
        .join()
        .expect("parallel source server joins");
    assert_eq!(nomic_result, Ok(FetchResult::Completed));
    assert_eq!(gemma_result, Ok(FetchResult::Completed));
    let (overlap_lock, _) = &*parallel_probe.overlap_gate;
    let overlap_state = overlap_lock.lock().expect("overlap state lock");
    assert_eq!(
        overlap_state.max_active, 2,
        "parallel two-model fetch should put both model.safetensors requests in flight together"
    );
    assert_eq!(
        parallel_probe.nomic_model_requests.load(Ordering::SeqCst),
        1,
        "parallel probe should make exactly one Nomic model.safetensors request"
    );
    assert_eq!(
        parallel_probe.gemma_model_requests.load(Ordering::SeqCst),
        1,
        "parallel probe should make exactly one Gemma model.safetensors request"
    );
    assert_eq!(
        parallel_probe.nomic_model_requests.load(Ordering::SeqCst)
            + parallel_probe.gemma_model_requests.load(Ordering::SeqCst),
        2,
        "parallel probe should observe exactly one model.safetensors request per model id"
    );
}

#[test]
fn concurrent_writer_crash_mid_write_leaves_lockfile_releasable_for_next_writer() {
    // A crashed process must not leave a lock that blocks the next catalog operation forever.
    if child_fetch_if_requested() {
        return;
    }

    let temp = tempfile::tempdir().expect("tempdir");
    let (first_request_rx, release_tx, request_count, server_thread) =
        start_first_response_gated_nomic_server(temp.path());
    let mut child = spawn_fetch_child(
        "concurrency::concurrent_writer_crash_mid_write_leaves_lockfile_releasable_for_next_writer",
        temp.path(),
        NOMIC_ID,
        "Completed",
    );
    assert!(
        first_request_rx.recv().is_ok(),
        "crashing writer should reach the configured source through the real fetch path"
    );
    assert!(
        model_lock_path(temp.path(), NOMIC_ID).exists(),
        "fetch should create a durable per-model lockfile sentinel before the source request"
    );
    assert!(
        try_acquire_model_lock(temp.path(), NOMIC_ID).is_err(),
        "while the fetch child is alive inside the source response, the model lock should not be acquirable"
    );
    child.kill().expect("crash child killed mid-fetch");
    let status = child.wait().expect("crash child waits");
    assert!(!status.success(), "child writer should exit mid-write");
    release_tx
        .send(())
        .expect("release abandoned source response");
    let released_lock =
        try_acquire_model_lock(temp.path(), NOMIC_ID).expect("released lock is acquirable");
    drop(released_lock);
    let result = catalog_at(temp.path(), "host-A")
        .fetch(None, NOMIC_ID, OutputMode::Interactive)
        .expect("fetch returns");
    assert_eq!(result, FetchResult::Completed);
    assert!(
        request_count.load(Ordering::SeqCst) >= 2,
        "recovery should perform a fresh source request after the crashed fetch"
    );
    assert!(
        temp.path().join(NOMIC_ID).join("catalog.toml").exists(),
        "post-crash writer should acquire the released OS lock and complete the manifest"
    );
    server_thread.join().expect("server thread joins");
}

#[test]
fn concurrent_acceptance_record_writes_serialize_via_lockfile_no_lost_writes_strict_monotonic_accepted_at()
 {
    // Concurrent acceptance writes should preserve both records in timestamp order.
    if let Some(root) = std::env::var_os("CG_CATALOG_ACCEPTANCE_CHILD_ROOT") {
        let scope = std::env::var("CG_CATALOG_ACCEPTANCE_CHILD_SCOPE").expect("scope env");
        let root = std::path::PathBuf::from(root);
        let mut record = acceptance(&scope);
        record.artifact_sha = fixture_sha(&root.join(GEMMA_ID).join("model.safetensors"));
        record.accepted_at = if scope == "host-A" { TxId(1) } else { TxId(2) };
        record.accepted_by = Principal::Human(format!("{scope}@example.test"));
        catalog_with_child_lock_checkpoint(catalog_at(root, &scope))
            .record_acceptance(GEMMA_ID, record)
            .expect("child acceptance returns");
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let held_lock = hold_model_lock(temp.path(), GEMMA_ID);
    let first_checkpoint = LockCheckpoint::new();
    let second_checkpoint = LockCheckpoint::new();
    let mut first = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("concurrency::concurrent_acceptance_record_writes_serialize_via_lockfile_no_lost_writes_strict_monotonic_accepted_at")
        .env("CG_CATALOG_ACCEPTANCE_CHILD_ROOT", temp.path())
        .env("CG_CATALOG_ACCEPTANCE_CHILD_SCOPE", "host-A")
        .env("CG_CATALOG_TEST_LOCK_CHECKPOINT", first_checkpoint.path())
        .spawn()
        .expect("first acceptance child starts");
    let mut second = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("concurrency::concurrent_acceptance_record_writes_serialize_via_lockfile_no_lost_writes_strict_monotonic_accepted_at")
        .env("CG_CATALOG_ACCEPTANCE_CHILD_ROOT", temp.path())
        .env("CG_CATALOG_ACCEPTANCE_CHILD_SCOPE", "host-B")
        .env("CG_CATALOG_TEST_LOCK_CHECKPOINT", second_checkpoint.path())
        .spawn()
        .expect("second acceptance child starts");
    let first_held = assert_child_blocked(&mut first, "first acceptance writer", &first_checkpoint);
    let second_held =
        assert_child_blocked(&mut second, "second acceptance writer", &second_checkpoint);
    release_lock_checkpoint(first_held);
    release_lock_checkpoint(second_held);
    drop(held_lock);
    assert!(first.wait().expect("first child waits").success());
    assert!(second.wait().expect("second child waits").success());
    let records = acceptance_records(temp.path(), GEMMA_ID);
    let host_a_record = acceptance_record_for(&records, "host-A");
    let host_b_record = acceptance_record_for(&records, "host-B");
    assert_eq!(
        host_a_record
            .get("accepted_at")
            .and_then(toml::Value::as_integer),
        Some(1),
        "host-A timestamp should be durable"
    );
    assert_eq!(
        host_b_record
            .get("accepted_at")
            .and_then(toml::Value::as_integer),
        Some(2),
        "host-B timestamp should be durable and strictly greater than host-A"
    );
    assert_eq!(
        accepted_by_human(host_a_record),
        Some("host-A@example.test"),
        "host-A acceptance principal should be durable"
    );
    assert_eq!(
        accepted_by_human(host_b_record),
        Some("host-B@example.test"),
        "host-B acceptance principal should be durable and distinct"
    );
    let host_a = records
        .iter()
        .position(|record| record.get("scope_key").and_then(toml::Value::as_str) == Some("host-A"))
        .unwrap_or(usize::MAX);
    let host_b = records
        .iter()
        .position(|record| record.get("scope_key").and_then(toml::Value::as_str) == Some("host-B"))
        .unwrap_or(usize::MAX);
    assert!(
        host_a < host_b,
        "acceptance records should remain append-only and ordered"
    );
}

#[test]
fn concurrent_read_during_write_observes_either_pre_write_state_or_post_write_state_never_torn() {
    // Readers should never see a manifest saying cached while artifact bytes are missing or mismatched.
    if child_fetch_if_requested() {
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let observations = Arc::new(Mutex::new(Vec::new()));
    let root = temp.path().to_path_buf();
    let (first_request_rx, release_tx, _request_count, server_thread) =
        start_first_response_gated_nomic_server(temp.path());
    let seen = Arc::clone(&observations);
    let (waiting_tx, waiting_rx) = mpsc::channel();
    let (sample_tx, sample_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        sample_rx
            .recv()
            .expect("writer entered its critical section");
        let catalog = catalog_at(&root, "reader").with_lock_wait_observer_for_test(move || {
            waiting_tx.send(()).expect("reader reports lock contention");
        });
        let status = catalog
            .list()
            .expect("reader list returns")
            .into_iter()
            .find(|entry| entry.id == NOMIC_ID)
            .map(|entry| format!("{:?}", entry.cache_status))
            .unwrap_or_else(|| "Missing".to_string());
        let artifact = root.join(NOMIC_ID).join("model.safetensors");
        let sha_ok =
            artifact.exists() && fixture_sha(&artifact) == FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA;
        seen.lock()
            .expect("reader observations")
            .push((status, sha_ok));
    });
    let mut writer = spawn_fetch_child(
        "concurrency::concurrent_read_during_write_observes_either_pre_write_state_or_post_write_state_never_torn",
        temp.path(),
        NOMIC_ID,
        "Completed",
    );
    assert!(
        first_request_rx.recv().is_ok(),
        "writer should enter source fetch before reader samples"
    );
    sample_tx.send(()).expect("start snapshot read");
    waiting_rx
        .recv()
        .expect("reader reached the writer-held snapshot lock");
    release_tx.send(()).expect("release first source response");
    assert!(writer.wait().expect("writer child waits").success());
    reader.join().expect("reader joins");
    server_thread.join().expect("server thread joins");
    let observations = observations.lock().expect("reader observations");
    assert!(
        !observations.is_empty(),
        "the test must run a real catalog reader while the writer is in its critical section"
    );
    assert!(
        observations
            .iter()
            .all(|(status, sha_ok)| (status == "Cached" && *sha_ok)
                || (status == "NotCached" && !*sha_ok))
    );
}
