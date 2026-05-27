use std::fs;
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Barrier, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

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
    let catalog = catalog_at(root, "host-A");
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
) -> (std::process::Child, std::path::PathBuf) {
    let result_file = root.join(format!(
        ".race-result-{:?}-{}.txt",
        std::thread::current().id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos()
    ));
    let child = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg(test_name)
        .env("CG_CATALOG_CONCURRENCY_FETCH_ROOT", root)
        .env("CG_CATALOG_CONCURRENCY_FETCH_MODEL", NOMIC_ID)
        .env("CG_CATALOG_CONCURRENCY_RESULT_FILE", &result_file)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("race child starts");
    (child, result_file)
}

fn wait_for_fetch_result(child: (std::process::Child, std::path::PathBuf), label: &str) -> String {
    let (child, result_file) = child;
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

fn assert_child_blocked(child: &mut std::process::Child, label: &str) {
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        child.try_wait().expect("child status readable").is_none(),
        "{label} should still be blocked on the per-model lockfile"
    );
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
    std::thread::JoinHandle<()>,
) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(root, NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&request_count);
    let (first_request_tx, first_request_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let server_thread = std::thread::spawn(move || {
        let mut first_release = Some(release_rx);
        while let Ok(Some(request)) = server.recv_timeout(Duration::from_secs(5)) {
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(release_rx) = first_release.take() {
                let first_request_tx = first_request_tx.clone();
                std::thread::spawn(move || {
                    first_request_tx.send(()).expect("first request signalled");
                    release_rx
                        .recv_timeout(Duration::from_secs(3))
                        .expect("first response release signal");
                    respond_with_nomic_fixture(request);
                });
            } else {
                std::thread::spawn(move || respond_with_nomic_fixture(request));
            }
        }
    });
    (first_request_rx, release_tx, request_count, server_thread)
}

struct DelayedSourceProbe {
    nomic_model_requests: Arc<std::sync::atomic::AtomicUsize>,
    gemma_model_requests: Arc<std::sync::atomic::AtomicUsize>,
    server_thread: std::thread::JoinHandle<()>,
}

fn start_delayed_nomic_and_gemma_source_server(root: &std::path::Path) -> DelayedSourceProbe {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
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
    let nomic_counter = Arc::clone(&nomic_model_requests);
    let gemma_counter = Arc::clone(&gemma_model_requests);
    let server_thread = std::thread::spawn(move || {
        while let Ok(Some(request)) = server.recv_timeout(Duration::from_secs(5)) {
            let url = request.url().to_string();
            let is_model = url.ends_with("model.safetensors");
            if url.contains(GEMMA_ID) {
                if is_model {
                    gemma_counter.fetch_add(1, Ordering::SeqCst);
                }
                let gemma_source_dir = gemma_source_dir.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(200));
                    respond_with_model_dir(request, &gemma_source_dir);
                });
            } else {
                if is_model {
                    assert!(url.contains(NOMIC_ID), "unexpected source URL {url}");
                    nomic_counter.fetch_add(1, Ordering::SeqCst);
                }
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(200));
                    respond_with_fixture_for_model(request, "nomic");
                });
            }
        }
    });
    DelayedSourceProbe {
        nomic_model_requests,
        gemma_model_requests,
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
        let barrier = Arc::new(Barrier::new(2));
        let first_delay = if iteration % 2 == 0 {
            Duration::from_millis(0)
        } else {
            Duration::from_millis(30)
        };
        let second_delay = if iteration % 2 == 0 {
            Duration::from_millis(30)
        } else {
            Duration::from_millis(0)
        };
        let first_barrier = Arc::clone(&barrier);
        let first_root = race_root.clone();
        let first_handle = std::thread::spawn(move || {
            first_barrier.wait();
            std::thread::sleep(first_delay);
            spawn_race_child(test_name, &first_root)
        });
        let second_barrier = Arc::clone(&barrier);
        let second_root = race_root.clone();
        let second_handle = std::thread::spawn(move || {
            second_barrier.wait();
            std::thread::sleep(second_delay);
            spawn_race_child(test_name, &second_root)
        });
        let mut first = first_handle.join().expect("first race child spawned");
        let mut second = second_handle.join().expect("second race child spawned");
        assert_child_blocked(
            &mut first.0,
            "first same-model writer before parent release",
        );
        assert_child_blocked(
            &mut second.0,
            "second same-model writer before parent release",
        );
        drop(held_lock);
        assert!(
            first_request_rx
                .recv_timeout(Duration::from_millis(700))
                .is_ok(),
            "one writer should reach the configured source before publishing artifacts"
        );
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

    let baseline_root = temp.path().join("baseline");
    fs::create_dir_all(&baseline_root).expect("baseline root created");
    let baseline_probe = start_delayed_nomic_and_gemma_source_server(&baseline_root);
    let baseline_started = Instant::now();
    let baseline_result =
        catalog_at(&baseline_root, "host-A").fetch(None, NOMIC_ID, OutputMode::Headless);
    let t1 = baseline_started.elapsed();
    baseline_probe
        .server_thread
        .join()
        .expect("baseline source server joins");
    assert_eq!(baseline_result, Ok(FetchResult::Completed));
    assert!(
        t1 > Duration::from_millis(100),
        "single-fetch baseline must prove the delayed stub is actually slow"
    );
    assert_eq!(
        baseline_probe.nomic_model_requests.load(Ordering::SeqCst),
        1,
        "baseline should make exactly one Nomic model.safetensors request"
    );

    let parallel_root = temp.path().join("parallel");
    fs::create_dir_all(&parallel_root).expect("parallel root created");
    prepare_preaccepted_gemma_for_fetch(&parallel_root);
    let parallel_probe = start_delayed_nomic_and_gemma_source_server(&parallel_root);
    let barrier = Arc::new(Barrier::new(2));
    let parallel_started = Instant::now();
    let nomic_barrier = Arc::clone(&barrier);
    let nomic_root = parallel_root.clone();
    let nomic = std::thread::spawn(move || {
        nomic_barrier.wait();
        let result = catalog_at(nomic_root, "host-A").fetch(None, NOMIC_ID, OutputMode::Headless);
        (result, Instant::now())
    });
    let gemma_barrier = Arc::clone(&barrier);
    let gemma_root = parallel_root.clone();
    let gemma = std::thread::spawn(move || {
        gemma_barrier.wait();
        let result = catalog_at(gemma_root, "host-A").fetch(None, GEMMA_ID, OutputMode::Headless);
        (result, Instant::now())
    });
    let (nomic_result, nomic_done) = nomic.join().expect("nomic fetch thread joins");
    let (gemma_result, gemma_done) = gemma.join().expect("gemma fetch thread joins");
    let t2 = std::cmp::max(nomic_done, gemma_done).duration_since(parallel_started);
    parallel_probe
        .server_thread
        .join()
        .expect("parallel source server joins");
    assert_eq!(nomic_result, Ok(FetchResult::Completed));
    assert_eq!(gemma_result, Ok(FetchResult::Completed));
    assert!(
        t2 < Duration::from_millis(150) + t1,
        "parallel two-model fetch should stay near the single-fetch baseline"
    );
    assert!(
        t2 < (t1 * 2) - Duration::from_millis(50),
        "parallel two-model fetch should not serialize behind a global lock"
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
        first_request_rx
            .recv_timeout(Duration::from_millis(700))
            .is_ok(),
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
        catalog_at(root, &scope)
            .record_acceptance(GEMMA_ID, record)
            .expect("child acceptance returns");
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let held_lock = hold_model_lock(temp.path(), GEMMA_ID);
    let mut first = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("concurrency::concurrent_acceptance_record_writes_serialize_via_lockfile_no_lost_writes_strict_monotonic_accepted_at")
        .env("CG_CATALOG_ACCEPTANCE_CHILD_ROOT", temp.path())
        .env("CG_CATALOG_ACCEPTANCE_CHILD_SCOPE", "host-A")
        .spawn()
        .expect("first acceptance child starts");
    let mut second = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("concurrency::concurrent_acceptance_record_writes_serialize_via_lockfile_no_lost_writes_strict_monotonic_accepted_at")
        .env("CG_CATALOG_ACCEPTANCE_CHILD_ROOT", temp.path())
        .env("CG_CATALOG_ACCEPTANCE_CHILD_SCOPE", "host-B")
        .spawn()
        .expect("second acceptance child starts");
    assert_child_blocked(&mut first, "first acceptance writer");
    assert_child_blocked(&mut second, "second acceptance writer");
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
    let done = Arc::new(AtomicBool::new(false));
    let done_reader = Arc::clone(&done);
    let reader = std::thread::spawn(move || {
        loop {
            let catalog = catalog_at(&root, "reader");
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
            if done_reader.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
    let mut writer = spawn_fetch_child(
        "concurrency::concurrent_read_during_write_observes_either_pre_write_state_or_post_write_state_never_torn",
        temp.path(),
        NOMIC_ID,
        "Completed",
    );
    assert!(
        first_request_rx
            .recv_timeout(Duration::from_millis(700))
            .is_ok(),
        "writer should enter source fetch before reader samples"
    );
    std::thread::sleep(Duration::from_millis(250));
    release_tx.send(()).expect("release first source response");
    assert!(writer.wait().expect("writer child waits").success());
    done.store(true, Ordering::SeqCst);
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
