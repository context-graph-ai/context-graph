use cg_catalog::{
    CacheStatus, CgCatalogError, FetchResult, ManifestStub, OutputMode, Source,
    validate_artifact_identity,
};
use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};

use crate::common::{
    CANONICAL_LICENSE_GATE_MESSAGE, FIXTURE_NOMIC_CONFIG_JSON_SHA,
    FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA, FIXTURE_NOMIC_REVISION_SHA,
    FIXTURE_NOMIC_TOKENIZER_JSON_SHA, GEMMA_ID, NOMIC_ID, NOMIC_REVISION, TEST_NEW_NOMIC_REVISION,
    acceptance, acceptance_records, catalog_at, corrupt_fixture_dir, fixture_dir, fixture_sha,
    hold_model_lock, serve_nomic_fixture_until_idle_counted, write_nomic_cache,
    write_source_override, write_source_override_with_actual_shas,
};

fn stub_base_url(server: &tiny_http::Server) -> String {
    let addr = server.server_addr().to_ip().expect("stub server TCP addr");
    format!("http://{}:{}", addr.ip(), addr.port())
}

fn eviction_fetch_child_if_requested() -> bool {
    let Some(root) = std::env::var_os("CG_CATALOG_EVICTION_FETCH_ROOT") else {
        return false;
    };
    let result = catalog_at(std::path::PathBuf::from(root), "host-A").fetch(
        None,
        NOMIC_ID,
        OutputMode::Interactive,
    );
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
    true
}

fn spawn_eviction_fetch_child(test_name: &str, cache_root: &std::path::Path) -> Child {
    Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg(test_name)
        .env("CG_CATALOG_EVICTION_FETCH_ROOT", cache_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("eviction fetch child spawns")
}

fn first_stderr_line_within(child: &mut Child, timeout: std::time::Duration) -> Option<String> {
    let stderr = child.stderr.take().expect("stderr piped");
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stderr).read_line(&mut line).map(|_| line);
        let _ = sender.send(result);
    });
    let line = receiver
        .recv_timeout(timeout)
        .ok()
        .and_then(std::result::Result::ok);
    if line.is_none() {
        let _ = child.kill();
    }
    line
}

#[derive(Clone)]
struct SharedLogWriter {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl std::io::Write for SharedLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.lock().expect("log bytes").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn air_gap_fetch_from_path_copies_artifacts_into_cache_root_and_validates_identity_against_pinned_canonical_fixture_shas_not_tautology()
 {
    // Local artifact import should run the same identity validator as a network fetch.
    let temp = tempfile::tempdir().expect("tempdir");
    let source = fixture_dir("nomic");
    write_source_override(
        temp.path(),
        NOMIC_ID,
        &format!("file://{}", source.display()),
        NOMIC_REVISION,
    );
    let catalog = catalog_at(temp.path(), "host-A");
    let result = catalog
        .fetch_from_path(NOMIC_ID, &source)
        .expect("from-path returns");
    assert_eq!(result, FetchResult::Completed);
    let expected = [
        ("model.safetensors", FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA),
        ("tokenizer.json", FIXTURE_NOMIC_TOKENIZER_JSON_SHA),
        ("config.json", FIXTURE_NOMIC_CONFIG_JSON_SHA),
        ("REVISION", FIXTURE_NOMIC_REVISION_SHA),
    ];
    for (file, expected_sha) in expected {
        let source_path = source.join(file);
        let cached_path = temp.path().join(NOMIC_ID).join(file);
        assert_eq!(fixture_sha(&source_path), expected_sha);
        assert!(
            cached_path.exists(),
            "from-path import should copy {file} into the model cache"
        );
        assert_eq!(
            std::fs::read(&cached_path).expect("cached artifact readable"),
            std::fs::read(&source_path).expect("source artifact readable"),
            "from-path import should preserve {file} bytes"
        );
        assert_eq!(fixture_sha(&cached_path), expected_sha);
    }
    validate_artifact_identity(NOMIC_ID, &temp.path().join(NOMIC_ID))
        .expect("valid imported fixture passes the shared identity validator");
    let manifest_path = temp.path().join(NOMIC_ID).join("catalog.toml");
    assert!(
        manifest_path.exists(),
        "from-path import should write a durable catalog manifest"
    );
    let manifest = std::fs::read_to_string(&manifest_path).expect("manifest readable");
    let manifest = toml::from_str::<toml::Value>(&manifest).expect("manifest parses as toml");
    let expected_manifest_shape = ManifestStub {
        artifact_filenames: expected
            .iter()
            .map(|(file, _)| (*file).to_string())
            .collect(),
        per_file_sha256: expected
            .iter()
            .map(|(file, sha)| ((*file).to_string(), (*sha).to_string()))
            .collect(),
        total_byte_budget: source
            .read_dir()
            .expect("fixture dir readable")
            .map(|entry| {
                entry
                    .expect("fixture entry")
                    .metadata()
                    .expect("metadata")
                    .len()
            })
            .sum(),
        ..Default::default()
    };
    assert!(
        expected_manifest_shape
            .artifact_filenames
            .iter()
            .any(|file| file == "model.safetensors")
    );
    let artifact_filenames = manifest
        .get("artifact_filenames")
        .and_then(toml::Value::as_array)
        .expect("manifest artifact_filenames array");
    for (file, _expected_sha) in expected {
        assert!(
            artifact_filenames
                .iter()
                .any(|candidate| candidate.as_str() == Some(file)),
            "manifest should list copied artifact {file}"
        );
    }
    let per_file_sha256 = manifest
        .get("per_file_sha256")
        .and_then(toml::Value::as_table)
        .expect("manifest per_file_sha256 table");
    for (file, expected_sha) in expected {
        assert_eq!(
            per_file_sha256.get(file).and_then(toml::Value::as_str),
            Some(expected_sha),
            "manifest should record a structured pinned SHA for {file}"
        );
    }
    let shown = catalog.show(NOMIC_ID).expect("show returns imported model");
    assert_eq!(
        shown.artifact_filenames,
        vec![
            "model.safetensors".to_string(),
            "tokenizer.json".to_string(),
            "config.json".to_string(),
            "REVISION".to_string(),
            "catalog.toml".to_string(),
        ]
    );
    assert!(matches!(
        shown.source,
        Source::LocalPath { ref dir }
            if dir == &source || source.canonicalize().as_ref().is_ok_and(|canonical| dir == canonical)
    ));
}

#[test]
fn air_gap_fetch_from_path_with_corrupted_artifact_refuses_with_typed_error() {
    // Corrupted local bytes must be rejected before they become cache authority.
    let temp = tempfile::tempdir().expect("tempdir");
    let corrupt = corrupt_fixture_dir();
    write_source_override(
        temp.path(),
        NOMIC_ID,
        &format!("file://{}", fixture_dir("nomic").display()),
        NOMIC_REVISION,
    );
    let helper_result = validate_artifact_identity(NOMIC_ID, &corrupt.path().join(NOMIC_ID));
    assert!(
        matches!(
            &helper_result,
            Err(CgCatalogError::ArtifactValidationFailed {
                file,
                expected_sha,
                ..
            }) if file == "model.safetensors" && expected_sha == FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA
        ),
        "shared identity validator should reject the same corrupt local artifact as fetch_from_path: {helper_result:?}"
    );
    let result =
        catalog_at(temp.path(), "host-A").fetch_from_path(NOMIC_ID, corrupt.path().join(NOMIC_ID));
    assert!(matches!(
        result,
        Err(CgCatalogError::ArtifactValidationFailed {
            file,
            expected_sha,
            ..
        }) if file == "model.safetensors" && expected_sha == FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA
    ));
    assert!(
        !temp
            .path()
            .join(NOMIC_ID)
            .join("model.safetensors")
            .exists()
    );
}

#[test]
fn air_gap_fetch_from_path_for_license_gated_model_still_requires_acceptance_record_before_use() {
    // Manual install of bytes should not bypass license acceptance.
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("manual-gemma-airgap-import");
    fs::create_dir_all(&source).expect("manual source dir created");
    fs::write(
        source.join("model.safetensors"),
        b"unique gated Gemma air-gap model bytes\n",
    )
    .expect("manual model artifact written");
    fs::write(
        source.join("tokenizer.json"),
        json!({"fixture": "unique-gemma-airgap-tokenizer"})
            .to_string()
            .as_bytes(),
    )
    .expect("manual tokenizer artifact written");
    fs::write(
        source.join("config.json"),
        json!({"fixture": "unique-gemma-airgap-config", "dimension": 768})
            .to_string()
            .as_bytes(),
    )
    .expect("manual config artifact written");
    fs::write(source.join("REVISION"), b"unique-gemma-airgap-revision\n")
        .expect("manual revision artifact written");
    let expected_source = [
        (
            "model.safetensors",
            fixture_sha(&source.join("model.safetensors")),
            fs::read(source.join("model.safetensors")).expect("manual model readable"),
        ),
        (
            "tokenizer.json",
            fixture_sha(&source.join("tokenizer.json")),
            fs::read(source.join("tokenizer.json")).expect("manual tokenizer readable"),
        ),
        (
            "config.json",
            fixture_sha(&source.join("config.json")),
            fs::read(source.join("config.json")).expect("manual config readable"),
        ),
        (
            "REVISION",
            fixture_sha(&source.join("REVISION")),
            fs::read(source.join("REVISION")).expect("manual revision readable"),
        ),
    ];
    write_source_override_with_actual_shas(
        temp.path(),
        GEMMA_ID,
        &format!("file://{}", source.display()),
        "unique-gemma-airgap-revision",
        &source,
    );
    let catalog = catalog_at(temp.path(), "host-A");
    let result = catalog
        .fetch_from_path(GEMMA_ID, &source)
        .expect("from-path returns");
    assert_eq!(result, FetchResult::CompletedAcceptanceRequired);
    for (file, expected_sha, expected_bytes) in expected_source.iter() {
        let source_bytes = fs::read(source.join(*file)).expect("manual source still readable");
        assert_eq!(
            source_bytes.as_slice(),
            expected_bytes.as_slice(),
            "from-path import must not rewrite caller-supplied source {file}"
        );
        let cached_path = temp.path().join(GEMMA_ID).join(file);
        assert!(
            cached_path.exists(),
            "from-path import should land gated {file} bytes before acceptance"
        );
        let cached_metadata =
            fs::symlink_metadata(&cached_path).expect("cached artifact metadata readable");
        assert!(
            cached_metadata.file_type().is_file(),
            "from-path import should land a regular cached file for {file}, not a link"
        );
        assert!(
            !cached_metadata.file_type().is_symlink(),
            "from-path import should not symlink cached {file} back to the caller source"
        );
        let cached_bytes = fs::read(&cached_path).expect("cached artifact readable");
        assert_eq!(
            cached_bytes.as_slice(),
            expected_bytes.as_slice(),
            "from-path import should preserve caller-supplied {file} bytes"
        );
        assert_eq!(fixture_sha(&cached_path), expected_sha.as_str());
    }
    let detached_source = temp.path().join("manual-gemma-airgap-import-detached");
    fs::rename(&source, &detached_source).expect("manual source detached after import");
    for (file, expected_sha, expected_bytes) in expected_source.iter() {
        let cached_path = temp.path().join(GEMMA_ID).join(file);
        let cached_bytes = fs::read(&cached_path)
            .expect("cached artifact should survive after source dir is detached");
        assert_eq!(cached_bytes.as_slice(), expected_bytes.as_slice());
        assert_eq!(fixture_sha(&cached_path), expected_sha.as_str());
    }
    let shown = catalog.show(GEMMA_ID).expect("show returns");
    assert_eq!(shown.cache_status, CacheStatus::Cached);
    assert!(
        !shown.acceptance_recorded,
        "fetch_from_path must not auto-accept a gated model"
    );
    let fetch = catalog.fetch(None, GEMMA_ID, OutputMode::Interactive);
    assert!(matches!(
        fetch,
        Err(CgCatalogError::ManualAcceptanceRequired {
            canonical_message,
            ..
        }) if canonical_message == CANONICAL_LICENSE_GATE_MESSAGE
    ));
    let expected_artifact_sha = expected_source
        .iter()
        .find_map(|(file, sha, _)| (*file == "model.safetensors").then(|| sha.clone()))
        .expect("expected model sha captured");
    let manifest_path = temp.path().join(GEMMA_ID).join("catalog.toml");
    let manifest_before_stale_record =
        fs::read(&manifest_path).expect("manifest readable before stale acceptance");
    let mut stale_record = acceptance("host-A");
    stale_record.artifact_sha = "stale-gemma-artifact-sha".to_string();
    let stale_result = catalog.record_acceptance(GEMMA_ID, stale_record);
    assert!(matches!(
        stale_result,
        Err(CgCatalogError::ArtifactShaMismatch { expected, actual })
            if expected == expected_artifact_sha && actual == "stale-gemma-artifact-sha"
    ));
    let manifest_after_stale_record =
        fs::read(&manifest_path).expect("manifest readable after stale acceptance");
    assert_eq!(
        manifest_after_stale_record, manifest_before_stale_record,
        "stale artifact acceptance must not mutate the durable manifest"
    );
    let shown_after_stale_record = catalog.show(GEMMA_ID).expect("show returns");
    assert!(
        !shown_after_stale_record.acceptance_recorded,
        "stale artifact acceptance must not mutate acceptance state"
    );
    let stale_records = acceptance_records(temp.path(), GEMMA_ID);
    assert!(
        stale_records.is_empty(),
        "stale artifact acceptance must not append to the on-disk ledger"
    );
    let mut record = acceptance("host-A");
    record.artifact_sha = expected_artifact_sha.clone();
    catalog
        .record_acceptance(GEMMA_ID, record)
        .expect("explicit acceptance returns");
    let records = acceptance_records(temp.path(), GEMMA_ID);
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].get("artifact_sha").and_then(toml::Value::as_str),
        Some(expected_artifact_sha.as_str())
    );
    let accepted_fetch = catalog
        .fetch(None, GEMMA_ID, OutputMode::Interactive)
        .expect("accepted gated cache fetches");
    assert_eq!(accepted_fetch, FetchResult::AlreadyCachedAndAccepted);
}

#[test]
fn air_gap_pre_populated_cache_reads_with_zero_outbound_network_on_first_load() {
    // Warm cache reads should not contact a remote server just to validate revision state.
    let temp = tempfile::tempdir().expect("tempdir");
    write_nomic_cache(temp.path());
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let outbound_request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&outbound_request_count);
    let server_thread = std::thread::spawn(move || {
        if let Ok(Some(request)) = server.recv_timeout(std::time::Duration::from_millis(150)) {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = request.respond(tiny_http::Response::from_string("unexpected"));
        }
    });
    let result = catalog_at(temp.path(), "host-A")
        .fetch(None, NOMIC_ID, OutputMode::Interactive)
        .expect("fetch returns");
    assert_eq!(result, FetchResult::AlreadyCached);
    server_thread.join().expect("server thread joins");
    assert_eq!(
        outbound_request_count.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
}

#[test]
fn air_gap_with_partial_cache_and_no_network_returns_typed_artifact_missing_error_does_not_silently_attempt_fetch()
 {
    // Partial caches in air-gapped mode should be explicit errors, not hidden network attempts.
    let temp = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(temp.path().join(NOMIC_ID)).expect("partial cache dir");
    std::fs::write(
        temp.path().join(NOMIC_ID).join("catalog.toml"),
        "cache_status = \"partial\"",
    )
    .expect("partial manifest written");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let outbound_request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&outbound_request_count);
    let server_thread = std::thread::spawn(move || {
        if let Ok(Some(request)) = server.recv_timeout(std::time::Duration::from_millis(150)) {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = request.respond(tiny_http::Response::from_string("unexpected"));
        }
    });
    let result = catalog_at(temp.path(), "host-A").fetch(None, NOMIC_ID, OutputMode::Interactive);
    assert!(matches!(
        result,
        Err(CgCatalogError::CacheIncomplete { missing_files }) if missing_files.contains(&"model.safetensors".to_string())
    ));
    server_thread.join().expect("server thread joins");
    assert_eq!(
        outbound_request_count.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "partial cache should fail before any outbound fetch attempt"
    );
}

#[test]
fn air_gap_with_artifact_only_partial_cache_fails_before_source_fetch_even_without_partial_manifest()
 {
    // A model directory with only some artifacts is already a partial cache, even without a manifest.
    let temp = tempfile::tempdir().expect("tempdir");
    let model_dir = temp.path().join(NOMIC_ID);
    std::fs::create_dir_all(&model_dir).expect("partial cache dir");
    std::fs::copy(
        fixture_dir("nomic").join("model.safetensors"),
        model_dir.join("model.safetensors"),
    )
    .expect("partial model copied");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let outbound_request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&outbound_request_count);
    let server_thread = std::thread::spawn(move || {
        if let Ok(Some(request)) = server.recv_timeout(std::time::Duration::from_millis(150)) {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = request.respond(tiny_http::Response::from_string("unexpected"));
        }
    });
    let result = catalog_at(temp.path(), "host-A").fetch(None, NOMIC_ID, OutputMode::Interactive);
    assert!(
        matches!(result, Err(CgCatalogError::CacheIncomplete { .. })),
        "artifact-only partial cache should return CacheIncomplete, got {result:?}"
    );
    if let Err(CgCatalogError::CacheIncomplete { mut missing_files }) = result {
        missing_files.sort();
        assert_eq!(
            missing_files,
            vec![
                "REVISION".to_string(),
                "catalog.toml".to_string(),
                "config.json".to_string(),
                "tokenizer.json".to_string()
            ],
            "partial cache diagnostics should name every absent required artifact and manifest"
        );
    }
    server_thread.join().expect("server thread joins");
    assert_eq!(
        outbound_request_count.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "artifact-only partial cache should fail before any outbound fetch attempt"
    );
}

#[test]
fn air_gap_manifest_matching_corrupt_bytes_does_not_override_catalog_artifact_identity_validation()
{
    // A manifest is descriptive metadata; it must not authorize bytes that violate catalog identity.
    let temp = tempfile::tempdir().expect("tempdir");
    let model_dir = write_nomic_cache(temp.path());
    std::fs::write(
        model_dir.join("model.safetensors"),
        b"corrupted bytes whose sha is copied into the manifest\n",
    )
    .expect("corrupt model artifact written");
    let corrupt_sha = fixture_sha(&model_dir.join("model.safetensors"));
    std::fs::write(
        model_dir.join("catalog.toml"),
        format!(
            "model_id = \"{NOMIC_ID}\"\nkind = \"TextEmbedder\"\nlicense_spdx = \"Apache-2.0\"\nlicense_url = \"https://www.apache.org/licenses/LICENSE-2.0\"\nsource_revision = \"{NOMIC_REVISION}\"\ncache_status = \"cached\"\nartifact_filenames = [\"model.safetensors\", \"tokenizer.json\", \"config.json\", \"REVISION\", \"catalog.toml\"]\n[per_file_sha256]\nmodel.safetensors = \"{corrupt_sha}\"\ntokenizer.json = \"{FIXTURE_NOMIC_TOKENIZER_JSON_SHA}\"\nconfig.json = \"{FIXTURE_NOMIC_CONFIG_JSON_SHA}\"\nREVISION = \"{FIXTURE_NOMIC_REVISION_SHA}\"\n"
        ),
    )
    .expect("manifest with corrupt sha written");
    let result = catalog_at(temp.path(), "host-A").fetch(None, NOMIC_ID, OutputMode::Interactive);
    assert!(
        matches!(result, Err(CgCatalogError::ArtifactValidationFailed { .. })),
        "manifest-matching corrupt bytes should fail artifact validation, got {result:?}"
    );
    if let Err(CgCatalogError::ArtifactValidationFailed {
        file,
        expected_sha,
        actual_sha,
        mismatches,
    }) = result
    {
        assert_eq!(file, "model.safetensors");
        assert_eq!(expected_sha, FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA);
        assert_eq!(actual_sha, corrupt_sha);
        assert!(
            mismatches.iter().any(|mismatch| {
                mismatch.file == "model.safetensors"
                    && mismatch.expected_sha == FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA
                    && mismatch.actual_sha == corrupt_sha
            }),
            "artifact validation should report the catalog SHA, not the manifest-copied corrupt SHA"
        );
    }
}

#[test]
fn manual_cache_eviction_via_rm_rf_reflected_in_list_and_re_embed_re_fetches_loudly() {
    // Manual eviction should make list show NotCached and the next embed should announce a fetch.
    if eviction_fetch_child_if_requested() {
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let model_dir = write_nomic_cache(temp.path());
    std::fs::remove_dir_all(&model_dir).expect("manual eviction succeeds");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_thread =
        serve_nomic_fixture_until_idle_counted(server, std::sync::Arc::clone(&request_count));
    let catalog = catalog_at(temp.path(), "host-A");
    let entries = catalog.list().expect("list returns");
    assert!(
        entries
            .iter()
            .any(|entry| entry.id == NOMIC_ID && entry.cache_status == CacheStatus::NotCached)
    );
    let held_lock = hold_model_lock(temp.path(), NOMIC_ID);
    let mut child = spawn_eviction_fetch_child(
        "air_gap::manual_cache_eviction_via_rm_rf_reflected_in_list_and_re_embed_re_fetches_loudly",
        temp.path(),
    );
    let stderr = first_stderr_line_within(&mut child, std::time::Duration::from_millis(250));
    assert!(
        stderr.is_some(),
        "post-eviction refetch must emit first-touch progress within 250ms"
    );
    let stderr = stderr.expect("asserted present");
    assert!(stderr.contains(NOMIC_ID));
    assert!(stderr.contains("546938168"));
    assert!(stderr.contains("huggingface.co"));
    assert!(stderr.contains(&temp.path().join(NOMIC_ID).display().to_string()));
    assert!(
        child.try_wait().expect("child status readable").is_none(),
        "progress should be emitted while the post-eviction fetch is still blocked"
    );
    drop(held_lock);
    assert!(child.wait().expect("eviction fetch child waits").success());
    server_thread.join().expect("server thread joins");
    assert!(
        request_count.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "post-eviction refetch should exercise the configured source after progress"
    );
}

#[test]
fn hf_endpoint_returns_200_with_corrupted_bytes_sha_check_catches_it_catalog_refuses_with_typed_error_and_diagnostic()
 {
    // A successful HTTP status with poisoned bytes should fail SHA validation and leave no cache entry.
    let temp = tempfile::tempdir().expect("tempdir");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, TEST_NEW_NOMIC_REVISION);
    let request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&request_count);
    let server_thread = std::thread::spawn(move || {
        while let Ok(Some(request)) = server.recv_timeout(std::time::Duration::from_secs(5)) {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let corrupt = b"corrupted model.safetensors bytes with matching HTTP success".to_vec();
            let response = tiny_http::Response::from_data(corrupt.clone()).with_header(
                tiny_http::Header::from_bytes(
                    &b"Content-Length"[..],
                    corrupt.len().to_string().as_bytes(),
                )
                .expect("content-length header"),
            );
            let _ = request.respond(response);
        }
    });
    let log_bytes = Arc::new(Mutex::new(Vec::new()));
    let writer_bytes = Arc::clone(&log_bytes);
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_ansi(false)
        .with_writer(move || SharedLogWriter {
            bytes: Arc::clone(&writer_bytes),
        })
        .finish();
    let result = tracing::subscriber::with_default(subscriber, || {
        catalog_at(temp.path(), "host-A").fetch(None, NOMIC_ID, OutputMode::Interactive)
    });
    assert!(
        matches!(
            &result,
            Err(CgCatalogError::ArtifactValidationFailed { .. })
        ),
        "corrupt bytes should refuse with ArtifactValidationFailed"
    );
    let Err(CgCatalogError::ArtifactValidationFailed {
        file,
        expected_sha,
        actual_sha,
        ..
    }) = result
    else {
        unreachable!("asserted ArtifactValidationFailed above")
    };
    assert_eq!(file, "model.safetensors");
    assert_eq!(expected_sha, FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA);
    assert_ne!(actual_sha, expected_sha);
    let diagnostics =
        String::from_utf8(log_bytes.lock().expect("log bytes").clone()).expect("utf8 log");
    let diagnostic_records = diagnostics
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .collect::<Vec<_>>();
    assert!(
        diagnostic_records.iter().any(|record| {
            record.get("level") == Some(&json!("ERROR"))
                && record.pointer("/fields/event") == Some(&json!("artifact_validation_failed"))
                && record.pointer("/fields/model_id") == Some(&json!(NOMIC_ID))
                && record.pointer("/fields/file") == Some(&json!(file))
                && record.pointer("/fields/expected_sha") == Some(&json!(expected_sha))
                && record.pointer("/fields/actual_sha") == Some(&json!(actual_sha))
        }),
        "corrupt-artifact diagnostic should name model id, file, expected SHA, and actual SHA; logs were: {diagnostics}"
    );
    assert!(
        !temp.path().join("context-graph.db").exists(),
        "fetch(None, ...) must not infer a hidden Store under the model cache root"
    );
    server_thread.join().expect("server thread joins");
    assert!(
        request_count.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "corrupt-HF test must exercise the configured stub endpoint"
    );
    assert!(!temp.path().join(NOMIC_ID).join("catalog.toml").exists());
    assert!(
        !temp
            .path()
            .join(NOMIC_ID)
            .join("model.safetensors")
            .exists(),
        "corrupt bytes must not remain as a partial cache artifact"
    );
}
