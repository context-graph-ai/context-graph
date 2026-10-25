use cg_catalog::{
    CG_MODEL_CACHE_ROOT_ENV, CacheStatus, FetchObservation, FetchResult, OutputMode, ProgressEvent,
    SkipReason, Source,
};

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

use crate::common::{
    CountedNomicFixtureServer, FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA, GEMMA_ID, NOMIC_ID,
    NOMIC_REVISION, catalog_at, fixture_dir, fixture_sha, hold_model_lock,
    serve_nomic_fixture_until_idle_counted, stub_base_url, write_source_override,
    write_source_override_with_actual_shas,
};

const NOMIC_PRODUCTION_MODEL_SHA: &str =
    "9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c";
const NOMIC_PRODUCTION_TOKENIZER_SHA: &str =
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";
const NOMIC_PRODUCTION_CONFIG_SHA: &str =
    "9ab00bd92cee80a569f708140b7b6c1661a65891ff3765b1519e181ba2f2c92b";
const NOMIC_PRODUCTION_REVISION_SHA: &str =
    "b6e8c34dc906161922aedff26e55cc6ec85162b83826c43ef604452489c407f1";
const NOMIC_DEFAULT_SOURCE_REPO: &str = "nomic-ai/nomic-embed-text-v1.5";
const ALL_ZERO_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives under repo/crates/cg-catalog")
        .to_path_buf()
}

fn install_script() -> PathBuf {
    repo_root().join("scripts/install.sh")
}

fn run_install(home: &Path, prefix: &Path, cache_root: &Path, extra_args: &[&str]) -> Output {
    assert!(install_script().exists(), "repo-root install script exists");
    Command::new("bash")
        .arg(install_script())
        .arg("--prefix")
        .arg(prefix)
        .args(extra_args)
        .env("HOME", home)
        .env(CG_MODEL_CACHE_ROOT_ENV, cache_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("install script spawns")
}

fn assert_success(output: &Output, label: &str) {
    assert!(
        output.status.success(),
        "{label} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn fake_prefetch_tools_with_logs(
    bin_dir: &Path,
    request_log: &Path,
    sha_log: &Path,
    final_model_dir: &Path,
    mismatched_artifact: Option<(&str, &str)>,
) {
    fs::create_dir_all(bin_dir).expect("fake curl bin dir created");
    let fake_curl = bin_dir.join("curl");
    let curl_script = format!(
        r#"#!/usr/bin/env bash
set -euo pipefail

out=""
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o)
      out="$2"
      shift 2
      ;;
    --retry|--connect-timeout)
      shift 2
      ;;
    -*)
      shift
      ;;
    *)
      url="$1"
      shift
      ;;
  esac
done

final_model_dir='{}'
case "$out" in
  "$final_model_dir"/*)
    echo "default prefetch must download into a temporary model dir before final publication: $out" >&2
    exit 72
    ;;
esac
printf '%s\n' "$url" >> "{}"
case "$url" in
  */model.safetensors)
    printf '%s' 'default-branch-placeholder-model' > "$out"
    ;;
  */tokenizer.json)
    printf '%s' '{{"placeholder":"tokenizer"}}' > "$out"
    ;;
  */config.json)
    printf '%s' '{{"placeholder":"config"}}' > "$out"
    ;;
  */REVISION)
    echo "default prefetch must not request remote REVISION artifact: $url" >&2
    exit 66
    ;;
  *)
    echo "unexpected default prefetch URL: $url" >&2
    exit 67
    ;;
esac
	"#,
        final_model_dir.display(),
        request_log.display()
    );
    fs::write(&fake_curl, curl_script).expect("fake curl script written");
    let fake_sha256sum = bin_dir.join("sha256sum");
    let (mismatched_artifact, mismatched_sha) = mismatched_artifact.unwrap_or(("", ""));
    let sha_script = format!(
        r#"#!/usr/bin/env bash
set -euo pipefail

path="$1"
printf '%s\n' "$path" >> "{}"
final_model_dir='{}'
[ ! -e "$final_model_dir" ] || {{
  echo "final model dir must not exist before sha validation completes: $final_model_dir" >&2
  exit 71
}}
[ -f "$path" ] || {{
  echo "sha256sum called before artifact exists: $path" >&2
  exit 68
}}
expected_sha=""
case "$path" in
  */model.safetensors)
    expected_sha='{}'
    ;;
  */tokenizer.json)
    expected_sha='{}'
    ;;
  */config.json)
    expected_sha='{}'
    ;;
  */REVISION)
    revision="$(cat "$path")"
    [ "$revision" = '{}' ] || {{
      echo "REVISION must contain the built-in pinned revision before sha validation: $revision" >&2
      exit 69
    }}
    expected_sha='{}'
    ;;
  *)
    echo "unexpected sha256sum path: $path" >&2
    exit 70
  ;;
esac
mismatched_artifact='{}'
mismatched_sha='{}'
if [ "$mismatched_artifact" = "$(basename "$path")" ]; then
  expected_sha="$mismatched_sha"
fi
printf '%s  %s\n' "$expected_sha" "$path"
	"#,
        sha_log.display(),
        final_model_dir.display(),
        NOMIC_PRODUCTION_MODEL_SHA,
        NOMIC_PRODUCTION_TOKENIZER_SHA,
        NOMIC_PRODUCTION_CONFIG_SHA,
        NOMIC_REVISION,
        NOMIC_PRODUCTION_REVISION_SHA,
        mismatched_artifact,
        mismatched_sha,
    );
    fs::write(&fake_sha256sum, sha_script).expect("fake sha256sum script written");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for script in [&fake_curl, &fake_sha256sum] {
            let mut permissions = fs::metadata(script)
                .unwrap_or_else(|_| panic!("{} metadata", script.display()))
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(script, permissions)
                .unwrap_or_else(|_| panic!("{} executable", script.display()));
        }
    }
}

#[test]
fn install_script_help_prints_supported_flags_without_mutating_prefix() {
    // The installer should be self-describing before a developer lets it touch a prefix.
    let temp = tempfile::tempdir().expect("tempdir");
    let prefix = temp.path().join("prefix");
    let output = Command::new("bash")
        .arg(install_script())
        .arg("--help")
        .arg("--prefix")
        .arg(&prefix)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("install help spawns");
    assert_success(&output, "install --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--prefix"));
    assert!(stdout.contains("--dry-run"));
    assert!(stdout.contains("--prefetch-defaults"));
    assert!(
        !prefix.exists(),
        "--help should not create install directories before the user chooses a prefix"
    );
}

#[test]
fn install_prefetch_without_source_override_uses_local_revision_marker() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    let cache_root = temp.path().join("cache");
    let fake_bin = temp.path().join("fake-bin");
    let request_log = temp.path().join("curl-requests.log");
    let sha_log = temp.path().join("sha-requests.log");
    let model_dir = cache_root.join(NOMIC_ID);
    fs::create_dir_all(&home).expect("home created");
    fake_prefetch_tools_with_logs(&fake_bin, &request_log, &sha_log, &model_dir, None);
    let path = format!(
        "{}:{}",
        fake_bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    let output = Command::new("bash")
        .arg(install_script())
        .arg("--prefix")
        .arg(&prefix)
        .arg("--prefetch-defaults")
        .env("HOME", &home)
        .env(CG_MODEL_CACHE_ROOT_ENV, &cache_root)
        .env("PATH", path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("install script spawns");

    assert_success(
        &output,
        "install --prefetch-defaults without source override",
    );
    assert!(
        request_log.exists(),
        "installer should invoke curl for default artifacts"
    );
    let requests = fs::read_to_string(&request_log).expect("curl request log readable");
    let requests = requests.lines().collect::<Vec<_>>();
    let expected_urls = [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
    ]
    .into_iter()
    .map(|artifact| {
        format!("https://huggingface.co/{NOMIC_DEFAULT_SOURCE_REPO}/resolve/{NOMIC_REVISION}/{artifact}")
    })
    .collect::<std::collections::BTreeSet<_>>();
    let actual_urls = requests
        .iter()
        .copied()
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        requests.len(),
        3,
        "default prefetch should request only model/tokenizer/config from the pinned source\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        actual_urls, expected_urls,
        "default prefetch must request the exact pinned Hugging Face artifact URLs"
    );
    assert!(
        requests.iter().all(|url| !url.ends_with("/REVISION")),
        "default prefetch must synthesize built-in REVISION locally instead of fetching a remote REVISION artifact; requests={requests:?}"
    );

    assert!(
        sha_log.exists(),
        "installer should validate prefetched artifacts with sha256sum"
    );
    let sha_paths = fs::read_to_string(&sha_log).expect("sha request log readable");
    let sha_paths = sha_paths.lines().collect::<Vec<_>>();
    let mut validation_parents = std::collections::BTreeSet::new();
    for artifact in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        let matching_path = sha_paths.iter().find(|path| {
            let path = Path::new(path);
            path.file_name()
                .is_some_and(|file_name| file_name == artifact)
                && path
                    .parent()
                    .is_some_and(|parent| parent != model_dir.as_path())
        });
        assert!(
            matching_path.is_some(),
            "installer must validate prefetched {artifact} in a temporary model dir before publishing, not in the final model dir; sha_paths={sha_paths:?}"
        );
        validation_parents.insert(
            Path::new(matching_path.expect("matching path checked"))
                .parent()
                .expect("validation path parent")
                .to_path_buf(),
        );
    }
    assert_eq!(
        validation_parents.len(),
        1,
        "installer should validate all default artifacts in one temporary model dir before publishing; validation_parents={validation_parents:?}"
    );
    for (artifact, expected_contents) in [
        ("model.safetensors", "default-branch-placeholder-model"),
        ("tokenizer.json", r#"{"placeholder":"tokenizer"}"#),
        ("config.json", r#"{"placeholder":"config"}"#),
    ] {
        let actual = fs::read_to_string(model_dir.join(artifact))
            .unwrap_or_else(|_| panic!("installer should publish {artifact}"));
        assert_eq!(
            actual, expected_contents,
            "installer should publish validated {artifact} bytes into the final model cache"
        );
    }
    let revision = fs::read_to_string(model_dir.join("REVISION"))
        .expect("installer should publish local REVISION marker");
    assert_eq!(
        revision.trim(),
        NOMIC_REVISION,
        "default prefetch should publish the built-in pinned revision as the local REVISION marker"
    );
    let manifest_text = fs::read_to_string(model_dir.join("catalog.toml"))
        .expect("installer should publish manifest");
    let manifest = toml::from_str::<toml::Value>(&manifest_text)
        .expect("installer manifest should be valid TOML");
    assert_eq!(
        manifest
            .get("source_revision")
            .and_then(toml::Value::as_str),
        Some(NOMIC_REVISION),
        "manifest should record the built-in pinned revision after no-override prefetch\n{manifest_text}"
    );

    for (mismatched_artifact, mismatched_sha) in [
        (
            "model.safetensors",
            "1111111111111111111111111111111111111111111111111111111111111111",
        ),
        (
            "tokenizer.json",
            "2222222222222222222222222222222222222222222222222222222222222222",
        ),
        (
            "config.json",
            "3333333333333333333333333333333333333333333333333333333333333333",
        ),
        (
            "REVISION",
            "4444444444444444444444444444444444444444444444444444444444444444",
        ),
        ("model.safetensors", ALL_ZERO_SHA),
        ("tokenizer.json", ALL_ZERO_SHA),
        ("config.json", ALL_ZERO_SHA),
        ("REVISION", ALL_ZERO_SHA),
    ] {
        let case_name = if mismatched_sha == ALL_ZERO_SHA {
            format!("{}-zero-sha", mismatched_artifact.replace('.', "-"))
        } else {
            mismatched_artifact.replace('.', "-")
        };
        let bad_prefix = temp.path().join(format!("bad-prefix-{case_name}"));
        let bad_cache_root = temp.path().join(format!("bad-cache-{case_name}"));
        let bad_fake_bin = temp.path().join(format!("bad-fake-bin-{case_name}"));
        let bad_request_log = temp
            .path()
            .join(format!("bad-curl-requests-{case_name}.log"));
        let bad_sha_log = temp
            .path()
            .join(format!("bad-sha-requests-{case_name}.log"));
        let bad_model_dir = bad_cache_root.join(NOMIC_ID);
        fake_prefetch_tools_with_logs(
            &bad_fake_bin,
            &bad_request_log,
            &bad_sha_log,
            &bad_model_dir,
            Some((mismatched_artifact, mismatched_sha)),
        );
        let bad_path = format!(
            "{}:{}",
            bad_fake_bin.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let bad_output = Command::new("bash")
            .arg(install_script())
            .arg("--prefix")
            .arg(&bad_prefix)
            .arg("--prefetch-defaults")
            .env("HOME", &home)
            .env(CG_MODEL_CACHE_ROOT_ENV, &bad_cache_root)
            .env("PATH", bad_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap_or_else(|_| {
                panic!("install script spawns with mismatched fake sha for {mismatched_artifact}")
            });
        assert!(
            !bad_output.status.success(),
            "installer must fail when {mismatched_artifact} sha256sum output is {mismatched_sha} instead of the pinned production SHA\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&bad_output.stdout),
            String::from_utf8_lossy(&bad_output.stderr)
        );
        assert!(
            !bad_model_dir.exists(),
            "installer must not publish a final default model directory after {mismatched_artifact} SHA validation failure"
        );
    }
}

fn first_touch_child_if_requested() -> bool {
    let Some(root) = std::env::var_os("CG_CATALOG_FIRST_TOUCH_ROOT") else {
        return false;
    };
    let result = catalog_at(std::path::PathBuf::from(root), "host-A").fetch(
        None,
        NOMIC_ID,
        OutputMode::Interactive,
    );
    match result {
        Ok(result) => {
            if let Ok(expected) = std::env::var("CG_CATALOG_FIRST_TOUCH_EXPECT_RESULT") {
                assert_eq!(format!("{result:?}"), expected);
            }
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
    true
}

fn spawn_first_touch_child(test_name: &str, cache_root: &Path) -> Child {
    Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg(test_name)
        .env("CG_CATALOG_FIRST_TOUCH_ROOT", cache_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("first-touch child spawns")
}

fn start_nomic_source_override(root: &Path) -> (Arc<AtomicUsize>, CountedNomicFixtureServer) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(root, NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = Arc::new(AtomicUsize::new(0));
    let server_thread = serve_nomic_fixture_until_idle_counted(server, Arc::clone(&request_count));
    (request_count, server_thread)
}

struct SourceFixtureServer {
    server: Arc<tiny_http::Server>,
    shutdown: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
}

impl SourceFixtureServer {
    fn stop_and_join(&mut self) -> std::thread::Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        self.server.unblock();
        match self.server_thread.take() {
            Some(server_thread) => server_thread.join(),
            None => Ok(()),
        }
    }

    fn shutdown_and_join(mut self) {
        self.stop_and_join().expect("server thread joins");
    }
}

impl Drop for SourceFixtureServer {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    format!("{:x}", Sha256::digest(bytes))
}

fn write_source_override_with_sha_override(
    root: &Path,
    model_id: &str,
    base_url: &str,
    revision: &str,
    source_dir: &Path,
    bad_artifact: &str,
    bad_sha: &str,
) {
    let config_dir = root.join("cg");
    fs::create_dir_all(&config_dir).expect("catalog config dir created");
    let mut sha_table = toml::Table::new();
    for artifact in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        let sha = if artifact == bad_artifact {
            bad_sha.to_string()
        } else {
            fixture_sha(&source_dir.join(artifact))
        };
        sha_table.insert(artifact.to_string(), toml::Value::String(sha));
    }
    let mut source = toml::Table::new();
    source.insert(
        "base_url".to_string(),
        toml::Value::String(base_url.to_string()),
    );
    source.insert(
        "revision".to_string(),
        toml::Value::String(revision.to_string()),
    );
    source.insert("sha256".to_string(), toml::Value::Table(sha_table));
    let mut sources = toml::Table::new();
    sources.insert(model_id.to_string(), toml::Value::Table(source));
    let mut config = toml::Table::new();
    config.insert("sources".to_string(), toml::Value::Table(sources));
    fs::write(
        config_dir.join("catalog-sources.toml"),
        toml::to_string(&toml::Value::Table(config)).expect("source override serializes"),
    )
    .expect("source override written");
}

fn serve_source_fixture_counted(
    server: tiny_http::Server,
    source_dir: PathBuf,
    request_count: Arc<AtomicUsize>,
    request_paths: Arc<Mutex<Vec<String>>>,
) -> SourceFixtureServer {
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let server = Arc::new(server);
    let worker_server = Arc::clone(&server);
    let server_thread = std::thread::spawn(move || {
        while !server_shutdown.load(Ordering::SeqCst) {
            match worker_server.recv() {
                Ok(request) => {
                    request_count.fetch_add(1, Ordering::SeqCst);
                    let url = request.url().to_string();
                    request_paths
                        .lock()
                        .expect("request paths")
                        .push(url.clone());
                    let filename = match url.as_str() {
                        "/model.safetensors" => "model.safetensors",
                        "/tokenizer.json" => "tokenizer.json",
                        "/config.json" => "config.json",
                        "/REVISION" => "REVISION",
                        _ => {
                            let _ = request.respond(
                                tiny_http::Response::from_string("unexpected artifact path")
                                    .with_status_code(404),
                            );
                            continue;
                        }
                    };
                    let bytes = fs::read(source_dir.join(filename)).expect("source fixture bytes");
                    let _ = request.respond(tiny_http::Response::from_data(bytes));
                }
                Err(_) => break,
            }
        }
    });
    SourceFixtureServer {
        server,
        shutdown,
        server_thread: Some(server_thread),
    }
}

fn start_nomic_source_override_without_revision(
    root: &Path,
) -> (
    Arc<AtomicUsize>,
    Arc<Mutex<Vec<String>>>,
    SourceFixtureServer,
) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(root, NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = Arc::new(AtomicUsize::new(0));
    let request_paths = Arc::new(Mutex::new(Vec::new()));
    let count = Arc::clone(&request_count);
    let paths = Arc::clone(&request_paths);
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let server = Arc::new(server);
    let worker_server = Arc::clone(&server);
    let server_thread = std::thread::spawn(move || {
        while !server_shutdown.load(Ordering::SeqCst) {
            match worker_server.recv() {
                Ok(request) => {
                    count.fetch_add(1, Ordering::SeqCst);
                    let url = request.url().to_string();
                    paths.lock().expect("request paths").push(url.clone());
                    let filename = ["tokenizer.json", "config.json", "REVISION"]
                        .into_iter()
                        .find(|candidate| url.ends_with(candidate))
                        .unwrap_or("model.safetensors");
                    if filename == "REVISION" {
                        let _ = request.respond(
                            tiny_http::Response::from_string("missing revision")
                                .with_status_code(404),
                        );
                    } else {
                        let bytes =
                            fs::read(fixture_dir("nomic").join(filename)).expect("fixture bytes");
                        let _ = request.respond(tiny_http::Response::from_data(bytes));
                    }
                }
                Err(_) => break,
            }
        }
    });
    (
        request_count,
        request_paths,
        SourceFixtureServer {
            server,
            shutdown,
            server_thread: Some(server_thread),
        },
    )
}

fn first_stderr_line_while_child_is_in_flight(
    child: &mut Child,
    _timeout: std::time::Duration,
) -> String {
    let stderr = child.stderr.take().expect("stderr is piped");
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = BufReader::new(stderr).read_line(&mut line);
        let _ = sender.send(line);
    });
    let line = receiver.recv();
    if line.is_err() {
        let _ = child.kill();
    }
    assert!(
        line.is_ok(),
        "first-touch progress must be emitted while the fetch is still in flight"
    );
    let line = line.expect("line checked above");
    assert!(
        child.try_wait().expect("child status readable").is_none(),
        "first-touch child exited before the parent released the fetch lock"
    );
    line
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    fs::metadata(path)
        .expect("metadata readable")
        .permissions()
        .mode()
        & 0o111
        != 0
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.exists()
}

#[test]
fn prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only() {
    // Install-time prefetch should land Apache defaults while clearly skipping manual licenses.
    let temp = tempfile::tempdir().expect("tempdir");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_thread =
        serve_nomic_fixture_until_idle_counted(server, std::sync::Arc::clone(&request_count));
    let catalog = catalog_at(temp.path(), "host-A");
    let summary = catalog.prefetch_defaults().expect("prefetch returns");
    assert!(summary.fetched.iter().any(|entry| {
        entry.model_id == NOMIC_ID && entry.artifact_sha == FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA
    }));
    assert!(
        summary
            .skipped
            .iter()
            .any(|entry| entry.model_id == GEMMA_ID
                && entry.reason == SkipReason::ManualAcceptanceRequired)
    );
    server_thread.shutdown_and_join();
    assert!(
        request_count.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "prefetch_defaults should exercise the configured Apache source override"
    );
    let artifact = temp.path().join(NOMIC_ID).join("model.safetensors");
    assert!(artifact.exists(), "prefetch should land the Nomic artifact");
    assert_eq!(
        fixture_sha(&artifact),
        FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA,
        "prefetched artifact should match the catalog SHA"
    );
    assert!(
        temp.path().join(NOMIC_ID).join("catalog.toml").exists(),
        "prefetch should publish a durable cache manifest"
    );
    assert!(
        !temp.path().join(GEMMA_ID).exists(),
        "prefetch defaults must not create a gated-model cache directory"
    );
    let entries = catalog.list().expect("post-prefetch list returns");
    assert!(
        entries
            .iter()
            .any(|entry| entry.id == NOMIC_ID && entry.cache_status == CacheStatus::Cached),
        "post-prefetch list should report Nomic cached from disk"
    );
}

#[test]
fn first_touch_lazy_fetch_emits_stderr_signal_within_two_hundred_fifty_milliseconds_naming_model_and_source_and_byte_size()
 {
    // First touch on a fresh host must announce the large download before doing network work.
    if first_touch_child_if_requested() {
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let (request_count, server_thread) = start_nomic_source_override(temp.path());
    let held_lock = hold_model_lock(temp.path(), NOMIC_ID);
    let started = std::time::Instant::now();
    let mut child = spawn_first_touch_child(
        "prefetch_first_touch::first_touch_lazy_fetch_emits_stderr_signal_within_two_hundred_fifty_milliseconds_naming_model_and_source_and_byte_size",
        temp.path(),
    );
    let stderr = first_stderr_line_while_child_is_in_flight(
        &mut child,
        std::time::Duration::from_millis(250),
    );
    assert!(started.elapsed() <= std::time::Duration::from_millis(250));
    assert!(stderr.contains(NOMIC_ID));
    assert!(
        stderr.contains("546938168"),
        "stderr should print the exact byte budget, not a rounded substring"
    );
    assert!(
        stderr.contains("huggingface.co"),
        "stderr should name the resolved source host"
    );
    drop(held_lock);
    assert!(child.wait().expect("first-touch child waits").success());
    server_thread.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "first-touch fetch should use the configured local source override after the progress line"
    );
}

#[test]
fn first_touch_lazy_fetch_observer_event_payload_carries_resolved_source_and_byte_budget() {
    // Sinks need model, source, byte size, and target path without joining extra catalog state.
    let temp = tempfile::tempdir().expect("tempdir");
    let expected_observation_shape = FetchObservation {
        model_id: NOMIC_ID.to_string(),
        observation_type: "model_fetch_starting".to_string(),
        state_delta: serde_json::json!({
            "model_id": NOMIC_ID,
            "expected_bytes": 546_938_168_u64,
            "target_path": temp.path().join(NOMIC_ID),
        }),
    };
    assert_eq!(expected_observation_shape.model_id, NOMIC_ID);
    assert_eq!(
        expected_observation_shape
            .state_delta
            .get("expected_bytes")
            .and_then(serde_json::Value::as_u64),
        Some(546_938_168)
    );
    let (request_count, server_thread) = start_nomic_source_override(temp.path());
    let held_lock = hold_model_lock(temp.path(), NOMIC_ID);
    let (event_tx, event_rx) = mpsc::channel();
    let fetch_call_started = Arc::new(AtomicBool::new(false));
    let observer_fetch_call_started = Arc::clone(&fetch_call_started);
    let catalog = catalog_at(temp.path(), "host-A").with_progress_observer(move |event| {
        let _ = event_tx.send((
            Instant::now(),
            observer_fetch_call_started.load(Ordering::SeqCst),
            event.clone(),
        ));
    });
    assert!(
        event_rx.try_recv().is_err(),
        "observer registration must not emit a canned fetch event before fetch is called"
    );
    let (call_tx, call_rx) = mpsc::channel();
    let fetch = std::thread::spawn(move || {
        fetch_call_started.store(true, Ordering::SeqCst);
        let _ = call_tx.send(Instant::now());
        catalog.fetch(None, NOMIC_ID, OutputMode::Interactive)
    });
    let started = call_rx
        .recv()
        .expect("fetch thread reaches catalog.fetch call");
    let event = event_rx.recv();
    assert!(
        event.is_ok(),
        "observer receives progress event before fetch can complete"
    );
    let (observed_at, emitted_after_fetch_call, event) =
        event.expect("asserted observer event is present");
    assert!(
        emitted_after_fetch_call,
        "observer event must be emitted after fetch is called, not during registration"
    );
    assert!(
        observed_at.duration_since(started) <= Duration::from_millis(250),
        "observer event should arrive within the first-touch latency budget"
    );
    match event {
        ProgressEvent::FetchStarting {
            model_id,
            source,
            expected_bytes,
            target_path,
        } => {
            assert_eq!(model_id, NOMIC_ID);
            assert_eq!(expected_bytes, 546_938_168);
            assert_eq!(target_path, temp.path().join(NOMIC_ID));
            assert!(matches!(
                source,
                Source::HuggingFace { repo, revision }
                    if repo == "nomic-ai/nomic-embed-text-v1.5" && revision == NOMIC_REVISION
            ));
        }
    }
    assert!(
        !fetch.is_finished(),
        "observer event must be emitted before the lock-blocked fetch completes"
    );
    drop(held_lock);
    assert_eq!(
        fetch.join().expect("fetch thread joins"),
        Ok(FetchResult::Completed)
    );
    server_thread.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "first-touch fetch should complete through the local source override, not the network"
    );
}

#[test]
fn install_script_writes_uninstall_at_known_absolute_path_and_prints_bundled_default_name_and_size_and_prefetch_cta()
 {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    let cache_root = home.join(".cache/cg/models");
    fs::create_dir_all(&home).expect("home created");

    let output = run_install(&home, &prefix, &cache_root, &["--dry-run"]);
    assert_success(&output, "install --dry-run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(NOMIC_ID),
        "dry-run install output should name the bundled default model"
    );
    assert!(
        stdout.contains("546938168"),
        "dry-run install output should print the exact default byte count"
    );
    assert!(
        stdout.contains("cg model fetch --defaults"),
        "dry-run install output should include the prefetch CTA"
    );

    let uninstall = prefix.join("cg-uninstall.sh");
    assert!(
        uninstall.exists(),
        "install should write paired uninstall script"
    );
    assert!(
        is_executable(&uninstall),
        "paired uninstall script should be executable"
    );
    let content = fs::read_to_string(&uninstall).expect("uninstall script readable");
    assert!(
        content.contains(&cache_root.display().to_string()),
        "uninstall script should pin the exact cache root path"
    );
    assert!(
        content.contains(&prefix.join("bin/cg").display().to_string()),
        "uninstall script should pin the exact cg binary path"
    );
    assert!(
        content.contains("# BEGIN cg-injected-hook-"),
        "uninstall script should carry the cg hook marker prefix used for surgical removal"
    );
}

#[test]
fn install_script_prefetch_defaults_flag_drives_apache_model_download_before_first_use() {
    if first_touch_child_if_requested() {
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    let cache_root = temp.path().join("cache");
    fs::create_dir_all(&home).expect("home created");

    let install_server = tiny_http::Server::http("127.0.0.1:0").expect("install stub server");
    let install_base_url = stub_base_url(&install_server);
    let configured_revision = "installer-configured-revision-trap";
    let source_revision = "installer-source-revision-not-built-in";
    let source_revision_sha = sha256_bytes(source_revision.as_bytes());
    let source_fixture = temp.path().join("install-source-fixture");
    fs::create_dir_all(&source_fixture).expect("install source fixture dir created");
    fs::write(
        source_fixture.join("model.safetensors"),
        b"installer-source-model-bytes-not-local-fixture",
    )
    .expect("source model artifact written");
    fs::write(
        source_fixture.join("tokenizer.json"),
        br#"{"source":"installer-source-tokenizer-not-local-fixture"}"#,
    )
    .expect("source tokenizer artifact written");
    fs::write(
        source_fixture.join("config.json"),
        br#"{"source":"installer-source-config-not-local-fixture"}"#,
    )
    .expect("source config artifact written");
    fs::write(source_fixture.join("REVISION"), source_revision).expect("source revision written");
    write_source_override_with_actual_shas(
        &cache_root,
        NOMIC_ID,
        &install_base_url,
        configured_revision,
        &source_fixture,
    );
    let install_request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let install_request_paths = Arc::new(Mutex::new(Vec::new()));
    let install_server_thread = serve_source_fixture_counted(
        install_server,
        source_fixture.clone(),
        std::sync::Arc::clone(&install_request_count),
        Arc::clone(&install_request_paths),
    );
    let output = run_install(&home, &prefix, &cache_root, &["--prefetch-defaults"]);
    assert_success(&output, "install --prefetch-defaults");
    install_server_thread.shutdown_and_join();
    assert!(
        install_request_count.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "--prefetch-defaults should fetch through the configured source override"
    );
    let requested_paths = install_request_paths
        .lock()
        .expect("install request paths")
        .clone();
    assert_eq!(
        requested_paths,
        vec![
            "/model.safetensors",
            "/tokenizer.json",
            "/config.json",
            "/REVISION"
        ],
        "--prefetch-defaults must request every artifact at the exact configured source paths"
    );

    let model_dir = cache_root.join(NOMIC_ID);
    for artifact in ["model.safetensors", "tokenizer.json", "config.json"] {
        let artifact_path = model_dir.join(artifact);
        assert!(
            artifact_path.exists(),
            "--prefetch-defaults should land {artifact} before first use"
        );
        assert_eq!(
            fixture_sha(&artifact_path),
            fixture_sha(&source_fixture.join(artifact)),
            "prefetched {artifact} should match the configured source SHA"
        );
        assert_eq!(
            fs::read(&artifact_path).expect("prefetched artifact readable"),
            fs::read(source_fixture.join(artifact)).expect("source artifact readable"),
            "prefetched {artifact} should be byte-for-byte source content"
        );
    }
    let revision_artifact = model_dir.join("REVISION");
    assert!(
        revision_artifact.exists(),
        "--prefetch-defaults must fetch REVISION from the configured source"
    );
    assert_eq!(
        fixture_sha(&revision_artifact),
        source_revision_sha,
        "prefetched REVISION should use the configured source bytes, not synthesized built-in revision text"
    );
    assert_eq!(
        fs::read_to_string(&revision_artifact).expect("prefetched revision readable"),
        source_revision,
        "prefetched REVISION should be byte-for-byte source content"
    );

    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override_with_actual_shas(
        &cache_root,
        NOMIC_ID,
        &base_url,
        source_revision,
        &source_fixture,
    );
    let request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_thread =
        serve_nomic_fixture_until_idle_counted(server, std::sync::Arc::clone(&request_count));
    let output = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("prefetch_first_touch::install_script_prefetch_defaults_flag_drives_apache_model_download_before_first_use")
        .env("CG_CATALOG_FIRST_TOUCH_ROOT", &cache_root)
        .env("CG_CATALOG_FIRST_TOUCH_EXPECT_RESULT", "AlreadyCached")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("warm first-use child runs");
    assert!(
        output.status.success(),
        "warm first-use child should observe AlreadyCached\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "prefetched first use must not emit lazy-fetch progress"
    );
    server_thread.shutdown_and_join();
    assert_eq!(
        request_count.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "prefetched first use should not perform outbound HTTP"
    );

    for (artifact, bad_sha) in [
        (
            "model.safetensors",
            "1111111111111111111111111111111111111111111111111111111111111111",
        ),
        (
            "tokenizer.json",
            "2222222222222222222222222222222222222222222222222222222222222222",
        ),
        (
            "config.json",
            "3333333333333333333333333333333333333333333333333333333333333333",
        ),
        (
            "REVISION",
            "4444444444444444444444444444444444444444444444444444444444444444",
        ),
        ("model.safetensors", ALL_ZERO_SHA),
        ("tokenizer.json", ALL_ZERO_SHA),
        ("config.json", ALL_ZERO_SHA),
        ("REVISION", ALL_ZERO_SHA),
    ] {
        let case_name = if bad_sha == ALL_ZERO_SHA {
            format!("{}-zero-sha", artifact.replace('.', "-"))
        } else {
            format!("{}-bad-sha", artifact.replace('.', "-"))
        };
        let bad_cache_root = temp.path().join(format!("source-override-{case_name}"));
        let bad_prefix = temp.path().join(format!("source-prefix-{case_name}"));
        let bad_server =
            tiny_http::Server::http("127.0.0.1:0").expect("bad source override server");
        let bad_base_url = stub_base_url(&bad_server);
        write_source_override_with_sha_override(
            &bad_cache_root,
            NOMIC_ID,
            &bad_base_url,
            configured_revision,
            &source_fixture,
            artifact,
            bad_sha,
        );
        let bad_request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let bad_request_paths = Arc::new(Mutex::new(Vec::new()));
        let bad_server_thread = serve_source_fixture_counted(
            bad_server,
            source_fixture.clone(),
            std::sync::Arc::clone(&bad_request_count),
            Arc::clone(&bad_request_paths),
        );
        let bad_output = run_install(
            &home,
            &bad_prefix,
            &bad_cache_root,
            &["--prefetch-defaults"],
        );
        bad_server_thread.shutdown_and_join();
        assert!(
            bad_request_count.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "--prefetch-defaults should contact the configured source before failing {artifact} SHA validation"
        );
        assert!(
            !bad_output.status.success(),
            "source override prefetch must fail when configured {artifact} SHA is {bad_sha}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&bad_output.stdout),
            String::from_utf8_lossy(&bad_output.stderr)
        );
        assert!(
            !bad_cache_root.join(NOMIC_ID).exists(),
            "source override prefetch must not publish the final model directory after configured {artifact} SHA validation failure"
        );
    }
}

#[test]
fn install_script_prefetch_defaults_fails_when_source_override_revision_artifact_is_missing() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    let cache_root = temp.path().join("cache");
    fs::create_dir_all(&home).expect("home created");

    let (request_count, request_paths, server_thread) =
        start_nomic_source_override_without_revision(&cache_root);
    let output = run_install(&home, &prefix, &cache_root, &["--prefetch-defaults"]);
    server_thread.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "--prefetch-defaults should contact the configured source before failing"
    );
    assert!(
        request_paths
            .lock()
            .expect("request paths")
            .iter()
            .any(|path| path.ends_with("REVISION")),
        "--prefetch-defaults must request REVISION from the configured source"
    );
    assert!(
        !output.status.success(),
        "missing REVISION must fail install prefetch instead of synthesizing one\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let model_dir = cache_root.join(NOMIC_ID);
    assert!(
        !model_dir.exists(),
        "source override failure before complete validation must not publish the final model directory or catalog.toml"
    );
    for artifact in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        assert!(
            !model_dir.join(artifact).exists(),
            "source override failure before complete validation must not publish partial final artifact {artifact}"
        );
    }
}
