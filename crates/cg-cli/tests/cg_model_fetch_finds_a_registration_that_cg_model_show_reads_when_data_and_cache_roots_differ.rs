//! `cg model show` and `cg model fetch` must agree on where a registration lives.
//!
//! ## The promise
//!
//! An operator who points `CG_DATA_ROOT` and `CG_MODEL_CACHE_ROOT` at two different directories
//! (the documented shape — registrations are data, fetched artifact bytes are cache) still gets a
//! `cg model fetch` that finds the registration `cg model show` already reads. An
//! already-available API-endpoint registration answers `AlreadyCached` (exit 0); a registration
//! whose license requires manual acceptance answers with its manual-install instructions, never
//! `NotFound`.
//!
//! Inline TOML fixtures keep the registration checks independent of network access.

use std::fs;
use std::path::Path;
use std::process::Command;

const API_ENDPOINT_ENTRY: &str = r#"
[[entries]]
id = "offline-fixture-api-endpoint-v1"
kind = "TextEmbedder"
capability = { dimension = 768, output_shape = "vector", input_modality = "text" }
license = { spdx = "Apache-2.0", url = "https://www.apache.org/licenses/LICENSE-2.0", requires_manual_acceptance = false }
source = { ApiEndpoint = { base_url = "https://example.invalid/v1", provider = "fixture", model = "fixture-api", auth_env_var = "FIXTURE_API_KEY" } }
byte_size = 0
cache_status = "NotCached"
local_path = "/tmp/offline-fixture-api-endpoint-v1"
install_instructions = "fixture API endpoint; no artifacts are downloaded"
audit_footnote = "fixture only"
acceptance_recorded = false
manual_install_url = "https://example.invalid/api-install"
artifact_filenames = []
per_file_sha256 = []
"#;

const MANUAL_LICENCE_ENTRY: &str = r#"
[[entries]]
id = "offline-fixture-manual-v1"
kind = "TextEmbedder"
capability = { dimension = 768, output_shape = "vector", input_modality = "text" }
license = { spdx = "LicenseRef-Fixture", url = "https://example.invalid/license", requires_manual_acceptance = true }
source = { LocalPath = { dir = "/tmp/offline-fixture-manual-v1" } }
byte_size = 1
cache_status = "NotCached"
local_path = "/tmp/offline-fixture-manual-v1"
install_instructions = "Install offline-fixture-manual-v1 from the approved fixture source."
audit_footnote = "fixture only"
acceptance_recorded = false
manual_install_url = "https://example.invalid/manual-install"
artifact_filenames = ["model.safetensors"]
per_file_sha256 = []
"#;

fn cg_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_cg"))
}

/// Two disposable roots: registrations live under `data_root`, fetched artifact bytes would land
/// under `model_cache_root` — the documented shape an operator who separates the two is entitled
/// to rely on.
struct DisjointRoots {
    data_root: tempfile::TempDir,
    model_cache_root: tempfile::TempDir,
}

impl DisjointRoots {
    fn new(entries_toml: &str) -> Self {
        let data_root = tempfile::TempDir::new().expect("data root");
        let model_cache_root = tempfile::TempDir::new().expect("model cache root");
        let cg_dir = data_root.path().join("cg");
        fs::create_dir_all(&cg_dir).expect("create data root cg dir");
        fs::write(cg_dir.join("catalog-registrations.toml"), entries_toml)
            .expect("write catalog registrations fixture");
        Self {
            data_root,
            model_cache_root,
        }
    }

    fn run(&self, args: &[&str]) -> (i32, String, String) {
        run_cg_with_roots(args, self.data_root.path(), self.model_cache_root.path())
    }
}

fn run_cg_with_roots(
    args: &[&str],
    data_root: &Path,
    model_cache_root: &Path,
) -> (i32, String, String) {
    let output = Command::new(cg_bin())
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("CG_MODEL_CACHE_ROOT", model_cache_root)
        .env("HOME", data_root)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .output()
        .expect("spawn compiled cg binary");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn show_reads_the_api_endpoint_registration_from_the_data_root() {
    let roots = DisjointRoots::new(API_ENDPOINT_ENTRY);
    let (code, stdout, stderr) = roots.run(&[
        "model",
        "show",
        "offline-fixture-api-endpoint-v1",
        "--format",
        "json",
    ]);
    assert_eq!(
        code, 0,
        "show must find a registration stored at CG_DATA_ROOT regardless of \
         CG_MODEL_CACHE_ROOT; stdout={stdout}, stderr={stderr}"
    );
}

#[test]
fn fetch_answers_already_cached_for_an_api_endpoint_registration_when_the_roots_differ() {
    let roots = DisjointRoots::new(API_ENDPOINT_ENTRY);
    let (code, stdout, stderr) = roots.run(&[
        "model",
        "fetch",
        "offline-fixture-api-endpoint-v1",
        "--format",
        "json",
    ]);
    assert_eq!(
        code, 0,
        "an API-endpoint registration that show already finds must fetch as AlreadyCached, not \
         NotFound, when CG_MODEL_CACHE_ROOT differs from CG_DATA_ROOT; stdout={stdout}, \
         stderr={stderr}"
    );
    assert!(
        stdout.contains("AlreadyCached"),
        "expected AlreadyCached in fetch output, got stdout={stdout}, stderr={stderr}"
    );
    assert!(
        !stdout.contains("NotFound") && !stderr.contains("NotFound"),
        "fetch must never answer NotFound for a registration show can read; \
         stdout={stdout}, stderr={stderr}"
    );
}

#[test]
fn show_reads_the_manual_licence_registration_from_the_data_root() {
    let roots = DisjointRoots::new(MANUAL_LICENCE_ENTRY);
    let (code, stdout, stderr) = roots.run(&[
        "model",
        "show",
        "offline-fixture-manual-v1",
        "--format",
        "json",
    ]);
    assert_eq!(
        code, 0,
        "show must find a registration stored at CG_DATA_ROOT regardless of \
         CG_MODEL_CACHE_ROOT; stdout={stdout}, stderr={stderr}"
    );
}

#[test]
fn fetch_gives_manual_install_instructions_for_a_manual_licence_registration_when_the_roots_differ()
{
    let roots = DisjointRoots::new(MANUAL_LICENCE_ENTRY);
    let (_code, stdout, stderr) = roots.run(&[
        "model",
        "fetch",
        "offline-fixture-manual-v1",
        "--format",
        "json",
    ]);
    assert!(
        stdout.contains("ManualAcceptanceRequired") || stdout.contains("manual"),
        "a manual-licence registration that show already finds must give its manual-install \
         instructions when CG_MODEL_CACHE_ROOT differs from CG_DATA_ROOT, never NotFound; \
         stdout={stdout}, stderr={stderr}"
    );
    assert!(
        !stdout.contains("NotFound") && !stderr.contains("NotFound"),
        "fetch must never answer NotFound for a registration show can read; \
         stdout={stdout}, stderr={stderr}"
    );
}
