//! `cg model fetch` seeds the catalog's model cache root
//! (`cg_catalog::default_cache_root()`); the offline event-history embedder loads from its own
//! private resolver in `context_graph::embedder::nomic`. Both are supposed to name the same
//! directory for a given environment, including under `XDG_CACHE_HOME` and
//! `CG_MODEL_CACHE_ROOT`, so that a model fetched once is loadable offline afterward without a
//! second download or a hidden path override.
//!
//! This test places a copy of the already-installed canonical fixture
//! (`~/.cache/cg/models/nomic-embed-text-v1.5`) at the exact directory
//! `cg_catalog::default_cache_root()` names, then asks the public offline embedder constructor
//! to load from the environment the catalog resolved against. Both resolvers must agree under
//! every supported override, `CG_MODEL_CACHE_ROOT` included and `XDG_CACHE_HOME` alone (no
//! `CG_MODEL_CACHE_ROOT` override) included — a model `cg model fetch` places at the directory
//! the catalog names must always be the same directory the offline embedder loads from, never a
//! second, resolver-specific location reachable only through the network.

use std::env::VarError;
use std::ffi::OsString;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::{Embedder, EmbedderConfig, EmbeddingInput};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";

/// The real `HOME`, captured once before any test in this file overrides the process
/// environment — every test below points `HOME` at a disposable temp directory so that a
/// resolver falling back to `$HOME/.cache/cg/models` can never write into, or read stale bytes
/// out of, this host's actual canonical fixture cache. Only [`copy_canonical_fixture_into`]
/// reads from this real path, and it never writes to it.
fn real_home() -> std::path::PathBuf {
    static REAL_HOME: OnceLock<std::path::PathBuf> = OnceLock::new();
    REAL_HOME
        .get_or_init(|| {
            std::path::PathBuf::from(
                std::env::var_os("HOME")
                    .expect("HOME must be set to locate the canonical cache before any override"),
            )
        })
        .clone()
}

/// The canonical, already-installed fixture directory this host's production resolver names,
/// captured once — while `real_home()` is captured, before any test in this file overrides
/// `HOME`/`XDG_CACHE_HOME`/`CG_MODEL_CACHE_ROOT` — so a later override can never change which
/// directory this proof reads its starting fixture from.
fn canonical_fixture_dir() -> std::path::PathBuf {
    static CANONICAL_FIXTURE_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    CANONICAL_FIXTURE_DIR
        .get_or_init(nomic_real_model::canonical_model_dir)
        .clone()
}

fn copy_canonical_fixture_into(source: &Path, destination: &Path) {
    assert!(
        source.join("model.safetensors").is_file(),
        "the canonical Nomic fixture must already be installed at {} for this proof to run \
         offline; a missing cache is BLOCKED, not a reason to download",
        source.display()
    );
    assert_ne!(
        source, destination,
        "the destination must never be the real canonical fixture directory itself — every test \
         in this file overrides HOME to a disposable temp directory precisely so this can never \
         happen"
    );
    std::fs::create_dir_all(destination).expect("create destination model directory");
    for file in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        std::fs::copy(source.join(file), destination.join(file))
            .unwrap_or_else(|err| panic!("copy canonical fixture file {file}: {err}"));
    }
}

/// Serializes every test in this file that mutates the process environment, restoring the
/// exact prior state (including "was unset") on drop.
struct EnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Result<String, VarError>)>,
}

impl EnvGuard {
    fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Force `real_home()` and `canonical_fixture_dir()` to capture the process's real HOME
        // and the production-resolved fixture directory now, while still holding the lock that
        // serializes every HOME mutation in this file — this is the only place HOME is ever set,
        // so this is the last point at which HOME, and the resolver `canonical_fixture_dir()`
        // depends on, are guaranteed unmodified.
        let _ = real_home();
        let _ = canonical_fixture_dir();
        let keys: Vec<&'static str> = vars.iter().map(|(key, _)| *key).collect();
        let saved = keys.iter().map(|key| (*key, std::env::var(key))).collect();
        for (key, value) in vars {
            // SAFETY: serialized by `LOCK` above; restored by `Drop` below.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        EnvGuard { _lock: lock, saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.iter().rev() {
            // SAFETY: see `EnvGuard::set`.
            unsafe {
                match value {
                    Ok(value) => std::env::set_var(key, value),
                    Err(_) => std::env::remove_var(key),
                }
            }
        }
    }
}

fn assert_os(value: Option<OsString>, expected: &Path) {
    assert_eq!(
        value.as_deref().map(Path::new),
        Some(expected),
        "expected the resolved directory to equal the destination fixture was copied into"
    );
}

#[test]
fn catalog_and_offline_embedder_agree_under_cg_model_cache_root_override() {
    let source = canonical_fixture_dir();
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("override-root");
    let fake_home = tmp.path().join("fake-home");
    std::fs::create_dir_all(&fake_home).expect("create fake home dir");
    let _guard = EnvGuard::set(&[
        ("HOME", Some(fake_home.to_str().unwrap())),
        ("CG_MODEL_CACHE_ROOT", Some(root.to_str().unwrap())),
        ("XDG_CACHE_HOME", None),
    ]);

    let catalog_root = cg_catalog::default_cache_root();
    assert_eq!(
        catalog_root, root,
        "the catalog resolver must honor CG_MODEL_CACHE_ROOT verbatim"
    );

    copy_canonical_fixture_into(&source, &catalog_root.join(NOMIC_MODEL_ID));

    let embedder = NomicTextEmbedder::new(EmbedderConfig::default_nomic_offline());
    let result = embedder.embed(EmbeddingInput::Text("hello".to_string()));
    assert!(
        result.is_ok(),
        "under CG_MODEL_CACHE_ROOT the offline embedder must load the model the catalog just \
         placed at {}: {:?}",
        catalog_root.join(NOMIC_MODEL_ID).display(),
        result.err()
    );
    assert_os(Some(catalog_root.clone().into()), &root);
}

/// `cg model fetch` seeds the catalog's resolved cache root, and the offline embedder must
/// load from that SAME root, including when only `XDG_CACHE_HOME` (no `CG_MODEL_CACHE_ROOT`
/// override) distinguishes it from the ordinary default: a model fetched into the catalog's
/// directory must never be invisible to the offline embedder under this environment.
#[test]
fn catalog_and_offline_embedder_agree_under_xdg_cache_home_alone() {
    let source = canonical_fixture_dir();
    let tmp = tempfile::tempdir().expect("tempdir");
    let xdg = tmp.path().join("xdg-cache");
    let fake_home = tmp.path().join("fake-home");
    std::fs::create_dir_all(&xdg).expect("create xdg cache dir");
    std::fs::create_dir_all(&fake_home).expect("create fake home dir");
    let _guard = EnvGuard::set(&[
        ("HOME", Some(fake_home.to_str().unwrap())),
        ("CG_MODEL_CACHE_ROOT", None),
        ("XDG_CACHE_HOME", Some(xdg.to_str().unwrap())),
    ]);

    // Simulate `cg model fetch --defaults`: it seeds wherever the catalog resolver names.
    let catalog_root = cg_catalog::default_cache_root();
    copy_canonical_fixture_into(&source, &catalog_root.join(NOMIC_MODEL_ID));

    // The offline embedder must load from that same fetched location.
    let embedder = NomicTextEmbedder::new(EmbedderConfig::default_nomic_offline());
    let result = embedder.embed(EmbeddingInput::Text("hello".to_string()));
    assert!(
        result.is_ok(),
        "the offline embedder must load the model `cg model fetch` placed at {} under \
         XDG_CACHE_HOME, the same directory the catalog resolver named; instead it answered: \
         {:?} — the catalog and embedder cache-root resolvers disagree under XDG_CACHE_HOME",
        catalog_root.join(NOMIC_MODEL_ID).display(),
        result.err()
    );
}
