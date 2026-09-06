//! The four checked event-search entrypoints (`prepare_document`, `prepare_query`,
//! `embed_prepared_document`, `embed_prepared_query`) are callable on any embedder handle a
//! caller happens to be holding — the bundled Nomic embedder's own concrete type included, not
//! only `&dyn Embedder` / `Arc<dyn Embedder>`. This file imports the sealed extension trait that
//! makes that true, `context_graph::EventSearchEmbedder`, and every assertion below runs
//! identically no matter which handle shape it is called through.
//!
//! This file never touches the real `~/.cache/cg`: every test points `HOME` at a disposable temp
//! directory before copying the already-installed canonical fixture into it, exactly as
//! `tests/catalog_and_offline_nomic_embedder_disagree_on_the_model_cache_root.rs` does.

use std::env::VarError;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::{CgError, Embedder, EmbedderConfig, EventSearchEmbedder};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";

/// The real `HOME`, captured once before any test in this file overrides the process
/// environment — mirrors `catalog_and_offline_nomic_embedder_disagree_on_the_model_cache_root.rs`.
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
        "the destination must never be the real canonical fixture directory itself — this file \
         overrides HOME to a disposable temp directory precisely so this can never happen"
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

/// Serializes every test in this file that mutates the process environment, restoring the exact
/// prior state (including "was unset") on drop.
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
        // serializes every HOME mutation in this file.
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

fn isolated_offline_nomic_embedder() -> (Arc<NomicTextEmbedder>, tempfile::TempDir, EnvGuard) {
    let source = canonical_fixture_dir();
    let tmp = tempfile::tempdir().expect("tempdir");
    let fake_home = tmp.path().join("fake-home");
    std::fs::create_dir_all(&fake_home).expect("create fake home dir");
    let guard = EnvGuard::set(&[
        ("HOME", Some(fake_home.to_str().unwrap())),
        ("CG_MODEL_CACHE_ROOT", None),
        ("XDG_CACHE_HOME", None),
    ]);
    let destination = fake_home.join(".cache/cg/models").join(NOMIC_MODEL_ID);
    copy_canonical_fixture_into(&source, &destination);
    let embedder = Arc::new(NomicTextEmbedder::new(
        EmbedderConfig::default_nomic_offline(),
    ));
    (embedder, tmp, guard)
}

/// The four checked event-search entrypoints answer identically, byte for byte,
/// whichever handle shape the caller happens to be holding.
///
/// (a) directly on the concrete `Arc<NomicTextEmbedder>` handle;
/// (b) on `&dyn Embedder`;
/// (c) on `Arc<dyn Embedder>`.
///
/// A `compile_fail` doctest cannot express that the extension trait is sealed (integration tests
/// don't support `compile_fail`), so sealing is instead pinned by the public-surface guard added
/// in the paired commit (`tests/acceptance/nomic_loads.rs`): it asserts `embedder.rs` declares
/// `pub trait EventSearchEmbedder` with a private `Sealed` supertrait and exactly one blanket
/// impl, which is the shape that makes an outside implementation of `EventSearchEmbedder`
/// impossible.
#[test]
fn the_four_role_methods_give_byte_identical_results_across_every_embedder_handle_route() {
    let (nomic, _tmp, _guard) = isolated_offline_nomic_embedder();

    // (a) directly on the concrete handle — this is the call the ruling requires and the one
    // that cannot compile until `EventSearchEmbedder` exists.
    let doc_a = nomic
        .prepare_document("the incident review closed today")
        .expect("route (a) prepare_document must succeed");
    let query_a = nomic
        .prepare_query("find the incident review")
        .expect("route (a) prepare_query must succeed");
    let embed_doc_a = nomic
        .embed_prepared_document(&doc_a)
        .expect("route (a) embed_prepared_document must succeed");
    let embed_query_a = nomic
        .embed_prepared_query(&query_a)
        .expect("route (a) embed_prepared_query must succeed");

    // (b) through `&dyn Embedder`.
    let as_trait_object: &dyn Embedder = nomic.as_ref();
    let doc_b = as_trait_object
        .prepare_document("the incident review closed today")
        .expect("route (b) prepare_document must succeed");
    let query_b = as_trait_object
        .prepare_query("find the incident review")
        .expect("route (b) prepare_query must succeed");
    let embed_doc_b = as_trait_object
        .embed_prepared_document(&doc_b)
        .expect("route (b) embed_prepared_document must succeed");
    let embed_query_b = as_trait_object
        .embed_prepared_query(&query_b)
        .expect("route (b) embed_prepared_query must succeed");

    // (c) through `Arc<dyn Embedder>`.
    let as_arc_trait_object: Arc<dyn Embedder> = nomic.clone();
    let doc_c = as_arc_trait_object
        .prepare_document("the incident review closed today")
        .expect("route (c) prepare_document must succeed");
    let query_c = as_arc_trait_object
        .prepare_query("find the incident review")
        .expect("route (c) prepare_query must succeed");
    let embed_doc_c = as_arc_trait_object
        .embed_prepared_document(&doc_c)
        .expect("route (c) embed_prepared_document must succeed");
    let embed_query_c = as_arc_trait_object
        .embed_prepared_query(&query_c)
        .expect("route (c) embed_prepared_query must succeed");

    for (route, doc) in [("b", &doc_b), ("c", &doc_c)] {
        assert_eq!(
            doc_a.token_count(),
            doc.token_count(),
            "route {route} prepared-document token_count must equal route (a)'s"
        );
        assert_eq!(
            doc_a.byte_boundaries(),
            doc.byte_boundaries(),
            "route {route} prepared-document byte_boundaries must equal route (a)'s"
        );
        assert_eq!(
            doc_a.encoding_profile_id(),
            doc.encoding_profile_id(),
            "route {route} prepared-document encoding_profile_id must equal route (a)'s"
        );
    }
    for (route, query) in [("b", &query_b), ("c", &query_c)] {
        assert_eq!(
            query_a.token_count(),
            query.token_count(),
            "route {route} prepared-query token_count must equal route (a)'s"
        );
        assert_eq!(
            query_a.encoding_profile_id(),
            query.encoding_profile_id(),
            "route {route} prepared-query encoding_profile_id must equal route (a)'s"
        );
    }
    assert_eq!(
        embed_doc_a, embed_doc_b,
        "route (b) embed_prepared_document output must be byte-identical to route (a)'s"
    );
    assert_eq!(
        embed_doc_a, embed_doc_c,
        "route (c) embed_prepared_document output must be byte-identical to route (a)'s"
    );
    assert_eq!(
        embed_query_a, embed_query_b,
        "route (b) embed_prepared_query output must be byte-identical to route (a)'s"
    );
    assert_eq!(
        embed_query_a, embed_query_c,
        "route (c) embed_prepared_query output must be byte-identical to route (a)'s"
    );
}

/// A non-canonical external embedder — the crate's own test-support double, never Nomic — still
/// answers `EventSearchPreparationUnsupported` when the four methods are called directly on its
/// concrete handle (route (a)), the same way it already does through `&dyn Embedder`: the
/// blanket `EventSearchEmbedder` impl adds a calling convention, never a new capability.
#[test]
fn a_noncanonical_external_embedder_answers_unsupported_directly_on_its_concrete_handle() {
    let mock = LexicalMockEmbedder::default();

    let document_result = mock.prepare_document("the incident review closed today");
    assert!(
        matches!(
            document_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "a non-canonical external embedder's concrete handle must answer \
         EventSearchPreparationUnsupported from prepare_document, never succeed or fall back to \
         plain embedding; got {document_result:?}"
    );

    let query_result = mock.prepare_query("find the incident review");
    assert!(
        matches!(
            query_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "a non-canonical external embedder's concrete handle must answer \
         EventSearchPreparationUnsupported from prepare_query, never succeed or fall back to \
         plain embedding; got {query_result:?}"
    );
}
