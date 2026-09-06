//! The download-enabled default Nomic configuration must answer for the event-history role the
//! same way the offline configuration does when nothing is installed yet: a declared capability,
//! and a typed "not installed" refusal naming the exact missing directory — never "unsupported",
//! and never a fetch. A host that checks `event_search_capability()` at startup must see `Some`
//! regardless of whether its configuration is allowed to fetch later; only the state of the
//! installed artifact decides support, never the fetch policy.
//!
//! The same holds one stage later. A prepared document or query is an opaque value a caller can
//! hold across process boundaries and hand back at embed time, so `embed_prepared_document` and
//! `embed_prepared_query` are reached on handles that were never asked to prepare anything. On a
//! handle whose model is not installed, those two must answer the same typed "not installed"
//! refusal naming the same directory — a prepared value is never the thing that starts a download.
//! A user who never consented to a fetch must not get one because a stored prepared value came
//! back to a machine with an empty cache.
//!
//! This file never touches the real `~/.cache/cg` for the embedder under test: every test points
//! `HOME` and `CG_MODEL_CACHE_ROOT` at disposable, empty temp directories before constructing it,
//! exactly as `tests/catalog_and_offline_nomic_embedder_disagree_on_the_model_cache_root.rs` does.
//! Where a test needs a genuinely sealed prepared value to hand over, it mints one from the
//! already-installed canonical model FIRST, by absolute path, before any environment is redirected.

use std::env::VarError;
use std::sync::{Arc, Mutex, OnceLock};

use context_graph::embedder::load_observer::{EmbedderLoadEvent, EmbedderLoadObserver};
use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::{CgError, Embedder, EmbedderConfig, EventSearchEmbedder};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

/// Serializes every test in this file that mutates the process environment, restoring the exact
/// prior state (including "was unset") on drop — mirrors
/// `catalog_and_offline_nomic_embedder_disagree_on_the_model_cache_root.rs`'s `EnvGuard`.
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

#[derive(Default)]
struct RecordingLoadObserver {
    events: Mutex<Vec<EmbedderLoadEvent>>,
}

impl EmbedderLoadObserver for RecordingLoadObserver {
    fn observe(&self, event: EmbedderLoadEvent) {
        self.events
            .lock()
            .expect("recording observer lock")
            .push(event);
    }
}

impl RecordingLoadObserver {
    fn saw_any_fetch(&self) -> bool {
        self.events
            .lock()
            .expect("recording observer lock")
            .iter()
            .any(|event| {
                matches!(
                    event,
                    EmbedderLoadEvent::FetchStarted { .. }
                        | EmbedderLoadEvent::FetchCompleted { .. }
                )
            })
    }
}

#[test]
fn default_nomic_configuration_reports_a_missing_model_as_not_installed_and_never_fetches() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache_root = tmp.path().join("empty-model-cache");
    let fake_home = tmp.path().join("fake-home");
    std::fs::create_dir_all(&cache_root).expect("create empty cache root");
    std::fs::create_dir_all(&fake_home).expect("create fake home dir");
    let _guard = EnvGuard::set(&[
        ("HOME", Some(fake_home.to_str().unwrap())),
        ("CG_MODEL_CACHE_ROOT", Some(cache_root.to_str().unwrap())),
    ]);

    let resolved_model_dir = cache_root.join("nomic-embed-text-v1.5");
    let observer = Arc::new(RecordingLoadObserver::default());
    let model = NomicTextEmbedder::new_with_load_observer(
        EmbedderConfig::default_nomic(),
        observer.clone(),
    );
    let embedder: &dyn Embedder = &model;

    assert!(
        embedder.event_search_capability().is_some(),
        "the download-enabled default configuration must declare the event-history capability \
         even with nothing installed yet, exactly as the offline configuration does; only the \
         installed artifact's state decides support, never whether the configuration is allowed \
         to fetch"
    );

    let document_result = embedder.prepare_document("probe text over an empty cache root");
    assert!(
        matches!(
            &document_result,
            Err(CgError::EmbedderModelNotAvailable { path, .. })
                if path == &resolved_model_dir.display().to_string()
        ),
        "prepare_document over the download-enabled default configuration with nothing installed \
         must answer the typed EmbedderModelNotAvailable error naming the resolved model \
         directory {}, never EventSearchPreparationUnsupported; got {document_result:?}",
        resolved_model_dir.display()
    );

    let query_result = embedder.prepare_query("probe text over an empty cache root");
    assert!(
        matches!(
            &query_result,
            Err(CgError::EmbedderModelNotAvailable { path, .. })
                if path == &resolved_model_dir.display().to_string()
        ),
        "prepare_query over the download-enabled default configuration with nothing installed \
         must answer the typed EmbedderModelNotAvailable error naming the resolved model \
         directory {}, never EventSearchPreparationUnsupported; got {query_result:?}",
        resolved_model_dir.display()
    );

    assert!(
        !observer.saw_any_fetch(),
        "checking the capability and preparing over a missing model must never start a download; \
         observed events: {:?}",
        observer.events.lock().expect("recording observer lock")
    );
}

#[test]
fn a_prepared_value_handed_to_a_missing_model_reports_it_not_installed_and_never_fetches() {
    // Mint genuinely sealed values from the already-installed canonical model, by absolute path,
    // BEFORE any environment redirection — so the values under test are real prepared values with
    // the canonical event-history profile, not test-local fabrications, and the handle that made
    // them is offline with downloads denied.
    let installed = nomic_real_model::nomic_live_embedder();
    let prepared_document = installed
        .prepare_document("the incident review closed with two open follow-ups")
        .expect("preparing a document over the installed canonical model must succeed");
    let prepared_query = installed
        .prepare_query("what did the incident review leave open")
        .expect("preparing a query over the installed canonical model must succeed");
    drop(installed);

    let tmp = tempfile::tempdir().expect("tempdir");
    let cache_root = tmp.path().join("empty-model-cache");
    let fake_home = tmp.path().join("fake-home");
    std::fs::create_dir_all(&cache_root).expect("create empty cache root");
    std::fs::create_dir_all(&fake_home).expect("create fake home dir");
    let _guard = EnvGuard::set(&[
        ("HOME", Some(fake_home.to_str().unwrap())),
        ("CG_MODEL_CACHE_ROOT", Some(cache_root.to_str().unwrap())),
    ]);

    let resolved_model_dir = cache_root.join("nomic-embed-text-v1.5");
    let observer = Arc::new(RecordingLoadObserver::default());
    let model = NomicTextEmbedder::new_with_load_observer(
        EmbedderConfig::default_nomic(),
        observer.clone(),
    );
    let embedder: &dyn Embedder = &model;

    let document_result = embedder.embed_prepared_document(&prepared_document);
    assert!(
        matches!(
            &document_result,
            Err(CgError::EmbedderModelNotAvailable { path, .. })
                if path == &resolved_model_dir.display().to_string()
        ),
        "embed_prepared_document over the download-enabled default configuration with nothing \
         installed must answer the typed EmbedderModelNotAvailable error naming the resolved \
         model directory {}, never EventSearchPreparationUnsupported and never a fetch; got \
         {document_result:?}",
        resolved_model_dir.display()
    );

    let query_result = embedder.embed_prepared_query(&prepared_query);
    assert!(
        matches!(
            &query_result,
            Err(CgError::EmbedderModelNotAvailable { path, .. })
                if path == &resolved_model_dir.display().to_string()
        ),
        "embed_prepared_query over the download-enabled default configuration with nothing \
         installed must answer the typed EmbedderModelNotAvailable error naming the resolved \
         model directory {}, never EventSearchPreparationUnsupported and never a fetch; got \
         {query_result:?}",
        resolved_model_dir.display()
    );

    assert!(
        !observer.saw_any_fetch(),
        "handing an already-prepared document or query to a handle whose model is not installed \
         must never start a download: the caller asked to embed a value it already holds, not to \
         install a model; observed events: {:?}",
        observer.events.lock().expect("recording observer lock")
    );
    assert!(
        !resolved_model_dir.exists(),
        "handing an already-prepared value to a handle whose model is not installed must leave \
         the resolved model directory {} untouched",
        resolved_model_dir.display()
    );
}
