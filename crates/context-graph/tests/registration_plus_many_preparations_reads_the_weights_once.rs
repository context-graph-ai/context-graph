//! Registering the event-history space and then preparing many passages must read the 547 MB
//! Nomic weights file exactly once. `EventHistoryTextEmbedder::registration` and the checked
//! `prepare_document`/`prepare_query` entrypoints share the same underlying `NomicTextEmbedder`
//! and its in-process model cache (`NOMIC_MODEL_CACHE`), so the actual weights parse
//! (`load_nomic_artifacts`) should happen once and every later call should reuse the cached
//! model.
//!
//! This test observes that through the existing load-observation seam
//! (`EmbedderLoadObserver`/`EmbedderLoadEvent`), the same seam `candle-embeddinggemma` uses to
//! report exactly one `LoadStarted`/`LoadCompleted` pair around its own heavy load.
//! `NomicTextEmbedder::load_model` emits `LoadStarted` before parsing the weights with
//! `load_nomic_artifacts` and `LoadCompleted` after it, and emits `EmbedderLoadEvent::CacheHit`
//! only on the branch that reuses an already-loaded model from `NOMIC_MODEL_CACHE`. So
//! registration plus fifty rounds of `prepare_document`/`prepare_query` must produce exactly one
//! `LoadCompleted`.

use std::sync::{Arc, Mutex};

use context_graph::embedder::load_observer::{EmbedderLoadEvent, EmbedderLoadObserver};
use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::{Embedder, EmbedderConfig, EventSearchEmbedder, ModelSource};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
const NOMIC_CANONICAL_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

fn canonical_fixture_dir() -> std::path::PathBuf {
    nomic_real_model::canonical_model_dir()
}

/// A local copy of the canonical, already-installed fixture under a disposable temp directory —
/// never the real cache — so this test cannot write into, or be confused with, the host's actual
/// model cache.
fn copy_canonical_fixture(destination: &std::path::Path) {
    let source = canonical_fixture_dir();
    assert!(
        source.join("model.safetensors").is_file(),
        "the canonical Nomic fixture must already be installed at {} for this proof to run \
         offline; a missing cache is BLOCKED, not a reason to download",
        source.display()
    );
    assert_ne!(
        source, destination,
        "the fixture copy destination must never equal the real cache source"
    );
    std::fs::create_dir_all(destination).expect("create destination model directory");
    for file in ["model.safetensors", "tokenizer.json", "config.json"] {
        std::fs::copy(source.join(file), destination.join(file))
            .unwrap_or_else(|err| panic!("copy canonical fixture file {file}: {err}"));
    }
    std::fs::copy(source.join("REVISION"), destination.join("REVISION"))
        .expect("copy canonical REVISION marker");
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

#[test]
fn registration_plus_fifty_preparations_reads_the_weights_exactly_once() {
    let tmp_home = tempfile::tempdir().expect("temp HOME");
    let model_dir = tmp_home.path().join(NOMIC_MODEL_ID);
    copy_canonical_fixture(&model_dir);

    let observer = Arc::new(RecordingLoadObserver::default());
    let config = EmbedderConfig {
        source: ModelSource::LocalPath(model_dir),
        offline: true,
        cache_dir: None,
        pinned_revision: Some(NOMIC_CANONICAL_REVISION.to_string()),
        network_policy: "deny-download".to_string(),
    };
    let model = Arc::new(NomicTextEmbedder::new_with_load_observer(
        config,
        observer.clone(),
    ));

    let registration =
        EventHistoryTextEmbedder::registration(model).expect("canonical fixture must register");
    let embedder: &dyn Embedder = registration.embedder.as_ref();

    for i in 0..50 {
        embedder
            .prepare_document(&format!("passage {i} covers a distinct event"))
            .unwrap_or_else(|err| panic!("prepare_document must succeed for passage {i}: {err:?}"));
        embedder
            .prepare_query(&format!("find passage {i}"))
            .unwrap_or_else(|err| panic!("prepare_query must succeed for passage {i}: {err:?}"));
    }

    let events = observer.events.lock().expect("recording observer lock");
    let load_completed = events
        .iter()
        .filter(|event| matches!(event, EmbedderLoadEvent::LoadCompleted { .. }))
        .count();
    assert_eq!(
        load_completed, 1,
        "registration plus 50 rounds of prepare_document/prepare_query must read and parse the \
         Nomic weights exactly once (one EmbedderLoadEvent::LoadCompleted), never once per call \
         and never zero; observed events: {events:?}"
    );
}
