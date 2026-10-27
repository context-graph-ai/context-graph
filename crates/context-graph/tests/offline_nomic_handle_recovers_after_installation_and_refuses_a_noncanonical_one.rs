//! `NomicTextEmbedder::event_history_artifact()` caches canonical and noncanonical verdicts for
//! present installations, shared across clones of a handle. An absent installation is checked
//! again on later calls, so a long-lived handle can accept a canonical model installed after its
//! first miss while refusing a noncanonical installation under the reserved event-history identity.
//!
//! Case (a): `EventHistoryTextEmbedder::registration` must accept the now-installed canonical
//! model on the same handle rather than continuing to answer as if nothing were installed. Case
//! (b): `prepare_document` must answer `EventSearchPreparationUnsupported` over noncanonical
//! weights discovered after that same initial miss, never silently succeed over them.

use std::sync::{Arc, Mutex};

use context_graph::embedder::load_observer::{EmbedderLoadEvent, EmbedderLoadObserver};
use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::{CgError, Embedder, EmbedderConfig, EventSearchEmbedder, ModelSource};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
const NONCANONICAL_REVISION: &str = "noncanonical-test-revision-same-dimension-r12-2";

fn canonical_fixture_dir() -> std::path::PathBuf {
    nomic_real_model::canonical_model_dir()
}

fn assert_canonical_fixture_present() {
    let source = canonical_fixture_dir();
    assert!(
        source.join("model.safetensors").is_file(),
        "the canonical Nomic fixture must already be installed at {} for this proof to run \
         offline; a missing cache is BLOCKED, not a reason to download",
        source.display()
    );
}

fn install_canonical_fixture(destination: &std::path::Path) {
    assert_canonical_fixture_present();
    let source = canonical_fixture_dir();
    std::fs::create_dir_all(destination).expect("create destination model directory");
    for file in ["model.safetensors", "tokenizer.json", "config.json"] {
        std::fs::copy(source.join(file), destination.join(file))
            .unwrap_or_else(|err| panic!("copy canonical fixture file {file}: {err}"));
    }
    std::fs::copy(source.join("REVISION"), destination.join("REVISION"))
        .expect("copy canonical REVISION marker");
}

fn install_noncanonical_same_dimension_fixture(destination: &std::path::Path) {
    assert_canonical_fixture_present();
    let source = canonical_fixture_dir();
    std::fs::create_dir_all(destination).expect("create destination model directory");
    for file in ["model.safetensors", "tokenizer.json", "config.json"] {
        std::fs::copy(source.join(file), destination.join(file))
            .unwrap_or_else(|err| panic!("copy canonical fixture file {file}: {err}"));
    }
    std::fs::write(destination.join("REVISION"), NONCANONICAL_REVISION)
        .expect("write noncanonical REVISION marker");
}

/// An unpinned `LocalPath` config — no `pinned_revision` — over `model_dir`, offline, downloads
/// denied. This is the exact shape `nomic.rs`'s `check_revision_marker` skips entirely: a config
/// with no expected revision passes that check trivially regardless of what is actually installed.
fn unpinned_local_offline_config(model_dir: std::path::PathBuf) -> EmbedderConfig {
    EmbedderConfig {
        source: ModelSource::LocalPath(model_dir),
        offline: true,
        cache_dir: None,
        pinned_revision: None,
        network_policy: "deny-download".to_string(),
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
    fn load_completed_count(&self) -> usize {
        self.events
            .lock()
            .expect("recording observer lock")
            .iter()
            .filter(|event| matches!(event, EmbedderLoadEvent::LoadCompleted { .. }))
            .count()
    }
}

#[test]
fn a_handle_that_first_saw_no_model_recovers_after_canonical_installation() {
    let tmp_home = tempfile::tempdir().expect("temp HOME");
    let model_dir = tmp_home.path().join(NOMIC_MODEL_ID);
    std::fs::create_dir_all(&model_dir).expect("create empty model directory");

    let observer = Arc::new(RecordingLoadObserver::default());
    let config = unpinned_local_offline_config(model_dir.clone());
    let model = Arc::new(NomicTextEmbedder::new_with_load_observer(
        config,
        observer.clone(),
    ));
    let embedder: &dyn Embedder = model.as_ref();

    let before_install = embedder.prepare_document("probe text over an empty cache");
    assert!(
        matches!(
            &before_install,
            Err(CgError::EmbedderModelNotAvailable { source, path })
                if source.starts_with("LocalPath(") && path == &model_dir.display().to_string()
        ),
        "preparation over an empty, isolated local model directory must refuse with the typed \
         EmbedderModelNotAvailable error naming the exact missing directory; got {before_install:?}"
    );

    install_canonical_fixture(&model_dir);

    let after_install = embedder.prepare_document("probe text after canonical installation");
    assert!(
        after_install.is_ok(),
        "the SAME long-lived handle must recover once the canonical model appears at its \
         resolved directory rather than staying stuck on its first 'absent' decision; got \
         {after_install:?}"
    );

    let registration = EventHistoryTextEmbedder::registration(Arc::clone(&model));
    assert!(
        registration.is_ok(),
        "EventHistoryTextEmbedder::registration on the SAME handle must accept the now-installed \
         canonical artifact rather than answering as if the artifact were still absent; got \
         {registration:?}"
    );

    assert_eq!(
        observer.load_completed_count(),
        1,
        "recovering after installation and then registering on the same handle must read and \
         parse the Nomic weights exactly once, never once per call; observed events: {:?}",
        observer.events.lock().expect("recording observer lock")
    );
}

#[test]
fn a_handle_that_first_saw_no_model_refuses_a_later_noncanonical_installation() {
    let tmp_home = tempfile::tempdir().expect("temp HOME");
    let model_dir = tmp_home.path().join(NOMIC_MODEL_ID);
    std::fs::create_dir_all(&model_dir).expect("create empty model directory");

    let config = unpinned_local_offline_config(model_dir.clone());
    let model = Arc::new(NomicTextEmbedder::new(config));
    let embedder: &dyn Embedder = model.as_ref();

    let before_install = embedder.prepare_document("probe text over an empty cache");
    assert!(
        matches!(
            &before_install,
            Err(CgError::EmbedderModelNotAvailable { .. })
        ),
        "the initial miss over an empty directory must answer EmbedderModelNotAvailable; got \
         {before_install:?}"
    );

    install_noncanonical_same_dimension_fixture(&model_dir);

    let after_install = embedder.prepare_document("probe text after a noncanonical installation");
    assert!(
        matches!(
            after_install,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "a noncanonical revision installed after the same initial miss must be refused with \
         EventSearchPreparationUnsupported, never silently admitted and stamped with the reserved \
         event-history identity just because the handle's first decision was 'absent'; got \
         {after_install:?}"
    );

    let registration = EventHistoryTextEmbedder::registration(Arc::clone(&model));
    assert!(
        registration.is_err(),
        "registration over a noncanonical installation discovered after an initial 'absent' \
         decision must be refused, never accepted; got {registration:?}"
    );
}
