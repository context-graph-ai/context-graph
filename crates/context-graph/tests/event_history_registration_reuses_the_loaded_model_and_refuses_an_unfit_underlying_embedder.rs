//! The advertised `EventHistoryTextEmbedder::over`/`registration` route is the one way to put the
//! dedicated event-history space into a store, and it must reuse whatever model instance the
//! caller already loaded rather than loading a second copy, and it must never advertise a model
//! it cannot actually serve.
//!
//! Proof method for "both roles share one model instance": this crate loads weights lazily and
//! caches the loaded artifact per `(model_dir, pinned_revision)` behind an `Arc`
//! (`NOMIC_MODEL_CACHE` in `src/embedder/nomic.rs`), but that cache is private and unobservable
//! from an integration test. What IS observable from here is the caller-held
//! `Arc<NomicTextEmbedder>` itself: `EventHistoryTextEmbedder::over` is documented to store a
//! clone of exactly the `Arc` it is handed, never to construct a fresh `NomicTextEmbedder`. So
//! this test takes `Arc::strong_count` of its own handle before registration, once right after
//! registration, and again after both checked roles have run through the registered wrapper — if
//! the count is stable at "mine plus the wrapper's one stored clone" across all three
//! measurements, no second model was constructed and reloaded in between.

use std::fs;
use std::sync::{Arc, Mutex};

use context_graph::embedder::load_observer::{EmbedderLoadEvent, EmbedderLoadObserver};
use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::embedder::{Embedder, EmbedderConfig};
use context_graph::embedding_space::event_history_embedding_space;
use context_graph::{CgError, EventSearchEmbedder, ModelSource, Store, StoreConfig};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

#[derive(Default)]
struct RecordingLoadObserver {
    events: Mutex<Vec<EmbedderLoadEvent>>,
}

impl RecordingLoadObserver {
    fn events(&self) -> Vec<EmbedderLoadEvent> {
        self.events.lock().expect("load observer events").clone()
    }
}

impl EmbedderLoadObserver for RecordingLoadObserver {
    fn observe(&self, event: EmbedderLoadEvent) {
        self.events
            .lock()
            .expect("load observer events")
            .push(event);
    }
}

fn admission_refuses_without_loading(config: EmbedderConfig) -> (bool, Vec<EmbedderLoadEvent>) {
    let observer = Arc::new(RecordingLoadObserver::default());
    let model = Arc::new(NomicTextEmbedder::new_with_load_observer(
        config,
        observer.clone(),
    ));
    let refused = matches!(
        EventHistoryTextEmbedder::registration(model),
        Err(CgError::EventHistoryEmbedderConstructionRefused { .. })
    );
    (refused, observer.events())
}

#[test]
fn public_default_nomic_offline_configuration_uses_the_canonical_installed_model_for_event_history()
{
    let config = EmbedderConfig::default_nomic_offline();
    assert_eq!(
        config.source,
        ModelSource::HuggingFace {
            repo: "nomic-ai/nomic-embed-text-v1.5".to_string(),
        },
        "the offline constructor must name the canonical Nomic Hugging Face repository"
    );
    assert!(
        config.offline,
        "the offline constructor must never enter a network fetch path"
    );
    assert_eq!(
        config.pinned_revision.as_deref(),
        Some(nomic_real_model::NOMIC_CANONICAL_REVISION),
        "the offline constructor must pin the canonical installed revision"
    );
    assert_eq!(
        config.cache_dir, None,
        "the offline constructor must use the canonical cache root without a custom override"
    );
    assert_eq!(
        config.network_policy, "deny-download",
        "the offline constructor must explicitly deny downloads"
    );

    let installed_revision =
        fs::read_to_string(nomic_real_model::canonical_model_dir().join("REVISION"))
            .expect("the canonical installed model fixture must include its revision marker");
    assert_eq!(
        installed_revision.trim(),
        nomic_real_model::NOMIC_CANONICAL_REVISION,
        "the installed fixture must be the revision the public constructor pins"
    );

    let registration = EventHistoryTextEmbedder::registration(Arc::new(NomicTextEmbedder::new(
        config,
    )))
    .expect("the public offline Nomic configuration must register event history from the installed canonical fixture");
    assert_eq!(
        registration.space,
        event_history_embedding_space(),
        "the public offline Nomic configuration must register the canonical event-history space"
    );
}

#[test]
fn event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder() {
    let model = Arc::new(NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
    ));
    assert_eq!(
        Arc::strong_count(&model),
        1,
        "the freshly built Arc must start with exactly one owner: this test"
    );

    let registration = EventHistoryTextEmbedder::registration(Arc::clone(&model)).expect(
        "registering the canonical event-history space over a fit underlying embedder must succeed",
    );
    assert_eq!(
        registration.space,
        event_history_embedding_space(),
        "the registration must declare the exact canonical event-history space, not a substitute"
    );
    assert_eq!(
        Arc::strong_count(&model),
        2,
        "registration must store a CLONE of the caller's own Arc (mine + the wrapper's), never \
         load a second model instance"
    );

    let wrapper = registration.embedder;
    let checked: &dyn Embedder = wrapper.as_ref();

    let prepared_document = checked
        .prepare_document("the incident review closed with three follow-up actions")
        .expect("document preparation over the real cached model must succeed");
    let document_output = checked
        .embed_prepared_document(&prepared_document)
        .expect("embedding a document prepared by the same wrapper must succeed");
    assert_eq!(
        document_output.embedding_space_id,
        event_history_embedding_space().embedding_space_id,
        "the document role must report the exact dedicated event-history space id"
    );

    let prepared_query = checked
        .prepare_query("find the incident review from last quarter")
        .expect("query preparation over the real cached model must succeed");
    let query_output = checked
        .embed_prepared_query(&prepared_query)
        .expect("embedding a query prepared by the same wrapper must succeed");
    assert_eq!(
        query_output.embedding_space_id,
        event_history_embedding_space().embedding_space_id,
        "the query role must report the exact dedicated event-history space id"
    );

    assert_eq!(
        Arc::strong_count(&model),
        2,
        "running both checked roles through the registered wrapper must not construct or load \
         any additional model instance — the count stays at mine plus the wrapper's one clone"
    );

    // The registration is also usable through the real Store registration API, not only held
    // directly — proving `registration()` really is "ready to pass to
    // `Store::open_with_embedder_registrations`", as its own doc comment claims.
    let dir = tempfile::tempdir().expect("store dir");
    let space = event_history_embedding_space();
    let store_registration = EventHistoryTextEmbedder::registration(Arc::clone(&model))
        .expect("a second registration over the same fit model must also succeed");
    let _store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.path().join("cg.db"),
            extra_embedding_spaces: vec![space],
            ..Default::default()
        },
        vec![store_registration],
    )
    .expect("a store must open with the event-history registration built through the real API");

    // RED arm 1: a non-768-dimension underlying embedder must be refused, not silently
    // advertised under the dedicated 768-dimension event-history space.
    let wrong_dimension_model = Arc::new(NomicTextEmbedder::new_for_dimension(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
        256,
    ));
    let wrong_dimension_result = EventHistoryTextEmbedder::over(wrong_dimension_model);
    assert!(
        matches!(
            &wrong_dimension_result,
            Err(CgError::EventHistoryEmbedderConstructionRefused { .. })
        ),
        "a non-768-dimension underlying embedder must be refused with the typed construction \
         error, got {wrong_dimension_result:?}"
    );

    // RED arm 2: a download-enabled config must be refused — the canonical registration promises
    // a cache-only, offline model, never one that might silently reach the network mid-run.
    let download_enabled_config = EmbedderConfig::default_nomic();
    let download_enabled_model = Arc::new(NomicTextEmbedder::new(download_enabled_config));
    let download_enabled_result = EventHistoryTextEmbedder::registration(download_enabled_model);
    assert!(
        matches!(
            &download_enabled_result,
            Err(CgError::EventHistoryEmbedderConstructionRefused { .. })
        ),
        "a download-enabled underlying config must be refused with the typed construction error, \
         got {download_enabled_result:?}"
    );

    // RED arm 3: a config pinned to a revision other than the canonical one must be refused —
    // the registration must never silently advertise weights it has not verified are the
    // canonical revision.
    let wrong_revision_config = EmbedderConfig {
        pinned_revision: Some("deadbeef-not-the-canonical-revision".to_string()),
        ..nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir())
    };
    let wrong_revision_model = Arc::new(NomicTextEmbedder::new(wrong_revision_config));
    let wrong_revision_result = EventHistoryTextEmbedder::over(wrong_revision_model);
    assert!(
        matches!(
            &wrong_revision_result,
            Err(CgError::EventHistoryEmbedderConstructionRefused { .. })
        ),
        "a config pinned to a non-canonical revision must be refused with the typed construction \
         error, got {wrong_revision_result:?}"
    );
}

/// The canonical event-history registration is an admission boundary, not a promise to discover
/// an unusable model once a caller has already installed the space. Every rejected configuration
/// below has a recording observer: admission must return the typed construction refusal before a
/// fetch or model load starts.
#[test]
fn canonical_event_history_admission_requires_pinned_verified_local_artifact_without_fetch() {
    let missing_root = tempfile::tempdir().expect("missing artifact root");
    let missing = nomic_real_model::local_offline_config(missing_root.path().join("missing"));

    let corrupt_root = tempfile::tempdir().expect("corrupt artifact root");
    let corrupt_dir = corrupt_root.path().join(nomic_real_model::NOMIC_MODEL_ID);
    fs::create_dir_all(&corrupt_dir).expect("corrupt artifact directory");
    fs::write(corrupt_dir.join("model.safetensors"), vec![0_u8; 1_000_000])
        .expect("corrupt model fixture");
    fs::write(corrupt_dir.join("tokenizer.json"), "not a tokenizer")
        .expect("corrupt tokenizer fixture");
    fs::write(corrupt_dir.join("config.json"), "not a config").expect("corrupt config fixture");
    fs::write(
        corrupt_dir.join("REVISION"),
        nomic_real_model::NOMIC_CANONICAL_REVISION,
    )
    .expect("corrupt revision fixture");
    let corrupt = nomic_real_model::local_offline_config(corrupt_dir);

    let catalog_managed_local = EmbedderConfig {
        source: ModelSource::LocalPath(nomic_real_model::canonical_model_dir()),
        offline: true,
        cache_dir: None,
        pinned_revision: Some(nomic_real_model::NOMIC_CANONICAL_REVISION.to_string()),
        network_policy: "catalog-managed-local".to_string(),
    };
    let unpinned_catalog_managed_local = EmbedderConfig {
        pinned_revision: None,
        ..catalog_managed_local.clone()
    };
    let download_enabled = EmbedderConfig::default_nomic();

    let (missing_refused, missing_events) = admission_refuses_without_loading(missing);
    let (corrupt_refused, corrupt_events) = admission_refuses_without_loading(corrupt);
    let (download_enabled_refused, download_enabled_events) =
        admission_refuses_without_loading(download_enabled);

    let revision_path = nomic_real_model::canonical_model_dir().join("REVISION");
    let installed_revision = fs::read_to_string(&revision_path)
        .expect("the cached real-model fixture supplies its revision marker");
    let installed = EventHistoryTextEmbedder::registration(Arc::new(NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
    )));
    let catalog_managed_registration = EventHistoryTextEmbedder::registration(Arc::new(
        NomicTextEmbedder::new(catalog_managed_local),
    ));
    let unpinned_catalog_managed_registration = EventHistoryTextEmbedder::registration(Arc::new(
        NomicTextEmbedder::new(unpinned_catalog_managed_local),
    ));

    assert_eq!(
        (
            missing_refused,
            corrupt_refused,
            download_enabled_refused,
            missing_events.is_empty(),
            corrupt_events.is_empty(),
            download_enabled_events.is_empty(),
            installed_revision.trim(),
            installed.is_ok(),
            catalog_managed_registration.is_ok(),
            unpinned_catalog_managed_registration.is_ok(),
        ),
        (
            true,
            true,
            true,
            true,
            true,
            true,
            nomic_real_model::NOMIC_CANONICAL_REVISION,
            true,
            true,
            true,
        ),
        "event-history admission must reject missing, corrupt, and download-enabled models \
         before any fetch or load starts, while accepting the exact installed revision for \
         catalog-managed local settings with or without an optional config pin; \
         missing={missing_events:?}, corrupt={corrupt_events:?}, \
         download_enabled={download_enabled_events:?}"
    );
}
