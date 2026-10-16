use super::*;

/// Cached canonical metadata still refuses pending publications and a changed installed revision.
#[test]
fn cached_canonical_metadata_checks_publication_state_and_installed_revision() {
    let directory = tempfile::tempdir().unwrap();
    let revision_path = directory.path().join("REVISION");
    file_system::write(&revision_path, crate::embedder::NOMIC_DEFAULT_REVISION).unwrap();
    let embedder = NomicTextEmbedder::new(EmbedderConfig {
        source: ModelSource::LocalPath(directory.path().to_path_buf()),
        offline: true,
        cache_dir: None,
        pinned_revision: None,
        network_policy: "offline".to_string(),
    });
    // Seed only the expensive metadata verdict, without claiming these fixtures are real
    // weights or warming a loaded-model cache. Exercise the production cached-admission path.
    embedder
        .canonical_event_history_artifact
        .set(EventHistoryArtifact::Canonical)
        .unwrap();
    assert_eq!(
        embedder.event_history_artifact(),
        EventHistoryArtifact::Canonical
    );
    let original_revision = installed_revision(directory.path()).unwrap();
    for name in [".catalog-publication", ".nomic-rollback/READY"] {
        let marker = directory.path().join(name);
        file_system::create_dir_all(marker.parent().unwrap()).unwrap();
        file_system::write(&marker, []).unwrap();
        assert_eq!(
            embedder.event_history_artifact(),
            EventHistoryArtifact::Noncanonical
        );
        assert!(matches!(
            embedder.load_model_without_installing(),
            Err(CgError::Engine(_))
        ));
        assert!(marker.exists());
        assert!(embedder.structured_loader_events().is_empty());
        // Explicit test-fixture cleanup, not production recovery. A transient refusal must
        // not poison the remembered metadata verdict after a valid installation is restored.
        file_system::remove_file(&marker).unwrap();
        assert_eq!(
            embedder.event_history_artifact(),
            EventHistoryArtifact::Canonical
        );
    }
    file_system::write(&revision_path, "different-complete-revision").unwrap();
    assert_ne!(
        installed_revision(directory.path()).unwrap(),
        original_revision
    );
    assert_eq!(
        embedder.event_history_artifact(),
        EventHistoryArtifact::Noncanonical
    );
}
