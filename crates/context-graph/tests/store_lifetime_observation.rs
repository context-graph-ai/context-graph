#![cfg(context_graph_internal_test_support)]

use context_graph::{
    EmbedderConfig, Store, StoreConfig, test_support::store_database_is_live_for_test,
};

#[test]
fn registry_observation_tracks_shared_store_database_ownership() {
    let directory = tempfile::tempdir().expect("temporary store directory");
    let path = directory.path().join("store-lifetime.db");
    let config = StoreConfig {
        db_path: path.clone(),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..StoreConfig::default()
    };

    assert!(
        !path.exists(),
        "the observation must not create the store file"
    );
    assert!(
        !store_database_is_live_for_test(&path),
        "a never-opened path has no live Store Database owner"
    );
    assert!(
        !path.exists(),
        "looking up a never-opened path must not create the store file"
    );

    let first = Store::open(config.clone()).expect("open first store");
    assert!(
        store_database_is_live_for_test(&path),
        "the first Store owns a live Database"
    );

    let second = Store::open(config).expect("open shared store");
    drop(first);
    assert!(
        store_database_is_live_for_test(&path),
        "the second shared Store still owns the Database after the first drops"
    );

    drop(second);
    assert!(
        !store_database_is_live_for_test(&path),
        "after every Store handle drops, the registry retains only a dead Weak handle"
    );
}
