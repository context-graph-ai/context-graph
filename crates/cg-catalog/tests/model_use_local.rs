//! `cg model use` selects the bundled local GGUF as the active understanding worker.
//!
//! Today `cg model connect` only accepts HTTP endpoints, so a fetched local model can never be made
//! the active worker — the $0-local daily-loop path is not walkable. `connect_local_model` closes
//! that: it makes a NoAuth connection active whose model IS the local GGUF, with consent
//! auto-recorded (a local in-process model has zero token/quota spend, so the consent gate that
//! guards a paid subscription is moot). No network, no real 3.1GB model — a fake cache file stands in.

use cg_catalog::{ModelConnectionCatalog, ModelKind, UNDERSTANDING_DEFAULT_ID};

#[test]
fn connect_local_model_makes_it_the_active_understanding_worker() {
    let temp = tempfile::tempdir().expect("temp root");
    let root = temp.path();
    let mut catalog = ModelConnectionCatalog::open(root);

    // Seed a fetched local model: a cache dir with a stand-in artifact file (never the real GGUF).
    let model_dir = root.join(UNDERSTANDING_DEFAULT_ID);
    std::fs::create_dir_all(&model_dir).expect("model dir");
    std::fs::write(model_dir.join("model.gguf"), b"fake-gguf").expect("artifact");
    catalog.register_local_model(UNDERSTANDING_DEFAULT_ID, ModelKind::LlmClient, &model_dir);

    // Select it as the active understanding worker.
    let connection = catalog
        .connect_local_model(UNDERSTANDING_DEFAULT_ID)
        .expect("selecting a fetched local model succeeds");

    // Consent is auto-recorded — running `cg model use` IS the opt-in for a zero-spend local model.
    assert!(
        connection.consent_recorded,
        "a local in-process model needs no separate consent step"
    );
    assert_eq!(connection.name, UNDERSTANDING_DEFAULT_ID);

    // The active connection now resolves the local GGUF as the understanding model.
    assert_eq!(
        catalog.active_connection_name().as_deref(),
        Some(UNDERSTANDING_DEFAULT_ID)
    );
    assert_eq!(
        catalog.active_model_id().as_deref(),
        Some(UNDERSTANDING_DEFAULT_ID)
    );
    let resolved = catalog
        .model_for_connection(UNDERSTANDING_DEFAULT_ID)
        .expect("the active connection resolves its model");
    assert_eq!(resolved.id, UNDERSTANDING_DEFAULT_ID);
    assert!(
        resolved.cache_path.is_some(),
        "the active worker is the in-process local GGUF"
    );

    // The memory-capture gate (active connection + consent recorded + no last error) passes.
    let conn = catalog
        .read_connection(UNDERSTANDING_DEFAULT_ID)
        .expect("connection persisted");
    assert!(conn.consent_recorded);
    assert!(
        catalog
            .last_error_for_connection(UNDERSTANDING_DEFAULT_ID)
            .is_none(),
        "a freshly-selected local model carries no failure"
    );
}

#[test]
fn connect_local_model_rejects_uncached_understanding_default() {
    let temp = tempfile::tempdir().expect("temp root");
    let root = temp.path();
    let mut catalog = ModelConnectionCatalog::open(root);

    let err = catalog
        .connect_local_model(UNDERSTANDING_DEFAULT_ID)
        .expect_err("an unfetched local model cannot be selected");
    assert!(
        err.contains("cg model fetch"),
        "the error guides the user to fetch first: {err}"
    );
    assert!(
        catalog.active_connection_name().is_none(),
        "a failed selection leaves no active connection"
    );
}
