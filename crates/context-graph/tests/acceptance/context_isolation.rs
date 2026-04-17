use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    CgError, CreateContext, CreateEntity, EntityType, ListEntityFilter, RecordObservation,
    StorageKey, Store,
};
use serde_json::{Value, json};
use tempfile::TempDir;

macro_rules! ok {
    ($expr:expr, $step:literal) => {{
        let result = $expr;
        assert!(result.is_ok(), "{}: expected Ok, got {:?}", $step, result);
        result.unwrap()
    }};
}

fn fixture() -> (TempDir, PathBuf, Store) {
    let tempdir = TempDir::new().expect("tempdir");
    let db_path = tempdir.path().join("context-graph.db");
    let store = Store::open(&db_path);
    (tempdir, db_path, store)
}

fn reopen(db_path: &Path) -> Store {
    Store::open(db_path)
}

fn map(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn context_input(name: &str) -> CreateContext {
    CreateContext {
        name: name.to_string(),
        labels: vec![],
        properties: BTreeMap::new(),
    }
}

// Use case: a fractional consultant's assistant lists services under client-beta and must never surface client-acme's entities, even after a process restart.
#[test]
fn context_isolation_entity_not_visible_cross_context() {
    let (_tempdir, db_path, store) = fixture();
    let ctx_acme = ok!(
        store.create_context(context_input("client-acme")),
        "create acme"
    );
    let ctx_beta = ok!(
        store.create_context(context_input("client-beta")),
        "create beta"
    );
    ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "service-checkout-api".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_acme.id,
        }),
        "create acme entity"
    );
    drop(store);
    let store = reopen(&db_path);
    let listed = ok!(
        store.list_entities(ListEntityFilter {
            context_id: Some(ctx_beta.id),
            ..ListEntityFilter::default()
        }),
        "list beta entities"
    );
    assert!(listed.is_empty());
}

// Use case: the assistant tries to link client-acme's checkout to client-beta's billing service; the engine refuses the cross-context edge at commit time and leaves no dangling row.
#[test]
fn context_isolation_traverse_stays_in_context() {
    let (_tempdir, db_path, store) = fixture();
    let ctx_acme = ok!(
        store.create_context(context_input("client-acme")),
        "create acme"
    );
    let ctx_beta = ok!(
        store.create_context(context_input("client-beta")),
        "create beta"
    );
    let checkout = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "service-checkout-api".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_acme.id,
        }),
        "create checkout"
    );
    let billing = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "service-billing-api".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_beta.id,
        }),
        "create billing"
    );
    let result = store.add_relationship(checkout.id, billing.id, "DEPENDS_ON".to_string());
    assert!(matches!(result, Err(CgError::CrossContextRelationship)));
    drop(store);
    let store = reopen(&db_path);
    let raw = ok!(
        store.admin_raw_scan(StorageKey::EdgeTable {
            from: checkout.id,
            rel: "DEPENDS_ON".to_string(),
        }),
        "raw edge scan"
    );
    assert!(raw.is_none());
}

// Use case: an observation authored under client-acme tries to attach to a client-beta entity; the engine refuses atomically so no phantom observation is discoverable later.
#[test]
fn context_isolation_observation_leak_blocked() {
    let (_tempdir, db_path, store) = fixture();
    let ctx_acme = ok!(
        store.create_context(context_input("client-acme")),
        "create acme"
    );
    let ctx_beta = ok!(
        store.create_context(context_input("client-beta")),
        "create beta"
    );
    let entity = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "service-billing-api".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_beta.id,
        }),
        "create beta entity"
    );
    let result = store.record_observation(RecordObservation {
        entity_id: entity.id,
        context_id: ctx_acme.id,
        observation_type: "latency_spike".to_string(),
        source: "agent".to_string(),
        data: map(&[("p99_ms", json!(450))]),
        observed_at: Some(Utc::now()),
    });
    assert!(matches!(result, Err(CgError::CrossContextObservation)));
    drop(store);
    let store = reopen(&db_path);
    let observations = ok!(
        store.admin_raw_observations_for_entity_in_context(entity.id, ctx_acme.id),
        "raw observation scan"
    );
    assert!(observations.is_empty());
}
