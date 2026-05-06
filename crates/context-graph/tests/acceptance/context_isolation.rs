use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    CgError, ContextId, CreateContext, CreateEntity, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, ListEntityFilter, ObservationId, RecordObservation,
    RetentionStatus, StorageKey, Store, StoreConfig,
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
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    (tempdir, db_path, store)
}

fn reopen(db_path: &Path) -> Store {
    Store::open(StoreConfig {
        db_path: db_path.to_path_buf(),
        ..Default::default()
    })
    .expect("open")
}

fn map(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn text_evidence(observation_id: ObservationId) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        context_id: ContextId::default(),
        kind: EvidenceKind::Text,
        source_ref: "webhook://agent/context-isolation".to_string(),
        mime_type: Some("text/plain".to_string()),
        content_hash: None,
        captured_at: Some(Utc::now()),
        byte_range: None,
        time_range_ms: None,
        frame_index: None,
        region: None,
        producer: EvidenceProducer {
            system: "acceptance".to_string(),
            model_name: "fixture".to_string(),
            model_version: "1".to_string(),
            pipeline_version: "1".to_string(),
        },
        text_projection: None,
        derived_from: None,
        retention_status: RetentionStatus::RetainedExternal,
        media_retention_until: None,
        properties: BTreeMap::new(),
    }
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
    let observation_id = ObservationId::new_v7();
    let result = store.record_observation(RecordObservation {
        id: observation_id,
        entity_id: entity.id,
        context_id: ctx_acme.id,
        observation_type: "latency_spike".to_string(),
        source: "agent".to_string(),
        observed_at: Utc::now(),
        evidence: vec![text_evidence(observation_id)],
        observed_properties: map(&[("p99_ms", json!(450))]),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
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
