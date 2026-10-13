use std::collections::BTreeMap;

use context_graph::*;

pub fn fixture(mut config: StoreConfig) -> (tempfile::TempDir, Store, ContextId) {
    let dir = tempfile::tempdir().expect("temporary store directory");
    config.db_path = dir.path().join("memory.db");
    config.default_text_embedder = None;
    let store = Store::open(config).expect("open store");
    let context = context(&store, "primary");
    (dir, store, context)
}

pub fn context(store: &Store, name: &str) -> ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            ..Default::default()
        })
        .expect("create context")
        .id
}

pub fn entity(store: &Store, context_id: ContextId, name: &str) -> EntityId {
    store
        .create_entity(CreateEntity {
            name: name.into(),
            context_id,
            properties: BTreeMap::from([("mode".into(), serde_json::json!("initial"))]),
            ..Default::default()
        })
        .expect("create entity")
        .id
}

pub fn intention(store: &Store, context_id: ContextId) -> IntentionId {
    store
        .create_intention(CreateIntention {
            description: "keep the service available".into(),
            context_id,
            ..Default::default()
        })
        .expect("create intention")
        .id
}

pub fn observation(context_id: ContextId, entity_id: EntityId) -> RecordObservation {
    RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "measurement".into(),
        source: "sensor".into(),
        observed_at: chrono::DateTime::from_timestamp_millis(
            i64::try_from(contextdb_core::Wallclock::now().0).expect("timestamp fits i64"),
        )
        .expect("valid timestamp"),
        evidence: vec![EvidenceRef {
            id: EvidenceId::new_v7(),
            source_ref: "s3://observations/measurement".into(),
            retention_status: RetentionStatus::RetainedExternal,
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    }
}
