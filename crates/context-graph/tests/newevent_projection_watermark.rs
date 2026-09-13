#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;

use context_graph::{
    AppendEventBatch, ContextId, CreateContext, EventAdapterDeclaration, EventCoverage, EventDraft,
    EventRenderRole, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: "watermark-proof".into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(key: &str, ordinal: u64) -> EventDraft {
    EventDraft {
        session_ref: "session-a".into(),
        source_event_key: key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "watermark-proof".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("event {ordinal}")),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn revision(store: &Store, context_id: ContextId) -> Option<Value> {
    let result = store
        .sync_database()
        .execute(
            "SELECT revision_tx FROM event_context_revisions WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("read event Context revision");
    assert!(result.rows.len() <= 1, "the Context primary key is unique");
    result.rows.first().map(|row| row[0].clone())
}

#[test]
fn real_append_advances_projection_revision_but_duplicate_and_empty_batches_do_not() {
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("event-watermark.db"),
        ..StoreConfig::default()
    })
    .expect("open Store");
    let context_id = store
        .create_context(CreateContext {
            name: "event watermark".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    assert_eq!(revision(&store, context_id), None);

    let first_batch = AppendEventBatch::live(
        context_id,
        declaration(),
        "source-a",
        vec![draft("event-1", 1)],
    );
    let first = store
        .append_event_batch(first_batch.clone())
        .expect("append first real event");
    let first_revision = revision(&store, context_id).expect("first append creates revision");
    assert_eq!(
        first_revision,
        Value::TxId(first.transaction_id.expect("real append commits")),
        "the event and its projection revision share one commit"
    );

    let duplicate = store
        .append_event_batch(first_batch)
        .expect("duplicate append is idempotent");
    assert_eq!(duplicate.inserted, 0);
    assert_eq!(duplicate.transaction_id, None);
    assert_eq!(revision(&store, context_id), Some(first_revision.clone()));

    let empty = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "source-a",
            Vec::new(),
        ))
        .expect("empty live batch is a no-op");
    assert_eq!(empty.inserted, 0);
    assert_eq!(empty.transaction_id, None);
    assert_eq!(revision(&store, context_id), Some(first_revision.clone()));

    let second = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "source-a",
            vec![draft("event-2", 2)],
        ))
        .expect("append second real event");
    let second_revision = revision(&store, context_id).expect("second append keeps revision row");
    assert_eq!(
        second_revision,
        Value::TxId(second.transaction_id.expect("second real append commits"))
    );
    assert_ne!(second_revision, first_revision);
}
