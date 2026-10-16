//! Source checkpoints track progress independently of unique event counts.
//!
//! The fixture uses only the public append and import APIs. SQL opens the database only after the
//! Store is dropped, to inspect the persisted rows that those APIs produced.

use std::collections::HashMap;
use std::io::{BufRead, Cursor};
use std::num::{NonZeroU32, NonZeroUsize};

use context_graph::{
    AdaptedSourceRecord, AppendEventBatch, CreateContext, EndOfSource, EventAdapterDeclaration,
    EventBatchLimits, EventCoverage, EventDraft, EventRange, EventRenderRole, EventSourceAdapter,
    EventSourceIdentity, EventSourceRead, EventStreamSelector, ImportEventOptions,
    SourceRecordPosition, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;
use serde_json::json;

const CONNECTOR: &str = "checkpoint-count-contract";
const SESSION: &str = "checkpoint-count-session";

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-v1".into(),
        anchor_declaration_version: "fixture-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/line".into()],
    }
}

fn draft() -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: "same-source-event-key".into(),
        source_position: json!({"line": 1}),
        occurred_at: None,
        author: "fixture-author".into(),
        kind: "fixture-kind".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("same immutable event content".into()),
        payload: json!({"fixture": "same-content"}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn open_store(path: std::path::PathBuf) -> (Store, context_graph::ContextId) {
    let store = Store::open(StoreConfig {
        db_path: path,
        ..StoreConfig::default()
    })
    .expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "checkpoint count contract".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    (store, context_id)
}

struct OneRecordSource {
    declaration: EventAdapterDeclaration,
    identity: EventSourceIdentity,
    record: Option<AdaptedSourceRecord>,
}

impl OneRecordSource {
    fn new(source_instance_ref: &str, source_len: u64) -> Self {
        Self {
            declaration: declaration(),
            identity: EventSourceIdentity {
                source_instance_ref: source_instance_ref.into(),
                session_ref: SESSION.into(),
                provisional_empty: false,
            },
            record: Some(AdaptedSourceRecord {
                drafts: vec![draft()],
                checkpoint_after_record: json!({"record": 1}),
                source_position_after_record: SourceRecordPosition {
                    byte_offset: source_len,
                    coordinate: json!({"line": 1}),
                },
            }),
        }
    }
}

impl EventSourceAdapter for OneRecordSource {
    fn declaration(&self) -> &EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        reader: &mut dyn BufRead,
        _source_locator_ref: &str,
        _session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        let mut identity_line = String::new();
        reader
            .read_line(&mut identity_line)
            .map_err(|error| context_graph::CgError::Engine(error.to_string()))?;
        Ok(self.identity.clone())
    }

    fn resume_source(
        &mut self,
        _identity: &EventSourceIdentity,
        _checkpoint: &serde_json::Value,
        _position: &SourceRecordPosition,
    ) -> context_graph::Result<()> {
        Ok(())
    }

    fn read_next_record(
        &mut self,
        reader: &mut dyn BufRead,
        _resolver: &mut dyn context_graph::EventCorrelationResolver,
    ) -> context_graph::Result<EventSourceRead> {
        let mut line = Vec::new();
        reader
            .read_until(b'\n', &mut line)
            .map_err(|error| context_graph::CgError::Engine(error.to_string()))?;
        if line.is_empty() {
            return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
        }
        Ok(EventSourceRead::Record(
            self.record.take().expect("one fixture record"),
        ))
    }
}

fn import_options(
    context_id: context_graph::ContextId,
    source_locator_ref: &str,
) -> ImportEventOptions {
    ImportEventOptions {
        context_id,
        source_locator_ref: source_locator_ref.into(),
        session_override: None,
        batch_limits: EventBatchLimits {
            max_events: NonZeroUsize::new(1).expect("nonzero"),
            max_normalized_bytes: NonZeroUsize::new(1024).expect("nonzero"),
        },
    }
}

fn scalar_count(db: &Database, sql: &str, params: HashMap<String, Value>) -> i64 {
    let result = db.execute(sql, &params).expect("count query");
    assert_eq!(result.rows.len(), 1, "one aggregate row");
    match result.rows[0][0] {
        Value::Int64(value) => value,
        ref value => panic!("expected integer count, got {value:?}"),
    }
}

fn checkpoint_latest_sequences(db: &Database, context_id: context_graph::ContextId) -> Vec<i64> {
    let result = db
        .execute(
            "SELECT latest_event_seq FROM event_source_checkpoints WHERE context_id = $context_id ORDER BY source_instance_ref",
            &HashMap::from([(
                "context_id".into(),
                Value::Uuid(context_id.as_uuid()),
            )]),
        )
        .expect("read source checkpoint rows");
    result
        .rows
        .iter()
        .map(|row| match row[0] {
            Value::Int64(value) => value,
            ref value => panic!("expected latest event sequence, got {value:?}"),
        })
        .collect()
}

#[test]
fn live_append_keeps_the_publicly_readable_event_when_no_source_commit_exists() {
    let root = tempfile::tempdir().expect("tempdir");
    let db_path = root.path().join("live-append.db");
    let (store, context_id) = open_store(db_path.clone());

    let appended = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "live-append-source",
            vec![draft()],
        ))
        .expect("public live append");
    assert_eq!((appended.inserted, appended.already_present), (1, 0));
    let event = appended
        .events
        .into_iter()
        .next()
        .expect("one appended event")
        .event;
    assert_eq!(
        store
            .get_event(context_id, event.id)
            .expect("public event read"),
        Some(event.clone())
    );
    assert_eq!(
        store
            .list_stream_events(
                EventStreamSelector {
                    context_id,
                    connector_id: CONNECTOR.into(),
                    source_instance_ref: "live-append-source".into(),
                },
                EventRange::Count {
                    after_seq: None,
                    limit: NonZeroU32::new(2).expect("nonzero"),
                },
            )
            .expect("public stream read"),
        vec![event]
    );

    drop(store);
    let db = Database::open(&db_path).expect("SQL inspection database");
    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    assert_eq!(
        scalar_count(
            &db,
            "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
            params.clone(),
        ),
        1
    );
    assert_eq!(
        scalar_count(
            &db,
            "SELECT COUNT(*) AS n FROM event_source_checkpoints WHERE context_id = $context_id",
            params,
        ),
        0,
        "a live append has no file-source commit to persist"
    );
}

#[test]
fn two_file_sources_can_checkpoint_the_same_unique_event() {
    let root = tempfile::tempdir().expect("tempdir");
    let db_path = root.path().join("file-import.db");
    let (store, context_id) = open_store(db_path.clone());
    let bytes = b"same source record\n";

    let first = store
        .import_event_stream(
            &mut OneRecordSource::new("source-instance-a", bytes.len() as u64),
            &mut Cursor::new(bytes.to_vec()),
            import_options(context_id, "fixture://checkpoint-count/a"),
        )
        .expect("first public file import");
    let second = store
        .import_event_stream(
            &mut OneRecordSource::new("source-instance-b", bytes.len() as u64),
            &mut Cursor::new(bytes.to_vec()),
            import_options(context_id, "fixture://checkpoint-count/b"),
        )
        .expect("second source accepts the already-present immutable event");
    assert_eq!((first.inserted, first.already_present), (1, 0));
    assert_eq!((second.inserted, second.already_present), (0, 1));

    drop(store);
    let db = Database::open(&db_path).expect("SQL inspection database");
    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    let unique_event_rows = scalar_count(
        &db,
        "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
        params.clone(),
    );
    let latest_sequences = checkpoint_latest_sequences(&db, context_id);

    assert_eq!(
        unique_event_rows, 1,
        "the immutable natural key admits one event row"
    );
    assert_eq!(
        latest_sequences,
        vec![1, 1],
        "one checkpoint row per source instance"
    );
    assert_eq!(latest_sequences.iter().sum::<i64>(), 2);
    assert_ne!(
        unique_event_rows,
        latest_sequences.iter().sum::<i64>(),
        "source checkpoint watermarks cannot be summed as unique captured events"
    );
}
