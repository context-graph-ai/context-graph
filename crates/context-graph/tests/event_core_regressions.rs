//! Event capture preserves checkpoints, source identity, and committed prefixes under races.
//!
//! Each case drives the public Store surface. The line adapter is deliberately
//! independent of store internals: its source positions are fixture constants,
//! and it has no knowledge of persisted checkpoints.

use std::collections::HashMap;
use std::io::{BufRead, Cursor, Read, Seek, SeekFrom};
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};

use chrono::{Duration, TimeZone, Utc};
use context_graph::{
    AdaptedSourceRecord, AppendEventBatch, CgError, CreateContext, EndOfSource,
    EventAdapterDeclaration, EventBatchLimits, EventCoverage, EventDraft, EventRange,
    EventRenderRole, EventSourceAdapter, EventSourceIdentity, EventSourceRead, EventStreamSelector,
    ImportEventOptions, SourceRecordPosition, Store, StoreConfig,
};
use contextdb_core::{ColumnType, Value};
use serde_json::json;

const CONNECTOR: &str = "event-core-regression";
const SOURCE_INSTANCE: &str = "fixture-stream-17";

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-1".into(),
        anchor_declaration_version: "fixture-1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/order".into()],
    }
}

fn store() -> (tempfile::TempDir, Store) {
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("event-core-regressions.db"),
        ..StoreConfig::default()
    })
    .expect("open Store");
    (root, store)
}

fn context(store: &Store, name: &str) -> context_graph::ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id
}

fn draft(key: &str) -> EventDraft {
    EventDraft {
        session_ref: "fixture-session".into(),
        source_event_key: key.into(),
        source_position: json!({"fixture_record": key, "order": 0}),
        occurred_at: Some(
            Utc.timestamp_opt(1_700_000_000, 0)
                .single()
                .expect("timestamp"),
        ),
        author: "fixture".into(),
        kind: "fixture".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("fixture {key}")),
        payload: json!({"fixture": key}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn stream(context_id: context_graph::ContextId) -> EventStreamSelector {
    EventStreamSelector {
        context_id,
        connector_id: CONNECTOR.into(),
        source_instance_ref: SOURCE_INSTANCE.into(),
    }
}

fn import_options(context_id: context_graph::ContextId, locator: &str) -> ImportEventOptions {
    ImportEventOptions {
        context_id,
        source_locator_ref: locator.into(),
        session_override: None,
        batch_limits: EventBatchLimits {
            max_events: NonZeroUsize::new(1).expect("one is nonzero"),
            max_normalized_bytes: NonZeroUsize::new(1024 * 1024).expect("one MiB is nonzero"),
        },
    }
}

/// A source fixture that maps its own line syntax to fixed absolute positions.
/// It deliberately has no persisted checkpoint input; the Store must seek the
/// reader to the durable boundary before asking it to decode the tail.
struct LineAdapter {
    declaration: EventAdapterDeclaration,
    decoded: Arc<AtomicUsize>,
}

impl LineAdapter {
    fn new(decoded: Arc<AtomicUsize>) -> Self {
        Self {
            declaration: declaration(),
            decoded,
        }
    }
}

impl EventSourceAdapter for LineAdapter {
    fn declaration(&self) -> &EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        reader: &mut dyn BufRead,
        _source_locator_ref: &str,
        _session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        let mut identity_prefix = String::new();
        reader
            .read_line(&mut identity_prefix)
            .map_err(|error| CgError::Engine(error.to_string()))?;
        Ok(EventSourceIdentity {
            source_instance_ref: SOURCE_INSTANCE.into(),
            session_ref: "fixture-session".into(),
            provisional_empty: false,
        })
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
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|error| CgError::Engine(error.to_string()))?;
        let (key, byte_offset) = match line.as_str() {
            "" => return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete)),
            "one\n" => ("one", 4),
            "two\n" => ("two", 8),
            "three\n" => ("three", 14),
            unexpected => panic!("fixture source line was unexpected: {unexpected:?}"),
        };
        self.decoded.fetch_add(1, Ordering::SeqCst);
        Ok(EventSourceRead::Record(AdaptedSourceRecord {
            drafts: vec![draft(key)],
            checkpoint_after_record: json!({"fixture_after": byte_offset}),
            source_position_after_record: SourceRecordPosition {
                byte_offset,
                coordinate: json!({"fixture_byte_offset": byte_offset}),
            },
        }))
    }
}

/// Pauses after source identification has completed, while the core verifies
/// the committed prefix. This puts the slow importer after its checkpoint read
/// and before locator attachment without relying on elapsed time.
struct PauseBeforeLocatorAttachment {
    inner: Cursor<Vec<u8>>,
    identified_read: bool,
    pause_armed: bool,
    paused: bool,
    paused_at_checkpoint_read: Sender<()>,
    continue_after_fast_import: Receiver<()>,
}

impl PauseBeforeLocatorAttachment {
    fn new(
        bytes: Vec<u8>,
        paused_at_checkpoint_read: Sender<()>,
        continue_after_fast_import: Receiver<()>,
    ) -> Self {
        Self {
            inner: Cursor::new(bytes),
            identified_read: false,
            pause_armed: false,
            paused: false,
            paused_at_checkpoint_read,
            continue_after_fast_import,
        }
    }
}

impl Read for PauseBeforeLocatorAttachment {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.pause_armed && !self.paused {
            self.paused = true;
            self.paused_at_checkpoint_read
                .send(())
                .expect("test waits for the slow checkpoint verification");
            self.continue_after_fast_import
                .recv()
                .expect("test releases the slow importer after the fast commit");
        }
        let read = self.inner.read(buffer)?;
        if read > 0 {
            self.identified_read = true;
        }
        Ok(read)
    }
}

impl Seek for PauseBeforeLocatorAttachment {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let offset = self.inner.seek(position)?;
        if offset == 0 && self.identified_read && !self.paused {
            self.pause_armed = true;
        }
        Ok(offset)
    }
}

#[test]
fn relocated_multibatch_source_reuses_the_stream_checkpoint_without_redecoding_prefix() {
    let (_root, store) = store();
    let context_id = context(&store, "relocated source");
    let initial_decodes = Arc::new(AtomicUsize::new(0));
    let mut initial_adapter = LineAdapter::new(initial_decodes.clone());
    let mut initial_reader = Cursor::new(b"one\ntwo\n".to_vec());

    let initial = store
        .import_event_stream(
            &mut initial_adapter,
            &mut initial_reader,
            import_options(context_id, "file:///inbox/original.log"),
        )
        .expect("two records commit as two bounded batches");
    assert_eq!(initial.batch_count, 2);
    assert_eq!(initial.inserted, 2);
    assert_eq!(initial_decodes.load(Ordering::SeqCst), 2);

    let relocated_decodes = Arc::new(AtomicUsize::new(0));
    let mut relocated_adapter = LineAdapter::new(relocated_decodes.clone());
    let mut relocated_reader = Cursor::new(b"one\ntwo\nthree\n".to_vec());
    let resumed = store
        .import_event_stream(
            &mut relocated_adapter,
            &mut relocated_reader,
            import_options(context_id, "file:///archive/copied.log"),
        )
        .expect("a copied locator resumes its identified stream");

    assert_eq!(resumed.inserted, 1, "only the copied source tail is new");
    assert_eq!(
        resumed.already_present, 0,
        "the committed prefix is never replayed"
    );
    assert_eq!(relocated_decodes.load(Ordering::SeqCst), 1);
    let events = store
        .list_stream_events(
            stream(context_id),
            EventRange::Count {
                after_seq: None,
                limit: NonZeroU32::new(8).expect("eight is nonzero"),
            },
        )
        .expect("read source stream");
    assert_eq!(
        events
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        vec!["one", "two", "three"]
    );
    assert_eq!(
        events.iter().map(|event| event.seq).collect::<Vec<_>>(),
        vec![1, 2, 3],
        "checkpoint reuse keeps one non-divergent source sequence"
    );

    let rewritten_decodes = Arc::new(AtomicUsize::new(0));
    let mut rewritten_adapter = LineAdapter::new(rewritten_decodes.clone());
    let mut rewritten_reader = Cursor::new(b"one\nTWO\nthree\n".to_vec());
    let rewritten = store
        .import_event_stream(
            &mut rewritten_adapter,
            &mut rewritten_reader,
            import_options(context_id, "file:///archive/rewritten-copy.log"),
        )
        .expect_err("a copied locator still verifies its committed prefix");
    assert!(matches!(rewritten, CgError::SourceHistoryDiverged { .. }));
    assert_eq!(
        rewritten_decodes.load(Ordering::SeqCst),
        0,
        "prefix verification refuses before an adapter decodes a rewritten source"
    );
}

#[test]
fn moved_locator_revalidates_its_checkpoint_before_a_slow_import_can_resume() {
    let (_root, store) = store();
    let context_id = context(&store, "moved locator checkpoint race");
    let initial_decodes = Arc::new(AtomicUsize::new(0));
    let mut initial_adapter = LineAdapter::new(initial_decodes);
    let mut initial_reader = Cursor::new(b"one\n".to_vec());
    store
        .import_event_stream(
            &mut initial_adapter,
            &mut initial_reader,
            import_options(context_id, "file:///inbox/original.log"),
        )
        .expect("commit the old checkpoint before either moved import starts");

    let (slow_checkpoint_read, slow_checkpoint_read_rx) = std::sync::mpsc::channel();
    let (continue_slow, continue_slow_rx) = std::sync::mpsc::channel();
    let slow_store = store.clone();
    let slow_decodes = Arc::new(AtomicUsize::new(0));
    let slow_decodes_in_thread = slow_decodes.clone();
    let slow = std::thread::spawn(move || {
        let mut adapter = LineAdapter::new(slow_decodes_in_thread);
        let mut reader = PauseBeforeLocatorAttachment::new(
            b"one\ntwo\nthree\n".to_vec(),
            slow_checkpoint_read,
            continue_slow_rx,
        );
        slow_store.import_event_stream(
            &mut adapter,
            &mut reader,
            import_options(context_id, "file:///archive/moved.log"),
        )
    });

    slow_checkpoint_read_rx
        .recv()
        .expect("slow import loaded the old checkpoint and is verifying its prefix");

    let fast_decodes = Arc::new(AtomicUsize::new(0));
    let mut fast_adapter = LineAdapter::new(fast_decodes.clone());
    let mut fast_reader = Cursor::new(b"one\ntwo\nthree\n".to_vec());
    let fast = store
        .import_event_stream(
            &mut fast_adapter,
            &mut fast_reader,
            import_options(context_id, "file:///archive/moved.log"),
        )
        .expect("fast import attaches the moved locator and advances its checkpoint");
    assert_eq!(
        fast.inserted, 2,
        "fast import starts from the old checkpoint"
    );
    assert_eq!(fast_decodes.load(Ordering::SeqCst), 2);

    continue_slow
        .send(())
        .expect("release slow attachment after fast checkpoint advancement");
    let slow = slow.join().expect("slow import thread completes");
    assert!(matches!(
        slow,
        Err(CgError::InvalidSourceCommitBoundary { .. })
    ));
    assert_eq!(
        slow_decodes.load(Ordering::SeqCst),
        0,
        "attachment rejects the stale checkpoint before a bounded tail batch decodes"
    );
    assert_eq!(
        store
            .list_stream_events(
                stream(context_id),
                EventRange::Count {
                    after_seq: None,
                    limit: NonZeroU32::new(8).expect("eight is nonzero"),
                },
            )
            .expect("read the fast import result")
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        ["one", "two", "three"],
        "the stale importer changes no event or source progress"
    );
}

#[test]
fn mixed_text_and_number_anchor_values_round_trip_as_two_distinct_rows() {
    let (_root, store) = store();
    let context_id = context(&store, "typed anchors");
    let mut mixed = draft("typed-anchor-event");
    mixed.anchors = json!({"mixed": ["1", 1]});
    let appended = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_INSTANCE,
            vec![mixed],
        ))
        .expect("the accepted typed anchor values insert");
    let event = &appended.events[0].event;
    assert_eq!(appended.inserted, 1);
    assert_eq!(
        event.anchor_count, 2,
        "each accepted typed value has an anchor"
    );
    assert_eq!(
        store
            .get_event(context_id, event.id)
            .expect("point read")
            .expect("accepted event")
            .anchors,
        json!({"mixed": ["1", 1]}),
        "event readback preserves both JSON types"
    );

    let anchors = store
        .sync_database()
        .execute(
            "SELECT original_value FROM event_anchors WHERE context_id = $context_id AND event_id = $event_id AND key = $key",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("event_id".into(), Value::Uuid(event.id.as_uuid())),
                ("key".into(), Value::Text("mixed".into())),
            ]),
        )
        .expect("look up anchors persisted for this event");
    assert_eq!(
        anchors.rows.len(),
        2,
        "the lookup returns both typed anchors"
    );
    let original_value = anchors
        .columns
        .iter()
        .position(|column| column == "original_value")
        .expect("selected original value column");
    let values = anchors
        .rows
        .iter()
        .map(|row| row[original_value].clone())
        .collect::<Vec<_>>();
    assert!(values.contains(&Value::Json(json!("1"))));
    assert!(values.contains(&Value::Json(json!(1))));
}

#[test]
fn millisecond_source_time_round_trips_retries_and_supports_an_exact_one_millisecond_range() {
    let (_root, store) = store();
    let context_id = context(&store, "millisecond source time");
    let occurred_at = Utc
        .timestamp_opt(1_700_000_000, 123_000_000)
        .single()
        .expect("millisecond timestamp");
    let mut timed = draft("millisecond-event");
    timed.occurred_at = Some(occurred_at);
    let first = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_INSTANCE,
            vec![timed.clone()],
        ))
        .expect("append millisecond source time");
    let event_id = first.events[0].event.id;
    assert_eq!(
        store
            .get_event(context_id, event_id)
            .expect("get stored event")
            .expect("event exists")
            .occurred_at,
        Some(occurred_at)
    );

    let retry = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_INSTANCE,
            vec![timed],
        ))
        .expect("retry accepted event");
    assert_eq!(retry.inserted, 0);
    assert_eq!(retry.already_present, 1);
    assert_eq!(retry.events[0].event.occurred_at, Some(occurred_at));

    let in_exact_window = store
        .list_stream_events(
            stream(context_id),
            EventRange::OccurredAt {
                start_inclusive: occurred_at,
                end_exclusive: occurred_at + Duration::milliseconds(1),
                limit: NonZeroU32::new(2).expect("two is nonzero"),
            },
        )
        .expect("read one-millisecond source-time window");
    assert_eq!(in_exact_window.len(), 1);
    assert_eq!(in_exact_window[0].id, event_id);
    assert_eq!(in_exact_window[0].occurred_at, Some(occurred_at));
    let event_table = store
        .sync_database()
        .table_meta("events")
        .expect("events table metadata");
    assert!(
        event_table.columns.iter().all(|column| !matches!(
            column.name.as_str(),
            "occurred_at_seconds" | "occurred_at_nanoseconds"
        )),
        "split source-time columns are absent from the engine schema"
    );
    assert_eq!(
        event_table
            .columns
            .iter()
            .find(|column| column.name == "occurred_at")
            .expect("single occurred_at column")
            .column_type,
        ColumnType::Timestamp,
        "events stores source time as one engine TIMESTAMP column"
    );

    let rows = store
        .sync_database()
        .execute(
            "SELECT id, occurred_at FROM events WHERE context_id = $context_id AND occurred_at >= $start_inclusive AND occurred_at < $end_exclusive",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                (
                    "start_inclusive".into(),
                    Value::Timestamp(occurred_at.timestamp_millis()),
                ),
                (
                    "end_exclusive".into(),
                    Value::Timestamp((occurred_at + Duration::milliseconds(1)).timestamp_millis()),
                ),
            ]),
        )
        .expect("read events through ordinary occurred_at SQL comparison");
    assert_eq!(rows.columns, vec!["id", "occurred_at"]);
    assert_eq!(
        rows.rows,
        vec![vec![
            Value::Uuid(event_id.as_uuid()),
            Value::Timestamp(occurred_at.timestamp_millis()),
        ]],
        "the engine TIMESTAMP value is readable through the ordinary one-millisecond comparison"
    );
}
