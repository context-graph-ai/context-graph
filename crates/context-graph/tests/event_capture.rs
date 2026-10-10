//! Executable contract for the event capture core.  The source fixture below
//! only supplies records to the private adapter boundary; policy enrollment and
//! delivery outcomes use ContextDB's authenticated in-process sync path.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Cursor};
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use chrono::{TimeZone, Utc};
use context_graph::{
    AdaptedSourceRecord, AppendEventBatch, CgError, CreateContext, EndOfSource,
    EventAdapterDeclaration, EventAppendDisposition, EventBatchLimits, EventCoverage,
    EventDeliveryState, EventPlacementPolicy, EventPlacementState, EventRange, EventRenderRole,
    EventSourceAdapter, EventSourceIdentity, EventSourceRead, EventStreamSelector,
    ImportEventOptions, SourceRecordPosition, Store, StoreConfig,
};
use contextdb_core::{TenantId, Value as DbValue};
use contextdb_engine::Database;
use contextdb_server::sync_client::ApplicationTablePolicyExpectation;
use contextdb_server::{FabricIdentity, InProcessBroker, SyncClient, SyncServer};
use serde_json::json;
use tempfile::TempDir;

const EVENT_POLICY: &str =
    "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE DELIVERY MANIFEST OVER event_anchors";
const ANCHOR_POLICY: &str = "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE";

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: "source-oracle".into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/order".into()],
    }
}

fn draft(key: &str, text: &str) -> context_graph::EventDraft {
    context_graph::EventDraft {
        session_ref: "session-a".into(),
        source_event_key: key.into(),
        source_position: json!({"record": key, "order": 0}),
        occurred_at: Some(Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()),
        author: "source-author".into(),
        kind: "source-kind".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(text.into()),
        payload: json!({"text": text, "source_fact": "fixture"}),
        anchors: json!({"file": "src/lib.rs"}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn context(store: &Store, name: &str) -> context_graph::ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create context")
        .id
}

fn store(root: &TempDir, name: &str, placement: Option<EventPlacementPolicy>) -> Store {
    Store::open(StoreConfig {
        db_path: root.path().join(format!("{name}.db")),
        event_placement: placement,
        ..StoreConfig::default()
    })
    .expect("open store")
}

fn append(
    store: &Store,
    context_id: context_graph::ContextId,
    drafts: Vec<context_graph::EventDraft>,
) -> context_graph::AppendEventReceipt {
    append_with_declaration(store, context_id, declaration(), drafts)
}

fn append_with_declaration(
    store: &Store,
    context_id: context_graph::ContextId,
    declaration: EventAdapterDeclaration,
    drafts: Vec<context_graph::EventDraft>,
) -> context_graph::AppendEventReceipt {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration,
            "source-instance-a",
            drafts,
        ))
        .expect("append accepted source events")
}

fn stream(context_id: context_graph::ContextId) -> EventStreamSelector {
    stream_for_source(context_id, "source-instance-a")
}

fn stream_for_source(
    context_id: context_graph::ContextId,
    source_instance_ref: &str,
) -> EventStreamSelector {
    EventStreamSelector {
        context_id,
        connector_id: "source-oracle".into(),
        source_instance_ref: source_instance_ref.into(),
    }
}

fn all_events(
    store: &Store,
    context_id: context_graph::ContextId,
) -> Vec<context_graph::ConversationEvent> {
    store
        .list_stream_events(
            stream(context_id),
            EventRange::Count {
                after_seq: None,
                limit: NonZeroU32::new(32).unwrap(),
            },
        )
        .expect("read source stream")
}

fn all_events_for_source(
    store: &Store,
    context_id: context_graph::ContextId,
    source_instance_ref: &str,
) -> Vec<context_graph::ConversationEvent> {
    store
        .list_stream_events(
            stream_for_source(context_id, source_instance_ref),
            EventRange::Count {
                after_seq: None,
                limit: NonZeroU32::new(32).unwrap(),
            },
        )
        .expect("read source stream")
}

enum ScriptStep {
    Record(AdaptedSourceRecord),
    Fail(&'static str),
}

struct ScriptedSource {
    declaration: EventAdapterDeclaration,
    identity: EventSourceIdentity,
    steps: VecDeque<ScriptStep>,
    identify_calls: Arc<AtomicUsize>,
    decode_calls: Arc<AtomicUsize>,
}

impl ScriptedSource {
    fn records(records: Vec<AdaptedSourceRecord>) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let identify_calls = Arc::new(AtomicUsize::new(0));
        let decode_calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                declaration: declaration(),
                identity: EventSourceIdentity {
                    source_instance_ref: "source-instance-a".into(),
                    session_ref: "session-a".into(),
                    provisional_empty: false,
                },
                steps: records.into_iter().map(ScriptStep::Record).collect(),
                identify_calls: identify_calls.clone(),
                decode_calls: decode_calls.clone(),
            },
            identify_calls,
            decode_calls,
        )
    }

    fn failing_after(records: Vec<AdaptedSourceRecord>) -> Self {
        let (mut source, _, _) = Self::records(records);
        source
            .steps
            .push_back(ScriptStep::Fail("simulated interruption"));
        source
    }
}

impl EventSourceAdapter for ScriptedSource {
    fn declaration(&self) -> &EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        reader: &mut dyn BufRead,
        _source_locator_ref: &str,
        _session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        self.identify_calls.fetch_add(1, Ordering::SeqCst);
        let mut bounded_identity_prefix = String::new();
        reader
            .read_line(&mut bounded_identity_prefix)
            .map_err(|error| CgError::Engine(error.to_string()))?;
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
        let mut record_bytes = Vec::new();
        reader
            .read_until(b'\n', &mut record_bytes)
            .map_err(|error| CgError::Engine(error.to_string()))?;
        if record_bytes.is_empty() {
            return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
        }
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        match self
            .steps
            .pop_front()
            .expect("fixture supplies each source record")
        {
            ScriptStep::Record(record) => Ok(EventSourceRead::Record(record)),
            ScriptStep::Fail(reason) => Err(CgError::Engine(reason.into())),
        }
    }
}

/// A provider-neutral source whose identity is provisional until it sees its first nonblank record.
/// It skips physical whitespace during record parsing, as a JSONL-style connector does.
struct WhitespaceCheckpointSource {
    declaration: EventAdapterDeclaration,
    record_end_offset: u64,
    identify_calls: Arc<AtomicUsize>,
    decode_calls: Arc<AtomicUsize>,
}

impl WhitespaceCheckpointSource {
    fn new(record_end_offset: u64) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let identify_calls = Arc::new(AtomicUsize::new(0));
        let decode_calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                declaration: declaration(),
                record_end_offset,
                identify_calls: identify_calls.clone(),
                decode_calls: decode_calls.clone(),
            },
            identify_calls,
            decode_calls,
        )
    }
}

impl EventSourceAdapter for WhitespaceCheckpointSource {
    fn declaration(&self) -> &EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        reader: &mut dyn BufRead,
        _source_locator_ref: &str,
        _session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        self.identify_calls.fetch_add(1, Ordering::SeqCst);
        let mut line = Vec::new();
        loop {
            line.clear();
            reader
                .read_until(b'\n', &mut line)
                .map_err(|error| CgError::Engine(error.to_string()))?;
            if line.is_empty() {
                return Ok(EventSourceIdentity {
                    source_instance_ref: "provisional-whitespace-source".into(),
                    session_ref: "session-a".into(),
                    provisional_empty: true,
                });
            }
            if !line.iter().all(u8::is_ascii_whitespace) {
                return Ok(EventSourceIdentity {
                    source_instance_ref: "first-record-source".into(),
                    session_ref: "session-a".into(),
                    provisional_empty: false,
                });
            }
        }
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
        loop {
            line.clear();
            reader
                .read_until(b'\n', &mut line)
                .map_err(|error| CgError::Engine(error.to_string()))?;
            if line.is_empty() {
                return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
            }
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            self.decode_calls.fetch_add(1, Ordering::SeqCst);
            if line != b"first\n" {
                return Err(CgError::Engine(
                    "fixture received an unexpected record".into(),
                ));
            }
            return Ok(EventSourceRead::Record(adapted(
                self.record_end_offset,
                "first",
                "first record",
            )));
        }
    }
}

fn adapted(offset: u64, key: &str, text: &str) -> AdaptedSourceRecord {
    AdaptedSourceRecord {
        drafts: vec![draft(key, text)],
        checkpoint_after_record: json!({"opaque_checkpoint": offset}),
        source_position_after_record: SourceRecordPosition {
            byte_offset: offset,
            coordinate: json!({"line_end": offset}),
        },
    }
}

fn import_options(context_id: context_graph::ContextId) -> ImportEventOptions {
    ImportEventOptions {
        context_id,
        source_locator_ref: "locator:source-a".into(),
        session_override: None,
        batch_limits: EventBatchLimits {
            max_events: NonZeroUsize::new(1).unwrap(),
            max_normalized_bytes: NonZeroUsize::new(1024 * 1024).unwrap(),
        },
    }
}

fn import(
    store: &Store,
    context_id: context_graph::ContextId,
    source: &mut ScriptedSource,
    bytes: &[u8],
) -> context_graph::Result<context_graph::ImportEventReceipt> {
    store.import_event_stream(
        source,
        &mut Cursor::new(bytes.to_vec()),
        import_options(context_id),
    )
}

fn import_whitespace_source(
    store: &Store,
    context_id: context_graph::ContextId,
    source: &mut WhitespaceCheckpointSource,
    bytes: &[u8],
) -> context_graph::Result<context_graph::ImportEventReceipt> {
    store.import_event_stream(
        source,
        &mut Cursor::new(bytes.to_vec()),
        import_options(context_id),
    )
}

#[test]
fn append_derives_occurrence_identity_and_source_order_without_caller_control() {
    let root = TempDir::new().expect("tempdir");
    let first = store(&root, "first", None);
    let second = store(&root, "second", None);
    let session_variant = store(&root, "session-variant", None);
    let first_context = context(&first, "event identity");
    let second_context = context(&second, "event identity");
    let session_variant_context = context(&session_variant, "event identity");

    let first_receipt = append(
        &first,
        first_context,
        vec![draft("one", "same message"), draft("two", "same message")],
    );
    let second_receipt = append(&second, second_context, vec![draft("one", "same message")]);
    let events = all_events(&first, first_context);
    let distinct_context = context(&first, "different event identity context");
    let different_context = append(&first, distinct_context, vec![draft("one", "same message")]);
    let mut different_session_draft = draft("one", "same message");
    different_session_draft.session_ref = "session-b".into();
    let different_session = append(
        &session_variant,
        session_variant_context,
        vec![different_session_draft],
    );
    let mut different_connector_declaration = declaration();
    different_connector_declaration.connector_id = "source-oracle-other".into();
    let different_connector = append_with_declaration(
        &first,
        first_context,
        different_connector_declaration,
        vec![draft("one", "same message")],
    );

    assert_eq!(
        events.iter().map(|event| event.seq).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_ne!(
        events[0].id, events[1].id,
        "distinct source occurrences stay distinct"
    );
    assert_eq!(
        events[0].id, second_receipt.events[0].event.id,
        "natural identity converges across nodes"
    );
    assert_eq!(first_receipt.first_seq, Some(1));
    assert_eq!(first_receipt.last_seq, Some(2));
    assert_eq!(
        events[0].payload,
        json!({"text": "same message", "source_fact": "fixture"})
    );
    assert_eq!(events[0].payload, second_receipt.events[0].event.payload);
    assert_eq!(events[0].payload, different_context.events[0].event.payload);
    assert_eq!(events[0].payload, different_session.events[0].event.payload);
    assert_eq!(
        events[0].seq, different_session.events[0].event.seq,
        "session comparison holds source sequence equal"
    );
    assert_eq!(
        events[0].payload,
        different_connector.events[0].event.payload
    );
    assert_ne!(
        events[0].id, different_context.events[0].event.id,
        "Context is part of an occurrence identity"
    );
    assert_ne!(
        events[0].id, different_session.events[0].event.id,
        "session is part of an occurrence identity"
    );
    assert_ne!(
        events[0].id, different_connector.events[0].event.id,
        "connector is part of an occurrence identity"
    );
}

fn capture_stream_summary(
    store: &Store,
    context_id: context_graph::ContextId,
) -> Vec<Vec<DbValue>> {
    let params = HashMap::from([("context_id".into(), DbValue::Uuid(context_id.as_uuid()))]);
    let db = store.sync_database();
    let groups = db.execute(
        "SELECT connector_id, source_instance_ref, COUNT(*) AS event_count FROM events WHERE context_id = $context_id GROUP BY connector_id, source_instance_ref",
        &params,
    ).expect("read maintained capture stream counts");
    let rows = db.execute(
        "SELECT connector_id, session_ref, source_instance_ref, seq FROM events WHERE context_id = $context_id",
        &params,
    ).expect("read fixture events for test-only session breakdown");
    let mut sessions = std::collections::BTreeMap::new();
    for row in &rows.rows {
        let [
            DbValue::Text(connector),
            DbValue::Text(session),
            DbValue::Text(source),
            DbValue::Int64(seq),
        ] = row.as_slice()
        else {
            panic!("unexpected fixture event row: {row:?}");
        };
        let entry = sessions
            .entry((connector.clone(), session.clone(), source.clone()))
            .or_insert((0_i64, 0_i64));
        entry.0 += 1;
        entry.1 = entry.1.max(*seq);
    }
    for group in &groups.rows {
        let [connector, source, count] = group.as_slice() else {
            panic!("unexpected maintained group: {group:?}");
        };
        let mut newest_params = params.clone();
        newest_params.insert("connector".into(), connector.clone());
        newest_params.insert("source".into(), source.clone());
        let newest = db.execute(
            "SELECT seq FROM events WHERE context_id = $context_id AND connector_id = $connector AND source_instance_ref = $source ORDER BY seq DESC LIMIT 1",
            &newest_params,
        ).expect("read indexed newest sequence");
        let stream_rows: Vec<_> = rows
            .rows
            .iter()
            .filter(|row| &row[0] == connector && &row[2] == source)
            .collect();
        assert_eq!(
            count,
            &DbValue::Int64(stream_rows.len() as i64),
            "maintained stream count equals fixture rows"
        );
        let max = stream_rows
            .iter()
            .map(|row| match row[3] {
                DbValue::Int64(seq) => seq,
                _ => panic!("sequence is an integer"),
            })
            .max()
            .unwrap();
        assert_eq!(
            newest.rows,
            vec![vec![DbValue::Int64(max)]],
            "indexed newest entry equals fixture frontier"
        );
    }
    sessions
        .into_iter()
        .map(|((connector, session, source), (count, max))| {
            vec![
                DbValue::Text(connector),
                DbValue::Text(session),
                DbValue::Text(source),
                DbValue::Int64(count),
                DbValue::Int64(max),
            ]
        })
        .collect()
}

#[test]
fn retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged() {
    let root = TempDir::new().expect("tempdir");
    let store = store(&root, "retry", None);
    let context_id = context(&store, "atomic retry");
    let initial = append(&store, context_id, vec![draft("one", "original")]);
    let summary = capture_stream_summary(&store, context_id);
    assert_eq!(summary[0][3..], [DbValue::Int64(1), DbValue::Int64(1)]);
    let retry = append(&store, context_id, vec![draft("one", "original")]);
    assert_eq!(
        capture_stream_summary(&store, context_id),
        summary,
        "retry leaves the authoritative stream frontier unchanged"
    );

    assert_eq!(retry.inserted, 0);
    assert_eq!(retry.already_present, 1);
    assert!(!retry.changed);
    assert!(retry.transaction_id.is_none());
    assert_eq!(
        retry.events[0].disposition,
        EventAppendDisposition::AlreadyPresent
    );

    let (mut first_import, source_calls, _) =
        ScriptedSource::records(vec![adapted(6, "one", "original")]);
    let _ = initial;
    let _ = import(&store, context_id, &mut first_import, b"first\n").expect("checkpoint baseline");
    assert_eq!(source_calls.load(Ordering::SeqCst), 1);

    let (mut divergent, _, _) = ScriptedSource::records(vec![AdaptedSourceRecord {
        drafts: vec![draft("two", "must roll back"), draft("one", "different")],
        checkpoint_after_record: json!({"opaque_checkpoint": 13}),
        source_position_after_record: SourceRecordPosition {
            byte_offset: 13,
            coordinate: json!({"line_end": 13}),
        },
    }]);
    let error = import(&store, context_id, &mut divergent, b"first\nsecond\n")
        .expect_err("same source identity with changed content is a divergence");
    assert!(matches!(error, CgError::SourceEventDiverged { .. }));
    assert_eq!(
        all_events(&store, context_id).len(),
        1,
        "the new row was rolled back with the divergent row"
    );

    assert_eq!(
        capture_stream_summary(&store, context_id),
        summary,
        "divergent capture rolls the stream facts back with its events"
    );

    let (mut recovery, _, decoded) = ScriptedSource::records(vec![adapted(13, "two", "recovered")]);
    import(&store, context_id, &mut recovery, b"first\nsecond\n")
        .expect("the failed batch did not advance checkpoint");
    assert_eq!(
        decoded.load(Ordering::SeqCst),
        1,
        "the uncommitted tail is parsed again"
    );
    assert_eq!(all_events(&store, context_id).len(), 2);
    assert_eq!(
        capture_stream_summary(&store, context_id)[0][3..],
        [DbValue::Int64(2), DbValue::Int64(2)],
        "committed retry advances count and frontier together"
    );
}

#[test]
fn import_verifies_prefix_then_adds_only_tail_or_refuses_rewritten_history() {
    let root = TempDir::new().expect("tempdir");
    let store = store(&root, "stream", None);
    let context_id = context(&store, "stream recovery");
    let (mut initial, identified, decoded) =
        ScriptedSource::records(vec![adapted(6, "one", "first")]);
    import(&store, context_id, &mut initial, b"first\n").expect("initial import");
    assert_eq!(identified.load(Ordering::SeqCst), 1);
    assert_eq!(decoded.load(Ordering::SeqCst), 1);

    let (mut caught_up, identified, decoded) = ScriptedSource::records(vec![]);
    let receipt =
        import(&store, context_id, &mut caught_up, b"first\n").expect("caught up reimport");
    assert_eq!(
        identified.load(Ordering::SeqCst),
        0,
        "known locator loads stored identity"
    );
    assert_eq!(
        decoded.load(Ordering::SeqCst),
        0,
        "caught-up prefix is hashed but never decoded"
    );
    assert_eq!(receipt.inserted, 0);

    let (mut tail, _, decoded) = ScriptedSource::records(vec![adapted(13, "two", "second")]);
    import(&store, context_id, &mut tail, b"first\nsecond\n").expect("append only tail");
    assert_eq!(decoded.load(Ordering::SeqCst), 1);
    assert_eq!(
        all_events(&store, context_id)
            .iter()
            .map(|event| event.seq)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );

    let (mut rewritten, _, decoded) = ScriptedSource::records(vec![]);
    let error = import(&store, context_id, &mut rewritten, b"first\nchanged\n")
        .expect_err("rewritten committed prefix refuses");
    assert!(matches!(error, CgError::SourceHistoryDiverged { .. }));
    assert_eq!(
        decoded.load(Ordering::SeqCst),
        0,
        "divergence is found before tail decoding"
    );
    assert_eq!(
        all_events(&store, context_id).len(),
        2,
        "refusal changes no accepted event"
    );

    let (mut mismatched_boundary, _, _) =
        ScriptedSource::records(vec![adapted(20, "three", "third")]);
    let error = import(
        &store,
        context_id,
        &mut mismatched_boundary,
        b"first\nsecond\nthird\nfourth\n",
    )
    .expect_err("adapter offset that differs from consumed bytes refuses");
    assert!(matches!(error, CgError::SourceHistoryDiverged { .. }));

    let mut interrupted = ScriptedSource::failing_after(vec![adapted(19, "three", "third")]);
    let error = import(
        &store,
        context_id,
        &mut interrupted,
        b"first\nsecond\nthird\nfourth\n",
    )
    .expect_err("simulated crash after a committed record");
    assert!(matches!(error, CgError::Engine(_)));
    let (mut resumed, _, decoded) = ScriptedSource::records(vec![adapted(26, "four", "fourth")]);
    import(
        &store,
        context_id,
        &mut resumed,
        b"first\nsecond\nthird\nfourth\n",
    )
    .expect("resume at durable boundary");
    assert_eq!(decoded.load(Ordering::SeqCst), 1);
    assert_eq!(
        all_events(&store, context_id)
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        vec!["one", "two", "three", "four"]
    );
}

#[test]
fn whitespace_checkpoint_reopens_upgrades_identity_and_refuses_a_changed_prefix_before_decode() {
    let whitespace = b" \t\n\n";
    let first_record = b"first\n";
    let appended = [whitespace.as_slice(), first_record.as_slice()].concat();
    let changed_whitespace = b" \r\n\n";
    assert_eq!(whitespace.len(), changed_whitespace.len());

    let root = TempDir::new().expect("tempdir");
    let initial_store = store(&root, "whitespace-checkpoint", None);
    let context_id = context(&initial_store, "whitespace checkpoint");

    let (mut empty, _, empty_decoded) = WhitespaceCheckpointSource::new(0);
    let empty_receipt =
        import_whitespace_source(&initial_store, context_id, &mut empty, whitespace)
            .expect("whitespace-only source commits a completed provisional checkpoint");
    assert!(empty_receipt.source.provisional_empty);
    assert_eq!(empty_receipt.inserted, 0);
    assert_eq!(empty_decoded.load(Ordering::SeqCst), 0);
    drop(initial_store);

    let store = store(&root, "whitespace-checkpoint", None);
    let (mut first, first_identified, first_decoded) =
        WhitespaceCheckpointSource::new(appended.len() as u64);
    let first_receipt = import_whitespace_source(&store, context_id, &mut first, &appended)
        .expect("unchanged whitespace prefix verifies before the first record upgrades identity");
    assert_eq!(
        first_receipt.source.source_instance_ref,
        "first-record-source"
    );
    assert!(!first_receipt.source.provisional_empty);
    assert_eq!(first_receipt.inserted, 1);
    assert_eq!(first_identified.load(Ordering::SeqCst), 1);
    assert_eq!(first_decoded.load(Ordering::SeqCst), 1);
    assert_eq!(
        all_events_for_source(&store, context_id, "first-record-source")
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        vec!["first"]
    );

    let (mut retry, retry_identified, retry_decoded) =
        WhitespaceCheckpointSource::new(appended.len() as u64);
    let retry_receipt = import_whitespace_source(&store, context_id, &mut retry, &appended)
        .expect("an unchanged upgraded source is a no-op");
    assert_eq!(retry_receipt.inserted, 0);
    assert_eq!(retry_identified.load(Ordering::SeqCst), 0);
    assert_eq!(retry_decoded.load(Ordering::SeqCst), 0);

    let changed = [changed_whitespace.as_slice(), first_record.as_slice()].concat();
    let (mut rewritten, rewritten_identified, rewritten_decoded) =
        WhitespaceCheckpointSource::new(changed.len() as u64);
    let error = import_whitespace_source(&store, context_id, &mut rewritten, &changed)
        .expect_err("a changed committed whitespace prefix refuses before record decode or write");
    assert!(matches!(error, CgError::SourceHistoryDiverged { .. }));
    assert_eq!(rewritten_identified.load(Ordering::SeqCst), 0);
    assert_eq!(rewritten_decoded.load(Ordering::SeqCst), 0);
    assert_eq!(
        all_events_for_source(&store, context_id, "first-record-source").len(),
        1,
        "refusing the rewritten prefix preserves the upgraded event"
    );
}

#[test]
fn event_reads_are_context_scoped_and_return_source_sequence_order() {
    let root = TempDir::new().expect("tempdir");
    let store = store(&root, "reads", None);
    let allowed = context(&store, "allowed event context");
    let other = context(&store, "other event context");
    let mut late = draft("late-time", "late");
    late.occurred_at = Some(Utc.timestamp_opt(1_700_000_200, 0).single().unwrap());
    let mut early = draft("early-time", "early");
    early.occurred_at = Some(Utc.timestamp_opt(1_700_000_100, 0).single().unwrap());
    let mut missing_time = draft("missing-time", "missing");
    missing_time.occurred_at = None;
    append(&store, allowed, vec![late, early, missing_time]);
    append(&store, other, vec![draft("other", "private")]);

    let ordered = all_events(&store, allowed);
    assert_eq!(
        ordered
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        vec!["late-time", "early-time", "missing-time"]
    );
    assert_eq!(
        store
            .get_event(other, ordered[0].id)
            .expect("scoped point read"),
        None
    );
    assert_eq!(all_events(&store, other).len(), 1);
}

struct AuthenticatedHub {
    broker: InProcessBroker,
    tenant: TenantId,
    stop: Arc<std::sync::atomic::AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl AuthenticatedHub {
    async fn stop(self) {
        self.stop.store(true, Ordering::SeqCst);
        self.task.await.expect("hub task stops");
    }
}

async fn authenticated_hub() -> AuthenticatedHub {
    let broker = InProcessBroker::new();
    let tenant = TenantId::from("event-capture-tenant");
    let db = Arc::new(Database::open_memory());
    let identity = Arc::new(FabricIdentity::generate());
    let hub_node = identity.node_id();
    let server = Arc::new(
        SyncServer::with_authenticated_transport_and_identity_for_test(
            db.clone(),
            broker.server_as(&hub_node),
            tenant.clone(),
            hub_node,
            identity,
        )
        .expect("admit engine read memory"),
    );
    for (table, policy) in [("events", EVENT_POLICY), ("event_anchors", ANCHOR_POLICY)] {
        db.execute(
            &format!("DECLARE TENANT TABLE POLICY {table} {policy}"),
            &HashMap::<String, DbValue>::new(),
        )
        .expect("hub declares event policy before enrollment");
    }
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let task = tokio::spawn({
        let server = server.clone();
        let stop = stop.clone();
        async move { server.run_until(stop).await }
    });
    broker
        .wait_for_registered_route_for_test(&contextdb_server::subjects::binding_subject(
            tenant.as_str(),
        ))
        .await;
    AuthenticatedHub {
        broker,
        tenant,
        stop,
        task,
    }
}

fn policy_expectation() -> ApplicationTablePolicyExpectation {
    ApplicationTablePolicyExpectation::new()
        .expect_table("events", EVENT_POLICY)
        .expect("event policy grammar")
        .expect_table("event_anchors", ANCHOR_POLICY)
        .expect("anchor policy grammar")
}

async fn bind_tenant_store(
    hub: &AuthenticatedHub,
    root: &TempDir,
    name: &str,
) -> (Store, SyncClient) {
    let store = store(root, name, Some(EventPlacementPolicy::TenantHub));
    let identity = Arc::new(FabricIdentity::generate());
    let node_id = identity.node_id();
    let client = SyncClient::with_authenticated_transport_and_identity_for_test(
        store.sync_database(),
        hub.broker.client_as(&node_id),
        hub.tenant.clone(),
        identity,
    )
    .expect("admit engine read memory");
    let binding = client
        .bind_application_table_policy(policy_expectation())
        .await
        .expect("authenticated hub policy binding");
    let receipt = store
        .bind_authenticated_tenant_policy(binding)
        .expect("store installs only the authenticated binding");
    assert_eq!(receipt.state, EventPlacementState::TenantHubBound);
    (store, client)
}

#[tokio::test(flavor = "multi_thread")]
async fn local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding()
 {
    let root = TempDir::new().expect("tempdir");
    let local = store(&root, "local", None);
    assert_eq!(
        local.event_placement_state().expect("local placement"),
        EventPlacementState::LocalOnly
    );
    assert!(
        local.sync_database().table_meta("events").is_some(),
        "fresh local store is ready with the complete schema"
    );
    assert_eq!(
        local
            .event_delivery_status(context(&local, "local context"))
            .expect("local delivery state")
            .state,
        EventDeliveryState::Disabled
    );

    let unbound = store(&root, "unbound", Some(EventPlacementPolicy::TenantHub));
    assert_eq!(
        unbound.event_placement_state().expect("unbound placement"),
        EventPlacementState::TenantHubUnbound
    );
    assert!(unbound.sync_database().table_meta("events").is_none());
    assert!(matches!(
        unbound.create_context(CreateContext {
            name: "unbound context".into(),
            labels: vec![],
            properties: Default::default(),
        }),
        Err(CgError::TenantPolicyBindingRequired { .. })
    ));
    let unbound_context = context_graph::ContextId::new_v7();
    assert!(matches!(
        unbound.append_event_batch(AppendEventBatch::live(
            unbound_context,
            declaration(),
            "source-instance-a",
            vec![draft("one", "blocked")]
        )),
        Err(CgError::TenantPolicyBindingRequired { .. })
    ));
    assert_eq!(
        unbound
            .event_delivery_status(unbound_context)
            .expect("blocked delivery state")
            .state,
        EventDeliveryState::Blocked
    );

    let hub = authenticated_hub().await;
    let (bound, client) = bind_tenant_store(&hub, &root, "bound").await;
    assert!(bound.sync_database().table_meta("events").is_some());
    let bound_context = context(&bound, "bound context");
    append(&bound, bound_context, vec![draft("one", "accepted")]);
    client.shutdown().await;
    hub.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn authenticated_custody_counts_equivalent_full_units_and_refuses_changed_members() {
    let root = TempDir::new().expect("tempdir");
    let hub = authenticated_hub().await;
    let (first, first_client) = bind_tenant_store(&hub, &root, "first-edge").await;
    let (second, second_client) = bind_tenant_store(&hub, &root, "second-edge").await;
    let _unrelated_context = context(&first, "unrelated committed work");
    let first_context = context(&first, "custody context");
    let second_context = context(&second, "custody context");

    let first_receipt = append(
        &first,
        first_context,
        vec![draft("same", "complete member")],
    );
    let second_receipt = append(
        &second,
        second_context,
        vec![draft("same", "complete member")],
    );
    assert_ne!(
        first_receipt
            .transaction_id
            .expect("first event commit has a node-local transaction ID"),
        second_receipt
            .transaction_id
            .expect("second event commit has a node-local transaction ID"),
        "prior work on one node changes only node-local commit bookkeeping"
    );
    first_client
        .push()
        .await
        .expect("first edge delivers full event and anchors");
    second_client
        .push()
        .await
        .expect("second edge receives authenticated equivalent outcome");
    let first_event_rows = first
        .sync_database()
        .execute("SELECT * FROM events", &HashMap::new())
        .expect("read first shared row")
        .rows;
    let second_event_rows = second
        .sync_database()
        .execute("SELECT * FROM events", &HashMap::new())
        .expect("read second shared row")
        .rows;
    assert_eq!(
        first_event_rows, second_event_rows,
        "identical accepted events retain byte-for-byte equal shared rows"
    );
    let first_anchor_rows = first
        .sync_database()
        .execute("SELECT * FROM event_anchors", &HashMap::new())
        .expect("read first complete anchor rows")
        .rows;
    let second_anchor_rows = second
        .sync_database()
        .execute("SELECT * FROM event_anchors", &HashMap::new())
        .expect("read second complete anchor rows")
        .rows;
    assert_eq!(
        first_anchor_rows.len(),
        1,
        "the complete event projects its declared anchor"
    );
    assert_eq!(
        second_anchor_rows.len(),
        1,
        "the equivalent event projects its declared anchor"
    );
    for (store, event_id) in [
        (&first, first_receipt.events[0].event.id),
        (&second, second_receipt.events[0].event.id),
    ] {
        let anchor = store
            .sync_database()
            .execute(
                "SELECT event_id, key, value FROM event_anchors",
                &HashMap::new(),
            )
            .expect("read named normalized anchor columns");
        let expected = vec![
            DbValue::Uuid(event_id.as_uuid()),
            DbValue::Text("file".into()),
            DbValue::Text("src/lib.rs".into()),
        ];
        assert_eq!(
            anchor.columns,
            vec!["event_id", "key", "value"],
            "the engine maps the named anchor projection in SELECT order"
        );
        assert_eq!(
            anchor.rows,
            vec![expected.clone()],
            "the normalized anchor retains its event, key, and canonical value"
        );
        assert_ne!(
            anchor.rows,
            vec![vec![
                expected[0].clone(),
                DbValue::Text("profile".into()),
                expected[2].clone(),
            ]],
            "a different anchor key cannot satisfy exact recall"
        );
        assert_ne!(
            anchor.rows,
            vec![vec![
                expected[0].clone(),
                expected[1].clone(),
                DbValue::Text("src/other.rs".into()),
            ]],
            "a different canonical anchor value cannot satisfy exact recall"
        );
    }
    assert_eq!(
        first_anchor_rows, second_anchor_rows,
        "identical accepted events retain byte-for-byte equal complete anchor rows"
    );
    assert_eq!(
        first
            .event_delivery_status(first_context)
            .expect("accepted status")
            .accepted,
        1
    );
    assert_eq!(
        second
            .event_delivery_status(second_context)
            .expect("equivalent status")
            .equivalent,
        1
    );

    let (changed, changed_client) = bind_tenant_store(&hub, &root, "changed-edge").await;
    let changed_context = context(&changed, "custody context");
    let mut changed_member = draft("same", "complete member");
    changed_member.anchors = json!({"file": "src/changed.rs"});
    append(&changed, changed_context, vec![changed_member]);
    changed_client
        .push()
        .await
        .expect("hub returns a refusal outcome");
    let status = changed
        .event_delivery_status(changed_context)
        .expect("refused delivery status");
    assert_eq!(
        status.refused, 1,
        "keep-first disagreement never becomes equivalent"
    );
    assert_eq!(status.pending, 0);

    changed_client.shutdown().await;
    second_client.shutdown().await;
    first_client.shutdown().await;
    hub.stop().await;
}
