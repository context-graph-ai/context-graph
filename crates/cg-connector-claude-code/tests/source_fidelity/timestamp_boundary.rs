use std::ffi::OsString;
use std::io::{BufReader, Cursor};
use std::num::NonZeroU32;

use base64::Engine as _;
use cg_connector_claude_code::{
    CONNECTOR_ID, ClaudeAssemblyContext, ClaudeCodeJsonlAdapter, ClaudeContent, ClaudeMessage,
    ClaudeMessageRole, ClaudeOccurrence, ClaudeOccurrenceAssembler, ClaudeOccurrenceMeta,
    ClaudeProviderFact, ClaudeProviderObject, ClaudeProviderValue, CorrelationResolution,
    EventCorrelationResolver, PriorRequestAnchorQuery, event_adapter_declaration,
};
use chrono::{DateTime, Utc};
use context_graph::{
    AppendEventBatch, CreateContext, EventCoverage, EventRange, EventSourceAdapter,
    EventSourceRead, EventStreamSelector, ImportEventOptions, Store, StoreConfig,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

use super::{open_fixture_store, serialized_store_rows};

const MEDIA_BYTES: [u8; 4] = [137, 80, 78, 71];
const SESSION: &str = "timestamp-boundary-session";
const SOURCE_INSTANCE: &str = "timestamp-boundary-source";

struct MissingPriorRequest;

impl EventCorrelationResolver for MissingPriorRequest {
    fn resolve_prior_request_anchor(
        &mut self,
        _: &PriorRequestAnchorQuery,
    ) -> context_graph::Result<CorrelationResolution> {
        Ok(CorrelationResolution::Missing)
    }
}

fn event_store(name: &str) -> (tempfile::TempDir, Store, context_graph::ContextId) {
    let directory = tempdir().expect("timestamp boundary Store directory");
    let store = open_fixture_store(StoreConfig {
        db_path: directory.path().join(format!("{name}.db")),
        ..StoreConfig::default()
    });
    let context = store
        .create_context(CreateContext {
            name: format!("timestamp boundary {name}"),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create timestamp boundary Context");
    (directory, store, context.id)
}

fn stored_events(
    store: &Store,
    context_id: context_graph::ContextId,
    source_instance_ref: &str,
) -> Vec<context_graph::ConversationEvent> {
    store
        .list_stream_events(
            EventStreamSelector {
                context_id,
                connector_id: CONNECTOR_ID.into(),
                source_instance_ref: source_instance_ref.into(),
            },
            EventRange::Count {
                after_seq: None,
                limit: NonZeroU32::new(8).expect("nonzero timestamp boundary event limit"),
            },
        )
        .expect("read timestamp boundary events")
}

fn raw_media_forms(bytes: &[u8]) -> [String; 3] {
    let numeric = serde_json::to_string(bytes).expect("serialize numeric media form");
    let base64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    let hex = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    [numeric, base64, hex]
}

fn assert_raw_media_absent(value: &impl serde::Serialize, location: &str) {
    let serialized = serde_json::to_string(value).expect("serialize retained event material");
    for raw in raw_media_forms(&MEDIA_BYTES) {
        assert!(
            !serialized.contains(&raw),
            "{location} must exclude raw media representation {raw}"
        );
    }
}

fn assert_refused_media_descriptor(value: &Value) {
    assert_eq!(value["type"], json!("image"));
    assert_eq!(
        value["bytes_refused"]["decoded_length"],
        json!(MEDIA_BYTES.len())
    );
    assert_eq!(
        value["bytes_refused"]["sha256"],
        json!(format!("{:x}", Sha256::digest(MEDIA_BYTES))),
        "the retained descriptor has the hash of the refused bytes"
    );
}

#[test]
fn typed_text_attachment_refuses_timestamp_media_without_leaking_it_to_store_or_checkpoint() {
    let descriptor = ClaudeProviderObject::new(vec![
        ClaudeProviderFact::new("type", "text_attachment"),
        ClaudeProviderFact::new(
            "timestamp",
            ClaudeProviderValue::object(vec![
                ClaudeProviderFact::new("type", "image"),
                ClaudeProviderFact::new(
                    "bytes",
                    ClaudeProviderValue::array(
                        MEDIA_BYTES
                            .into_iter()
                            .map(|byte| ClaudeProviderValue::from(u64::from(byte)))
                            .collect(),
                    ),
                ),
            ]),
        ),
    ]);
    let occurrence = ClaudeOccurrence::message(
        ClaudeOccurrenceMeta::new(
            SESSION,
            Some("90000000-0000-4000-8000-000000000021".into()),
            None,
            None,
        ),
        ClaudeMessage::new(
            ClaudeMessageRole::Assistant,
            Some("typed-text-attachment".into()),
            vec![ClaudeContent::TextAttachment {
                text: "safe attachment caption".into(),
                descriptor,
            }],
        ),
    );
    let mut assembler = ClaudeOccurrenceAssembler::new();
    let mut resolver = MissingPriorRequest;
    let typed_drafts = assembler
        .assemble(
            occurrence,
            ClaudeAssemblyContext {
                session_ref: SESSION.into(),
                source_instance_ref: SOURCE_INSTANCE.into(),
                record: 1,
                record_start_byte: 0,
            },
            &mut resolver,
        )
        .expect("assemble typed attachment through the shared assembler");

    assert_eq!(typed_drafts.len(), 1);
    let typed = &typed_drafts[0];
    assert_eq!(typed.text.as_deref(), Some("safe attachment caption"));
    assert_eq!(
        typed.occurred_at, None,
        "object-valued timestamp is not commit time"
    );
    assert_eq!(typed.coverage, EventCoverage::PayloadRefused);
    assert_refused_media_descriptor(&typed.payload["timestamp"]);
    assert_raw_media_absent(&typed_drafts, "typed EventDraft");

    let historical = br#"{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000021","sessionId":"timestamp-boundary-session","message":{"id":"typed-text-attachment","role":"assistant","content":[{"type":"text_attachment","text":"safe attachment caption","timestamp":{"type":"image","bytes":[137,80,78,71]}}]}}
"#;
    let mut historical_adapter =
        ClaudeCodeJsonlAdapter::new(OsString::from("timestamp-boundary-history"));
    historical_adapter
        .identify_source(
            &mut BufReader::new(Cursor::new(historical)),
            "fixture://timestamp-boundary-history",
            None,
        )
        .expect("identify historical attachment source");
    let mut historical_resolver = MissingPriorRequest;
    let historical_drafts = match historical_adapter
        .read_next_record(
            &mut BufReader::new(Cursor::new(historical)),
            &mut historical_resolver,
        )
        .expect("read historical attachment source")
    {
        EventSourceRead::Record(record) => record.drafts,
        other => panic!("expected one historical attachment record, got {other:?}"),
    };
    assert_eq!(
        historical_drafts, typed_drafts,
        "typed and historical attachment paths have identical normalized semantics"
    );

    let (_typed_directory, typed_store, typed_context) = event_store("typed-media");
    typed_store
        .append_event_batch(AppendEventBatch::live(
            typed_context,
            event_adapter_declaration(),
            SOURCE_INSTANCE,
            typed_drafts.clone(),
        ))
        .expect("append typed attachment draft to Store");
    let typed_events = stored_events(&typed_store, typed_context, SOURCE_INSTANCE);
    assert_eq!(typed_events.len(), 1);
    assert_eq!(
        typed_events[0].text.as_deref(),
        Some("safe attachment caption")
    );
    assert_eq!(typed_events[0].occurred_at, None);
    assert_eq!(typed_events[0].coverage, EventCoverage::PayloadRefused);
    assert_refused_media_descriptor(&typed_events[0].payload["timestamp"]);
    assert_raw_media_absent(&typed_events, "typed persisted event");
    assert_raw_media_absent(
        &serialized_store_rows(&typed_store, None),
        "every typed Store table",
    );

    let (_historical_directory, historical_store, historical_context) =
        event_store("historical-media");
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from("timestamp-boundary-history"));
    let receipt = historical_store
        .import_event_stream(
            &mut adapter,
            &mut Cursor::new(historical.to_vec()),
            ImportEventOptions {
                context_id: historical_context,
                source_locator_ref: "fixture://timestamp-boundary-history".into(),
                session_override: None,
                batch_limits: Default::default(),
            },
        )
        .expect("import historical attachment through Store lifecycle");
    let historical_events = stored_events(
        &historical_store,
        historical_context,
        &receipt.source.source_instance_ref,
    );
    assert_eq!(historical_events.len(), 1);
    assert_eq!(historical_events[0].occurred_at, None);
    assert_eq!(historical_events[0].coverage, EventCoverage::PayloadRefused);
    assert_eq!(
        historical_events[0].text.as_deref(),
        Some("safe attachment caption")
    );
    assert_refused_media_descriptor(&historical_events[0].payload["timestamp"]);
    assert_raw_media_absent(&historical_events, "historical persisted event");
    assert_raw_media_absent(
        &serialized_store_rows(&historical_store, Some("event_source_checkpoints")),
        "historical source checkpoint",
    );
    assert_raw_media_absent(
        &serialized_store_rows(&historical_store, None),
        "every historical Store table",
    );
}

#[test]
fn historical_numeric_envelope_timestamp_keeps_finer_item_time_and_both_originals() {
    let source = br#"{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000022","sessionId":"numeric-envelope-time-session","timestamp":42,"message":{"id":"numeric-envelope-time-message","role":"assistant","content":[{"type":"text","text":"item timestamp wins","timestamp":"2026-09-10T12:34:56.789123456Z"}]}}
"#;
    let expected_time: DateTime<Utc> = "2026-09-10T12:34:56.789Z"
        .parse()
        .expect("expected millisecond item timestamp");
    let (_directory, store, context_id) = event_store("numeric-envelope-time");
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from("numeric-envelope-time"));
    let receipt = store
        .import_event_stream(
            &mut adapter,
            &mut Cursor::new(source.to_vec()),
            ImportEventOptions {
                context_id,
                source_locator_ref: "fixture://numeric-envelope-time".into(),
                session_override: None,
                batch_limits: Default::default(),
            },
        )
        .expect("import numeric envelope timestamp source");
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].occurred_at, Some(expected_time));
    assert_eq!(
        events[0].source_remainder["timestamp"],
        json!("2026-09-10T12:34:56.789123456Z"),
        "the item source timestamp preserves its finer original string"
    );
    assert_eq!(
        events[0].source_remainder["envelope_timestamp"],
        json!(42),
        "the conflicting non-string envelope timestamp retains its original number"
    );
}
