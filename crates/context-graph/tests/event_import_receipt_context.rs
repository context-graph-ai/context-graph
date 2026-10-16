//! regression contract for source context on import refusals.
//!
//! These tests cover source context on refusals independently of coverage receipt fields.

use std::io::{BufRead, Cursor};
use std::num::NonZeroUsize;

use chrono::{TimeZone, Utc};
use context_graph::{
    AdaptedSourceRecord, CgError, CreateContext, EndOfSource, EventAdapterDeclaration,
    EventBatchLimits, EventCoverage, EventRenderRole, EventSourceAdapter, EventSourceIdentity,
    EventSourceRead, ImportEventOptions, SourceRecordPosition, Store, StoreConfig,
};
use serde_json::{Value, json};
use tempfile::TempDir;

const CONNECTOR: &str = "receipt-context-source";
const SESSION: &str = "known-session";
const SOURCE_INSTANCE: &str = "known-source-instance";

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: json!({
            "type": "object",
            "properties": {"declared": {"type": "string"}},
            "additionalProperties": false,
        }),
        source_position_order: vec!["/line".into()],
    }
}

fn draft(key: &str) -> context_graph::EventDraft {
    context_graph::EventDraft {
        session_ref: SESSION.into(),
        source_event_key: key.into(),
        source_position: json!({"line": 7, "record": key}),
        occurred_at: Some(
            Utc.timestamp_opt(1_700_000_000, 0)
                .single()
                .expect("timestamp"),
        ),
        author: "fixture-author".into(),
        kind: "fixture-kind".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("fixture text".into()),
        payload: json!({"fixture": key}),
        anchors: json!({"declared": "present"}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

struct OneRecordImportFixture {
    declaration: EventAdapterDeclaration,
    record: Option<AdaptedSourceRecord>,
}

impl OneRecordImportFixture {
    fn new(draft: context_graph::EventDraft, byte_len: u64) -> Self {
        Self {
            declaration: declaration(),
            record: Some(AdaptedSourceRecord {
                drafts: vec![draft],
                checkpoint_after_record: json!({"record": 7}),
                source_position_after_record: SourceRecordPosition {
                    byte_offset: byte_len,
                    coordinate: json!({"line": 7}),
                },
            }),
        }
    }
}

impl EventSourceAdapter for OneRecordImportFixture {
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
            .map_err(|error| CgError::Engine(error.to_string()))?;
        Ok(EventSourceIdentity {
            source_instance_ref: SOURCE_INSTANCE.into(),
            session_ref: SESSION.into(),
            provisional_empty: false,
        })
    }

    fn resume_source(
        &mut self,
        _identity: &EventSourceIdentity,
        _checkpoint: &Value,
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
            .map_err(|error| CgError::Engine(error.to_string()))?;
        if line.is_empty() {
            return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
        }
        Ok(EventSourceRead::Record(
            self.record.take().expect("fixture has one record"),
        ))
    }
}

fn import_refusal(draft: context_graph::EventDraft) -> CgError {
    let root = TempDir::new().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("receipt-context.db"),
        ..StoreConfig::default()
    })
    .expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "receipt context".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    let bytes = b"fixture-record\n";
    let mut source = OneRecordImportFixture::new(draft, bytes.len() as u64);

    store
        .import_event_stream(
            &mut source,
            &mut Cursor::new(bytes.to_vec()),
            ImportEventOptions {
                context_id,
                source_locator_ref: "fixture://receipt-context".into(),
                session_override: None,
                batch_limits: EventBatchLimits {
                    max_events: NonZeroUsize::new(1).expect("nonzero"),
                    max_normalized_bytes: NonZeroUsize::new(1024).expect("nonzero"),
                },
            },
        )
        .expect_err("malformed fixture is refused")
}

fn assert_known_source_context(error: CgError, expected_position: Value) {
    let CgError::SourceRecordMalformed {
        connector_id,
        session_ref,
        position,
        ..
    } = error
    else {
        panic!("expected typed source-record refusal, got {error:?}");
    };
    assert_eq!(connector_id, CONNECTOR);
    assert_eq!(session_ref, SESSION);
    assert_eq!(position, expected_position);
}

#[test]
fn badrequiredfield_import_refusal_retains_known_connector_session_and_position() {
    let mut malformed = draft("badrequiredfield");
    malformed.author.clear();

    assert_known_source_context(
        import_refusal(malformed),
        json!({"line": 7, "record": "badrequiredfield"}),
    );
}

#[test]
fn badanchor_import_refusal_retains_known_connector_session_and_position() {
    let mut malformed = draft("badanchor");
    malformed.anchors = json!({"undeclared": "value"});

    assert_known_source_context(
        import_refusal(malformed),
        json!({"line": 7, "record": "badanchor"}),
    );
}
