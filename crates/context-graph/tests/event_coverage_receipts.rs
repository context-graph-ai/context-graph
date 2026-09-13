//! Public receipt contract for accepted event coverage observations.

use std::collections::VecDeque;
use std::io::{BufRead, Cursor};
use std::num::NonZeroUsize;

use chrono::{TimeZone, Utc};
use context_graph::{
    AdaptedSourceRecord, AppendEventBatch, CreateContext, EndOfSource, EventAdapterDeclaration,
    EventBatchLimits, EventCoverage, EventRenderRole, EventSourceAdapter, EventSourceIdentity,
    EventSourceRead, ImportEventOptions, SourceRecordPosition, Store, StoreConfig,
};
use serde::Serialize;
use serde_json::{Value, json};
use tempfile::TempDir;

const CONNECTOR: &str = "coverage-receipt-source";
const SOURCE_INSTANCE: &str = "coverage-receipt-instance";
const SESSION: &str = "coverage-receipt-session";

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/line".into()],
    }
}

fn mixed_drafts() -> Vec<context_graph::EventDraft> {
    [
        EventCoverage::Complete,
        EventCoverage::Opaque,
        EventCoverage::PayloadRefused,
        EventCoverage::SourceDeclaredMissing,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, coverage)| context_graph::EventDraft {
        session_ref: SESSION.into(),
        source_event_key: format!("coverage-{index}"),
        source_position: json!({"line": index + 1}),
        occurred_at: Some(
            Utc.timestamp_opt(1_700_000_000 + index as i64, 0)
                .single()
                .unwrap(),
        ),
        author: "fixture-author".into(),
        kind: "fixture-kind".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("fixture event {index}")),
        payload: json!({"fixture": index}),
        anchors: json!({"fixture": "coverage"}),
        source_remainder: json!({}),
        coverage,
    })
    .collect()
}

fn store_and_context() -> (TempDir, Store, context_graph::ContextId) {
    let root = TempDir::new().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("coverage-receipts.db"),
        ..StoreConfig::default()
    })
    .expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "coverage receipts".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    (root, store, context_id)
}

fn serialized_coverage(receipt: &impl Serialize) -> Option<Value> {
    serde_json::to_value(receipt)
        .expect("serialize actual Store receipt")
        .get("coverage")
        .cloned()
}

fn mixed_coverage() -> Value {
    json!({
        "complete": 1,
        "opaque": 1,
        "payload_refused": 1,
        "source_declared_missing": 1,
    })
}

fn no_coverage_observations() -> Value {
    json!({
        "complete": 0,
        "opaque": 0,
        "payload_refused": 0,
        "source_declared_missing": 0,
    })
}

struct ScriptedSource {
    declaration: EventAdapterDeclaration,
    records: VecDeque<AdaptedSourceRecord>,
}

impl ScriptedSource {
    fn mixed() -> Self {
        let mut offset = 0_u64;
        let records = mixed_drafts()
            .into_iter()
            .enumerate()
            .map(|(index, draft)| {
                offset += 2;
                AdaptedSourceRecord {
                    drafts: vec![draft],
                    checkpoint_after_record: json!({"record": index + 1}),
                    source_position_after_record: SourceRecordPosition {
                        byte_offset: offset,
                        coordinate: json!({"line": index + 1}),
                    },
                }
            })
            .collect();
        Self {
            declaration: declaration(),
            records,
        }
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
        let mut prefix = String::new();
        reader
            .read_line(&mut prefix)
            .map_err(|error| context_graph::CgError::Engine(error.to_string()))?;
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
            .map_err(|error| context_graph::CgError::Engine(error.to_string()))?;
        if line.is_empty() {
            return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
        }
        Ok(EventSourceRead::Record(
            self.records
                .pop_front()
                .expect("fixture record for source line"),
        ))
    }
}

fn import_options(context_id: context_graph::ContextId) -> ImportEventOptions {
    ImportEventOptions {
        context_id,
        source_locator_ref: "fixture://coverage-receipts".into(),
        session_override: None,
        batch_limits: EventBatchLimits {
            max_events: NonZeroUsize::new(1).expect("nonzero"),
            max_normalized_bytes: NonZeroUsize::new(1024).expect("nonzero"),
        },
    }
}

#[test]
fn append_receipt_counts_mixed_coverage_for_insertions_and_already_present_observations() {
    let (_root, store, context_id) = store_and_context();
    let input = AppendEventBatch::live(context_id, declaration(), SOURCE_INSTANCE, mixed_drafts());

    let inserted = store
        .append_event_batch(input.clone())
        .expect("append events");
    assert_eq!((inserted.inserted, inserted.already_present), (4, 0));

    let already_present = store.append_event_batch(input).expect("replay events");
    assert_eq!(
        (already_present.inserted, already_present.already_present),
        (0, 4),
        "persistence outcomes remain independent from coverage observations"
    );
    assert_eq!(
        [
            serialized_coverage(&inserted),
            serialized_coverage(&already_present)
        ],
        [Some(mixed_coverage()), Some(mixed_coverage())],
        "the public serialized receipt counts each accepted coverage state for insertions and already-present observations"
    );
}

#[test]
fn multibatch_import_receipt_counts_mixed_coverage_and_empty_checkpoint_observation() {
    let (_root, store, context_id) = store_and_context();
    let bytes = b"a\nb\nc\nd\n";

    let imported = store
        .import_event_stream(
            &mut ScriptedSource::mixed(),
            &mut Cursor::new(bytes.to_vec()),
            import_options(context_id),
        )
        .expect("import mixed records in multiple batches");
    assert_eq!(
        (
            imported.batch_count,
            imported.inserted,
            imported.already_present
        ),
        (4, 4, 0)
    );

    let unchanged = store
        .import_event_stream(
            &mut ScriptedSource::mixed(),
            &mut Cursor::new(bytes.to_vec()),
            import_options(context_id),
        )
        .expect("recognize unchanged complete checkpoint");
    assert_eq!(
        (
            unchanged.batch_count,
            unchanged.inserted,
            unchanged.already_present
        ),
        (0, 0, 0)
    );
    assert_eq!(
        [
            serialized_coverage(&imported),
            serialized_coverage(&unchanged)
        ],
        [Some(mixed_coverage()), Some(no_coverage_observations())],
        "the public serialized import receipt counts observed coverage and reports zero observations for an unchanged checkpoint"
    );
}
