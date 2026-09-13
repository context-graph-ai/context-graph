use std::ffi::OsString;
use std::io::{BufReader, Cursor};
use std::num::NonZeroU32;

use cg_connector_claude_code::{
    CONNECTOR_ID, ClaudeAssemblyContext, ClaudeCodeJsonlAdapter, ClaudeContent, ClaudeMessage,
    ClaudeMessageRole, ClaudeOccurrence, ClaudeOccurrenceAssembler, ClaudeOccurrenceMeta,
    ClaudeProviderFact, ClaudeProviderObject, CorrelationResolution, EventCorrelationResolver,
    PriorRequestAnchorQuery,
};
use chrono::{DateTime, Utc};
use context_graph::{
    CreateContext, EventRange, EventSourceAdapter, EventSourceRead, EventStreamSelector,
    ImportEventOptions, Store, StoreConfig,
};
use serde_json::{Value, json};
use tempfile::tempdir;

use super::{open_fixture_store, serialized_store_rows};

const PNG_BYTES: [u8; 4] = [137, 80, 78, 71];
const PNG_ARRAY: &str = "[137,80,78,71]";
const PNG_BASE64: &str = "iVBORw==";
const PNG_HEX: &str = "89504e47";
const PNG_SHA256: &str = "0f4636c78f65d3639ece5a064b5ae753e3408614a14fb18ab4d7540d2c248543";

struct MissingResolver;

impl EventCorrelationResolver for MissingResolver {
    fn resolve_prior_request_anchor(
        &mut self,
        _: &PriorRequestAnchorQuery,
    ) -> context_graph::Result<CorrelationResolution> {
        Ok(CorrelationResolution::Missing)
    }
}

fn historical_drafts(bytes: &[u8], stem: &str) -> Vec<context_graph::EventDraft> {
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from(stem));
    adapter
        .identify_source(
            &mut BufReader::new(Cursor::new(bytes)),
            "fixture://connector-fidelity-delta",
            None,
        )
        .expect("identify independently constructed historical source");
    let mut resolver = MissingResolver;
    match adapter
        .read_next_record(&mut BufReader::new(Cursor::new(bytes)), &mut resolver)
        .expect("read independently constructed historical source")
    {
        EventSourceRead::Record(record) => record.drafts,
        other => panic!("expected complete historical record, got {other:?}"),
    }
}

fn event_store(name: &str) -> (tempfile::TempDir, Store, context_graph::ContextId) {
    let directory = tempdir().expect("delta Store directory");
    let store = open_fixture_store(StoreConfig {
        db_path: directory.path().join(format!("{name}.db")),
        ..StoreConfig::default()
    });
    let context = store
        .create_context(CreateContext {
            name: format!("connector fidelity delta {name}"),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create delta Context");
    (directory, store, context.id)
}

fn import_source(
    store: &Store,
    context_id: context_graph::ContextId,
    stem: &str,
    bytes: &[u8],
) -> context_graph::ImportEventReceipt {
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from(stem));
    store
        .import_event_stream(
            &mut adapter,
            &mut Cursor::new(bytes.to_vec()),
            ImportEventOptions {
                context_id,
                source_locator_ref: format!("fixture://{stem}"),
                session_override: None,
                batch_limits: Default::default(),
            },
        )
        .expect("persist independently constructed historical source")
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
                limit: NonZeroU32::new(16).expect("nonzero delta event read limit"),
            },
        )
        .expect("read persisted delta events")
}

fn raw_png_forms(serialized: &str) -> Vec<&'static str> {
    [PNG_ARRAY, PNG_BASE64, PNG_HEX]
        .into_iter()
        .filter(|form| serialized.contains(form))
        .collect()
}

fn record_failure(failures: &mut Vec<String>, condition: bool, message: impl Into<String>) {
    if !condition {
        failures.push(message.into());
    }
}

fn finish(failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "connector normalization delta failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn top_level_image_subagent_media_is_descriptor_only_in_drafts_and_every_store_table() {
    let source = br#"{"type":"image","uuid":"91000000-0000-4000-8000-000000000001","sessionId":"delta-subagent-session","filename":"subagent.png","subagent":{"id":"a","bytes":[137,80,78,71],"base64":"iVBORw==","hex":"89504e47"}}
"#;
    let drafts = historical_drafts(source, "delta-subagent-media");
    let (_directory, store, context_id) = event_store("delta-subagent-media");
    let receipt = import_source(&store, context_id, "delta-subagent-media", source);
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);
    let all_rows = serialized_store_rows(&store, None);
    let mut failures = Vec::new();

    record_failure(
        &mut failures,
        drafts.len() == 1,
        "historical draft count must be one",
    );
    record_failure(
        &mut failures,
        events.len() == 1,
        "persisted event count must be one",
    );
    for (location, value) in [
        (
            "historical EventDraft",
            serde_json::to_value(&drafts).expect("serialize drafts"),
        ),
        (
            "persisted ConversationEvent",
            serde_json::to_value(&events).expect("serialize events"),
        ),
    ] {
        record_failure(
            &mut failures,
            raw_png_forms(&value.to_string()).is_empty(),
            format!(
                "{location} retains raw PNG forms: {:?}",
                raw_png_forms(&value.to_string())
            ),
        );
    }
    record_failure(
        &mut failures,
        raw_png_forms(&all_rows).is_empty(),
        format!(
            "whole Store scan retains raw PNG forms: {:?}",
            raw_png_forms(&all_rows)
        ),
    );
    if let Some(draft) = drafts.first() {
        record_failure(
            &mut failures,
            draft.payload["subagent"]["id"] == json!("a"),
            format!(
                "draft payload must retain subagent identity a: {}",
                draft.payload
            ),
        );
        for key in ["bytes_refused", "base64_refused", "hex_refused"] {
            record_failure(
                &mut failures,
                draft.payload["subagent"][key]["refused"] == json!(true),
                format!("draft payload subagent.{key} must be a truthful refusal descriptor"),
            );
        }
    }
    if let Some(event) = events.first() {
        record_failure(
            &mut failures,
            event.payload["subagent"]["id"] == json!("a"),
            format!(
                "persisted payload must retain subagent identity a: {}",
                event.payload
            ),
        );
    }
    finish(failures);
}

#[test]
fn item_timestamps_on_opaque_image_and_text_attachment_blocks_win_without_item_remainders() {
    struct ItemCase {
        name: &'static str,
        source: &'static [u8],
        content: ClaudeContent,
        original_timestamp: &'static str,
    }

    let cases = vec![
        ItemCase {
            name: "thinking",
            source: br#"{"type":"assistant","uuid":"91000000-0000-4000-8000-000000000010","sessionId":"delta-item-time","message":{"id":"delta-thinking","role":"assistant","content":[{"type":"thinking","thinking":"opaque thought","timestamp":"2026-09-10T12:34:56.101123456Z"}]}}
"#,
            content: ClaudeContent::Opaque(ClaudeProviderObject::new(vec![
                ClaudeProviderFact::new("type", "thinking"),
                ClaudeProviderFact::new("thinking", "opaque thought"),
                ClaudeProviderFact::new("timestamp", "2026-09-10T12:34:56.101123456Z"),
            ])),
            original_timestamp: "2026-09-10T12:34:56.101123456Z",
        },
        ItemCase {
            name: "image",
            source: br#"{"type":"assistant","uuid":"91000000-0000-4000-8000-000000000011","sessionId":"delta-item-time","message":{"id":"delta-image","role":"assistant","content":[{"type":"image","filename":"delta.png","timestamp":"2026-09-10T12:34:56.202123456Z"}]}}
"#,
            content: ClaudeContent::RefusedMedia {
                descriptor: ClaudeProviderObject::new(vec![
                    ClaudeProviderFact::new("type", "image"),
                    ClaudeProviderFact::new("filename", "delta.png"),
                    ClaudeProviderFact::new("timestamp", "2026-09-10T12:34:56.202123456Z"),
                ]),
            },
            original_timestamp: "2026-09-10T12:34:56.202123456Z",
        },
        ItemCase {
            name: "text_attachment",
            source: br#"{"type":"assistant","uuid":"91000000-0000-4000-8000-000000000012","sessionId":"delta-item-time","message":{"id":"delta-attachment","role":"assistant","content":[{"type":"text_attachment","text":"retained attachment","filename":"delta.txt","timestamp":"2026-09-10T12:34:56.303123456Z"}]}}
"#,
            content: ClaudeContent::TextAttachment {
                text: "retained attachment".into(),
                descriptor: ClaudeProviderObject::new(vec![
                    ClaudeProviderFact::new("type", "text_attachment"),
                    ClaudeProviderFact::new("filename", "delta.txt"),
                    ClaudeProviderFact::new("timestamp", "2026-09-10T12:34:56.303123456Z"),
                ]),
            },
            original_timestamp: "2026-09-10T12:34:56.303123456Z",
        },
    ];
    let mut failures = Vec::new();
    for case in cases {
        let source_time: DateTime<Utc> = case
            .original_timestamp
            .parse()
            .expect("parse independent item timestamp");
        let expected = DateTime::<Utc>::from_timestamp_millis(source_time.timestamp_millis())
            .expect("independent item timestamp remains representable at milliseconds");
        let historical = historical_drafts(case.source, case.name);
        let typed = ClaudeOccurrence::message(
            ClaudeOccurrenceMeta::new(
                "delta-item-time",
                Some(format!("typed-{}", case.name)),
                None,
                None,
            ),
            ClaudeMessage::new(
                ClaudeMessageRole::Assistant,
                Some(case.name.into()),
                vec![case.content],
            )
            .with_provider_metadata(
                ClaudeProviderObject::default(),
                vec![ClaudeProviderObject::default()],
            )
            .expect("one explicit empty remainder for one typed item"),
        );
        let mut assembler = ClaudeOccurrenceAssembler::new();
        let mut resolver = MissingResolver;
        let typed = assembler
            .assemble(
                typed,
                ClaudeAssemblyContext {
                    session_ref: "delta-item-time".into(),
                    source_instance_ref: format!("typed-{}", case.name),
                    record: 1,
                    record_start_byte: 0,
                },
                &mut resolver,
            )
            .expect("assemble typed item without JSON construction");
        for (path, drafts) in [("historical", historical), ("typed", typed)] {
            let Some(draft) = drafts.first() else {
                failures.push(format!("{} {path} case produced no draft", case.name));
                continue;
            };
            record_failure(
                &mut failures,
                drafts.len() == 1,
                format!("{} {path} case must produce one draft", case.name),
            );
            record_failure(
                &mut failures,
                draft.occurred_at == Some(expected),
                format!(
                    "{} {path} item timestamp must supply occurred_at at exact milliseconds",
                    case.name
                ),
            );
            record_failure(
                &mut failures,
                draft.source_remainder["timestamp"] == json!(case.original_timestamp),
                format!(
                    "{} {path} source remainder must retain the original item timestamp",
                    case.name
                ),
            );
        }
    }
    finish(failures);
}

#[test]
fn invalid_envelope_timestamp_and_valid_item_timestamp_keep_both_original_values() {
    let source = br#"{"type":"assistant","uuid":"91000000-0000-4000-8000-000000000020","sessionId":"delta-timestamp-collision","timestamp":"bad-envelope","message":{"id":"delta-collision","role":"assistant","content":[{"type":"text","text":"first item time wins","timestamp":"2026-09-10T12:34:56.789123456Z"}]}}
"#;
    let legacy_source = br#"{"type":"assistant","uuid":"91000000-0000-4000-8000-000000000021","sessionId":"delta-timestamp-collision","timestamp":"bad-envelope","message":{"id":"delta-legacy","role":"assistant","content":[{"type":"text","text":"invalid envelope stays flat when no item time exists"}]}}
"#;
    let expected: DateTime<Utc> = "2026-09-10T12:34:56.789Z"
        .parse()
        .expect("expected millisecond item timestamp");
    let drafts = historical_drafts(source, "delta-timestamp-collision");
    let (_directory, store, context_id) = event_store("delta-timestamp-collision");
    let receipt = import_source(&store, context_id, "delta-timestamp-collision", source);
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);
    let legacy_drafts = historical_drafts(legacy_source, "delta-timestamp-legacy");
    let legacy_receipt = import_source(&store, context_id, "delta-timestamp-legacy", legacy_source);
    let legacy_events = stored_events(
        &store,
        context_id,
        &legacy_receipt.source.source_instance_ref,
    );
    let mut failures = Vec::new();
    for (path, occurred_at, remainder) in [
        (
            "historical draft",
            drafts.first().and_then(|draft| draft.occurred_at),
            drafts.first().map(|draft| &draft.source_remainder),
        ),
        (
            "persisted event",
            events.first().and_then(|event| event.occurred_at),
            events.first().map(|event| &event.source_remainder),
        ),
    ] {
        record_failure(
            &mut failures,
            occurred_at == Some(expected),
            format!("{path} must use the valid item timestamp for occurred_at"),
        );
        let Some(remainder) = remainder else {
            failures.push(format!("{path} must retain source remainder"));
            continue;
        };
        record_failure(
            &mut failures,
            remainder["timestamp"] == json!("2026-09-10T12:34:56.789123456Z"),
            format!("{path} source_remainder.timestamp must retain the item original"),
        );
        record_failure(
            &mut failures,
            remainder["envelope_timestamp"] == json!("bad-envelope"),
            format!("{path} source_remainder.envelope_timestamp must retain bad-envelope"),
        );
    }
    let mut assert_legacy_envelope =
        |path: &str, occurred_at: Option<DateTime<Utc>>, remainder: Option<&Value>| {
            let Some(remainder) = remainder else {
                failures.push(format!("{path} must retain the no-item-timestamp event"));
                return;
            };
            record_failure(
                &mut failures,
                occurred_at.is_none(),
                format!("{path} must not promote invalid envelope time to occurred_at"),
            );
            record_failure(
                &mut failures,
                remainder["timestamp"] == json!("bad-envelope"),
                format!("{path} must retain invalid envelope time at legacy flat timestamp"),
            );
            record_failure(
                &mut failures,
                remainder["envelope_timestamp"].is_null(),
                format!("{path} must reserve envelope_timestamp for item-time collisions"),
            );
        };
    assert_legacy_envelope(
        "historical draft",
        legacy_drafts.first().and_then(|draft| draft.occurred_at),
        legacy_drafts.first().map(|draft| &draft.source_remainder),
    );
    assert_legacy_envelope(
        "persisted event",
        legacy_events.first().and_then(|event| event.occurred_at),
        legacy_events.first().map(|event| &event.source_remainder),
    );
    finish(failures);
}

#[test]
fn base64_and_hex_media_descriptors_hash_the_decoded_png_bytes_independently() {
    let source = br#"{"type":"assistant","uuid":"91000000-0000-4000-8000-000000000030","sessionId":"delta-encoded-media","message":{"id":"delta-encoded-media","role":"assistant","content":[{"type":"image","filename":"base64.png","base64":"iVBORw=="},{"type":"image","filename":"hex.png","hex":"89504e47"}]}}
"#;
    let drafts = historical_drafts(source, "delta-encoded-media");
    let (_directory, store, context_id) = event_store("delta-encoded-media");
    let receipt = import_source(&store, context_id, "delta-encoded-media", source);
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);
    let mut failures = Vec::new();
    record_failure(
        &mut failures,
        PNG_BYTES.len() == 4,
        "independent PNG byte oracle must be four bytes",
    );
    for (path, values) in [
        (
            "historical draft",
            drafts
                .iter()
                .map(|draft| &draft.source_remainder)
                .collect::<Vec<_>>(),
        ),
        (
            "persisted event",
            events
                .iter()
                .map(|event| &event.source_remainder)
                .collect::<Vec<_>>(),
        ),
    ] {
        record_failure(
            &mut failures,
            values.len() == 2,
            format!("{path} must retain two independently encoded media items"),
        );
        for (index, key) in ["base64_refused", "hex_refused"].into_iter().enumerate() {
            let Some(remainder) = values.get(index) else {
                continue;
            };
            record_failure(
                &mut failures,
                remainder[key]["decoded_length"] == json!(4),
                format!("{path} {key} must record decoded length four"),
            );
            record_failure(
                &mut failures,
                remainder[key]["sha256"] == json!(PNG_SHA256),
                format!("{path} {key} must hash decoded PNG bytes independently"),
            );
        }
        let serialized = serde_json::to_string(&values).expect("serialize encoded media values");
        record_failure(
            &mut failures,
            raw_png_forms(&serialized).is_empty(),
            format!(
                "{path} retains raw encoded PNG form(s): {:?}",
                raw_png_forms(&serialized)
            ),
        );
    }
    finish(failures);
}
