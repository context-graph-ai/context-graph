use std::ffi::OsString;
use std::io::Cursor;
use std::num::NonZeroU32;

use cg_connector_claude_code::ClaudeCodeJsonlAdapter;
use context_graph::{
    CreateContext, EventBatchLimits, EventCoverage, EventRange, EventStreamSelector,
    ImportEventOptions, StoreConfig,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

use super::{
    assemble_records, assemble_records_with_context, assert_facts, assert_json_field_role,
    assert_json_retained, expected_facts, fixture_bytes, fixture_records, open_fixture_store,
    serialized_store_rows,
};

const VALID_MEDIA_SESSION: &str = "synthetic-session-valid-media";
const VALID_MEDIA_LOCATOR: &str = "fixture://valid-base64-media";
const FORBIDDEN_BYTES: &[u8] = b"FORBIDDEN_VALID_MEDIA_BYTES_7c92";
const FORBIDDEN_BASE64: &str = "Rk9SQklEREVOX1ZBTElEX01FRElBX0JZVEVTXzdjOTI=";
const FORBIDDEN_NESTED_BYTES: &[u8] = b"FORBIDDEN_NESTED_MEDIA_BYTES_4e81";
const FORBIDDEN_NESTED_BASE64: &str = "Rk9SQklEREVOX05FU1RFRF9NRURJQV9CWVRFU180ZTgx";
const FORBIDDEN_DATA_URL_BYTES: &[u8] = b"FORBIDDEN_DATA_URL_BYTES_9a27";
const FORBIDDEN_DATA_URL_BASE64: &str = "Rk9SQklEREVOX0RBVEFfVVJMX0JZVEVTXzlhMjc=";
const FORBIDDEN_BLOB_URL_BYTES: &[u8] = b"FORBIDDEN_BLOB_URL_MARKER_b315";
const FORBIDDEN_BLOB_URL_BASE64: &str = "Rk9SQklEREVOX0JMT0JfVVJMX01BUktFUl9iMzE1";

fn lower_hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("write hex to String");
    }
    encoded
}

fn assert_forbidden_forms_absent(serialized: &str) {
    for (bytes, base64) in [
        (FORBIDDEN_BYTES, FORBIDDEN_BASE64),
        (FORBIDDEN_NESTED_BYTES, FORBIDDEN_NESTED_BASE64),
        (FORBIDDEN_DATA_URL_BYTES, FORBIDDEN_DATA_URL_BASE64),
        (FORBIDDEN_BLOB_URL_BYTES, FORBIDDEN_BLOB_URL_BASE64),
    ] {
        let forbidden_text = std::str::from_utf8(bytes).expect("ASCII media marker");
        let forbidden_hex = lower_hex(bytes);
        for forbidden in [forbidden_text, base64, forbidden_hex.as_str()] {
            assert!(
                !serialized.contains(forbidden),
                "refused media cannot survive as raw, base64, or hex source content: {forbidden}"
            );
        }
    }
}

#[test]
fn opaque_and_media_facts_remain_ordered_and_refused_media_never_survives() {
    let records = [6, 7, 8, 9];
    let drafts = assemble_records("compact-rich.jsonl", &records);
    let facts = expected_facts("compact-rich.jsonl", &records);
    assert_facts("compact-rich.jsonl", &drafts, &facts);
    assert_eq!(drafts[0].coverage, EventCoverage::Complete);
    assert_eq!(
        drafts[1].coverage,
        EventCoverage::PayloadRefused,
        "record 6 item 1 is the refused image; item 0 is a complete text attachment"
    );
    for (draft_index, record) in [(5, 8), (6, 9)] {
        let raw_object: Value = serde_json::from_slice(
            &fixture_records("compact-rich.jsonl")
                .into_iter()
                .find_map(|(ordinal, _, raw)| (ordinal == record).then_some(raw))
                .expect("opaque oracle record"),
        )
        .expect("opaque oracle object");
        assert_json_retained(
            &drafts[draft_index].source_remainder,
            &raw_object,
            "complete opaque top-level source object",
        );
    }

    let oracle_serialized = serde_json::to_string(&drafts).expect("serialize drafts for leak scan");
    assert!(
        !oracle_serialized.contains("FORBIDDEN_MEDIA_MARKER_6f31"),
        "the immutable oracle's invalid base64 media marker cannot reach an EventDraft"
    );

    let valid_drafts = assemble_records_with_context(
        "refused-media-valid-base64.jsonl",
        &[1],
        VALID_MEDIA_SESSION,
        "valid-media-source",
    );
    assert_eq!(valid_drafts.len(), 1);
    assert_eq!(valid_drafts[0].coverage, EventCoverage::PayloadRefused);
    let expected_hash = format!("{:x}", Sha256::digest(FORBIDDEN_BYTES));
    let retained = json!({
        "payload": &valid_drafts[0].payload,
        "source_remainder": &valid_drafts[0].source_remainder,
    });
    assert_json_field_role(
        &retained,
        &[&["media_type"], &["source", "media_type"]],
        &json!("image/png"),
        "media type descriptor",
    );
    assert_json_field_role(
        &retained,
        &[&["source", "type"]],
        &json!("base64"),
        "media source descriptor",
    );
    assert_json_retained(
        &retained,
        &Value::from(u64::try_from(FORBIDDEN_BYTES.len()).expect("marker length fits u64")),
        "decoded media byte length",
    );
    assert_json_retained(&retained, &json!(expected_hash), "decoded media SHA-256");
    assert_forbidden_forms_absent(
        &serde_json::to_string(&valid_drafts).expect("serialize valid-base64 drafts"),
    );

    let nested_drafts = assemble_records_with_context(
        "refused-media-nested-and-urls.jsonl",
        &[1, 2, 3],
        "synthetic-session-nested-media",
        "nested-media-source",
    );
    assert_eq!(nested_drafts.len(), 3);
    assert!(
        nested_drafts
            .iter()
            .all(|draft| draft.coverage == EventCoverage::PayloadRefused),
        "nested opaque media plus media data: and blob: URLs are metadata-only refusals"
    );
    for (draft, media_type) in nested_drafts
        .iter()
        .zip(["image/png", "audio/wav", "video/mp4"])
    {
        let retained = json!({
            "payload": &draft.payload,
            "source_remainder": &draft.source_remainder,
        });
        assert_json_field_role(
            &retained,
            &[&["media_type"], &["source", "media_type"]],
            &json!(media_type),
            "sanitized nested media descriptor",
        );
    }
    assert_forbidden_forms_absent(
        &serde_json::to_string(&nested_drafts)
            .expect("serialize nested and URL media drafts for leak scan"),
    );

    let directory = tempdir().expect("media Store directory");
    let store = open_fixture_store(StoreConfig {
        db_path: directory.path().join("valid-media.db"),
        ..StoreConfig::default()
    });
    let context = store
        .create_context(CreateContext {
            name: "valid media privacy".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create media Context");
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from("valid-media"));
    let receipt = store
        .import_event_stream(
            &mut adapter,
            &mut Cursor::new(fixture_bytes("refused-media-valid-base64.jsonl").to_vec()),
            ImportEventOptions {
                context_id: context.id,
                source_locator_ref: VALID_MEDIA_LOCATOR.into(),
                session_override: None,
                batch_limits: EventBatchLimits::default(),
            },
        )
        .expect("import valid-base64 media through core lifecycle");
    assert_eq!(receipt.inserted, 1);
    assert_eq!(receipt.pending_tail_records, 0);

    let events = store
        .list_stream_events(
            EventStreamSelector {
                context_id: context.id,
                connector_id: cg_connector_claude_code::CONNECTOR_ID.into(),
                source_instance_ref: receipt.source.source_instance_ref,
            },
            EventRange::Count {
                after_seq: None,
                limit: NonZeroU32::new(4).expect("nonzero event limit"),
            },
        )
        .expect("read imported media event");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].coverage, EventCoverage::PayloadRefused);
    let stored_retained = json!({
        "payload": &events[0].payload,
        "source_remainder": &events[0].source_remainder,
    });
    assert_json_retained(
        &stored_retained,
        &json!(u64::try_from(FORBIDDEN_BYTES.len()).expect("marker length fits u64")),
        "stored decoded media byte length",
    );
    assert_json_retained(
        &stored_retained,
        &json!(format!("{:x}", Sha256::digest(FORBIDDEN_BYTES))),
        "stored decoded media SHA-256",
    );

    let mut nested_adapter = ClaudeCodeJsonlAdapter::new(OsString::from("nested-and-url-media"));
    let nested_receipt = store
        .import_event_stream(
            &mut nested_adapter,
            &mut Cursor::new(fixture_bytes("refused-media-nested-and-urls.jsonl").to_vec()),
            ImportEventOptions {
                context_id: context.id,
                source_locator_ref: "fixture://nested-and-url-media".into(),
                session_override: None,
                batch_limits: EventBatchLimits::default(),
            },
        )
        .expect("import nested opaque and URL media through core lifecycle");
    assert_eq!(nested_receipt.inserted, 3);
    assert_eq!(nested_receipt.pending_tail_records, 0);

    let checkpoint_rows = serialized_store_rows(&store, Some("event_source_checkpoints"));
    assert!(
        !checkpoint_rows.is_empty(),
        "the imported record has a durable source checkpoint to scan"
    );
    assert_forbidden_forms_absent(&checkpoint_rows);
    assert_forbidden_forms_absent(&serialized_store_rows(&store, None));
}
