use std::ffi::OsString;
use std::io::{BufReader, Cursor};
use std::num::NonZeroU32;

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
use tempfile::tempdir;

use super::open_fixture_store;

const PNG_SIGNATURE: [u8; 4] = [137, 80, 78, 71];
const PNG_SIGNATURE_ARRAY: &str = "[137,80,78,71]";
const PNG_SIGNATURE_BASE64: &str = "iVBORw==";
const PNG_SIGNATURE_HEX: &str = "89504e47";
const PNG_SIGNATURE_SHA256: &str =
    "0f4636c78f65d3639ece5a064b5ae753e3408614a14fb18ab4d7540d2c248543";

struct MissingPriorRequest;

impl EventCorrelationResolver for MissingPriorRequest {
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
            "fixture://independent-capture-regression",
            None,
        )
        .expect("identify independent historical source");
    let mut resolver = MissingPriorRequest;
    match adapter
        .read_next_record(&mut BufReader::new(Cursor::new(bytes)), &mut resolver)
        .expect("read independent historical source record")
    {
        EventSourceRead::Record(record) => record.drafts,
        other => panic!("expected a complete historical record, got {other:?}"),
    }
}

fn import_source(
    store: &Store,
    context_id: context_graph::ContextId,
    stem: &str,
    bytes: &[u8],
    session_override: Option<&str>,
) -> context_graph::Result<context_graph::ImportEventReceipt> {
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from(stem));
    store.import_event_stream(
        &mut adapter,
        &mut Cursor::new(bytes.to_vec()),
        ImportEventOptions {
            context_id,
            source_locator_ref: format!("fixture://{stem}"),
            session_override: session_override.map(str::to_owned),
            batch_limits: Default::default(),
        },
    )
}

fn event_store(name: &str) -> (tempfile::TempDir, Store, context_graph::ContextId) {
    let directory = tempdir().expect("independent capture Store directory");
    let store = open_fixture_store(StoreConfig {
        db_path: directory.path().join(format!("{name}.db")),
        ..StoreConfig::default()
    });
    let context = store
        .create_context(CreateContext {
            name: format!("independent {name} regression"),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create independent capture Context");
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
                limit: NonZeroU32::new(16).expect("nonzero independent event read limit"),
            },
        )
        .expect("read independently imported source events")
}

fn contains_raw_png_encoding(serialized: &str) -> bool {
    [PNG_SIGNATURE_ARRAY, PNG_SIGNATURE_BASE64, PNG_SIGNATURE_HEX]
        .into_iter()
        .any(|encoding| serialized.contains(encoding))
}

fn assert_no_raw_png_encoding(value: &Value, location: &str) {
    let serialized =
        serde_json::to_string(value).expect("serialize event material for raw-media scan");
    assert!(
        !contains_raw_png_encoding(&serialized),
        "{location} must not retain numeric-array, base64, or hexadecimal PNG source encodings"
    );
}

fn assert_refused_media_capture(
    source: &[u8],
    stem: &str,
    expected_drafts: usize,
    refused_index: usize,
    expected_filename: &str,
    expected_reference: &str,
    expect_nested_descriptor: bool,
) {
    assert!(
        contains_raw_png_encoding(PNG_SIGNATURE_ARRAY),
        "the leak scanner detects the refused numeric byte array"
    );
    assert!(
        contains_raw_png_encoding(PNG_SIGNATURE_BASE64),
        "the leak scanner detects the refused base64 encoding"
    );
    assert!(
        contains_raw_png_encoding(PNG_SIGNATURE_HEX),
        "the leak scanner detects the refused hexadecimal encoding"
    );

    let drafts = historical_drafts(source, stem);
    assert_eq!(drafts.len(), expected_drafts);
    let draft = &drafts[refused_index];
    assert_eq!(draft.coverage, EventCoverage::PayloadRefused);
    assert_eq!(draft.source_remainder["type"], json!("image"));
    assert_eq!(draft.source_remainder["filename"], json!(expected_filename));
    assert_eq!(
        draft.source_remainder["reference"],
        json!(expected_reference)
    );
    assert_eq!(
        draft.source_remainder["bytes_refused"]["decoded_length"],
        json!(4)
    );
    assert_eq!(
        draft.source_remainder["bytes_refused"]["sha256"],
        json!(PNG_SIGNATURE_SHA256)
    );
    if expect_nested_descriptor {
        assert_eq!(
            draft.source_remainder["nested"]["binary_refused"]["decoded_length"],
            json!(PNG_SIGNATURE.len())
        );
        assert_eq!(
            draft.source_remainder["nested"]["binary_refused"]["sha256"],
            json!(PNG_SIGNATURE_SHA256)
        );
        assert_eq!(
            draft.source_remainder["nested"]["label"],
            json!("thumbnail")
        );
    }
    assert_ne!(
        draft.source_remainder["bytes_refused"]["sha256"],
        json!(PNG_SIGNATURE_HEX),
        "the retained SHA-256 descriptor is allowed; a raw hexadecimal source encoding is not"
    );
    assert_no_raw_png_encoding(
        &serde_json::to_value(&drafts).expect("serialize historical drafts"),
        "all historical EventDrafts, including complete text",
    );

    let (_directory, store, context_id) = event_store(stem);
    let receipt =
        import_source(&store, context_id, stem, source, None).expect("import opaque media");
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);
    assert_eq!(events.len(), expected_drafts);
    assert_eq!(
        events[refused_index].coverage,
        EventCoverage::PayloadRefused
    );
    assert_eq!(
        events[refused_index].source_remainder["filename"],
        json!(expected_filename)
    );
    assert_eq!(
        events[refused_index].source_remainder["reference"],
        json!(expected_reference)
    );
    assert_eq!(
        events[refused_index].source_remainder["bytes_refused"]["sha256"],
        json!(PNG_SIGNATURE_SHA256)
    );
    assert_no_raw_png_encoding(
        &serde_json::to_value(&events).expect("serialize persisted events"),
        "persisted events, including complete text",
    );
    let all_store_rows = super::serialized_store_rows(&store, None);
    assert!(
        !all_store_rows.is_empty(),
        "the imported source creates rows and a durable checkpoint for the whole-Store scan"
    );
    assert!(
        !contains_raw_png_encoding(PNG_SIGNATURE_SHA256),
        "the scanner does not confuse the allowed descriptor hash with a raw PNG encoding"
    );
    assert!(
        !contains_raw_png_encoding(&all_store_rows),
        "every Store table, including event_source_checkpoints, excludes raw numeric-array, base64, and hexadecimal media encodings"
    );
}

#[test]
fn top_level_opaque_media_refuses_inline_bytes_without_losing_descriptor() {
    let source = br#"{"type":"image","uuid":"90000000-0000-4000-8000-000000000001","sessionId":"independent-media-session","filename":"cover.png","reference":"cas://sha256:opaque-cover","bytes":[137,80,78,71]}
"#;
    assert_refused_media_capture(
        source,
        "independent-top-level-media",
        1,
        0,
        "cover.png",
        "cas://sha256:opaque-cover",
        false,
    );
}

#[test]
fn nested_media_with_text_refuses_all_inline_encodings_without_losing_descriptor() {
    let source = br#"{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000010","sessionId":"independent-nested-media-session","message":{"id":"opaque-media-with-text","role":"assistant","content":[{"type":"text","text":"complete text must be scanned too"},{"type":"image","filename":"nested-cover.png","reference":"cas://sha256:nested-cover","bytes":[137,80,78,71],"base64":"iVBORw==","hex":"89504e47","nested":{"type":"image","media_type":"image/png","label":"thumbnail","binary":[137,80,78,71]}}]}}
"#;
    let drafts = historical_drafts(source, "independent-nested-media");
    assert_eq!(drafts.len(), 2);
    assert_eq!(drafts[0].coverage, EventCoverage::Complete);
    assert_eq!(
        drafts[0].text.as_deref(),
        Some("complete text must be scanned too")
    );
    assert_refused_media_capture(
        source,
        "independent-nested-media",
        2,
        1,
        "nested-cover.png",
        "cas://sha256:nested-cover",
        true,
    );
}

#[test]
fn first_opaque_block_keeps_message_model_and_usage_in_retained_source() {
    let source = br#"{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000002","sessionId":"independent-thinking-session","message":{"id":"thought-then-text","role":"assistant","model":"claude-independent","usage":{"input_tokens":17,"output_tokens":5},"content":[{"type":"thinking","thinking":"private but accepted provider block","signature":"sig-independent"},{"type":"text","text":"mapped text remains after thinking"}]}}
"#;
    let drafts = historical_drafts(source, "independent-thinking");
    assert_eq!(drafts.len(), 2);
    assert_eq!(drafts[0].kind, "raw:thinking");
    assert_eq!(
        drafts[0].source_remainder["message"]["model"],
        json!("claude-independent")
    );
    assert_eq!(
        drafts[0].source_remainder["message"]["usage"],
        json!({"input_tokens": 17, "output_tokens": 5})
    );
    assert_eq!(
        drafts[1].text.as_deref(),
        Some("mapped text remains after thinking")
    );

    let (_directory, store, context_id) = event_store("thinking-message-extras");
    let receipt = import_source(&store, context_id, "independent-thinking", source, None)
        .expect("import thinking followed by mapped text");
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);
    assert_eq!(events.len(), 2);
    assert_eq!(
        events[0].source_remainder["message"]["model"],
        json!("claude-independent")
    );
    assert_eq!(
        events[0].source_remainder["message"]["usage"]["output_tokens"],
        json!(5)
    );
}

#[test]
fn sole_later_embedded_session_beats_filename_but_override_and_conflicts_remain_honest() {
    let late_sole_session = br#"{"type":"metadata_note","uuid":"90000000-0000-4000-8000-000000000003","note":"valid metadata before session is known"}
{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000004","sessionId":"sole-vendor-session","message":{"id":"sole-session-message","role":"assistant","content":"the only embedded ID identifies this file"}}
"#;
    let (_directory, store, context_id) = event_store("late-sole-session");
    let receipt = import_source(
        &store,
        context_id,
        "filename-fallback-must-not-win",
        late_sole_session,
        None,
    )
    .expect("a sole later embedded session identifies a complete source");
    assert_eq!(receipt.source.session_ref, "sole-vendor-session");
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);
    assert_eq!(events.len(), 2);
    assert!(
        events
            .iter()
            .all(|event| event.session_ref == "sole-vendor-session")
    );
    assert_eq!(events[1].payload["sessionId"], json!("sole-vendor-session"));

    let (_directory, override_store, override_context) = event_store("late-sole-override");
    let override_receipt = import_source(
        &override_store,
        override_context,
        "override-wins",
        late_sole_session,
        Some("operator-selected-session"),
    )
    .expect("an explicit override remains the outward session");
    assert_eq!(
        override_receipt.source.session_ref,
        "operator-selected-session"
    );
    let override_events = stored_events(
        &override_store,
        override_context,
        &override_receipt.source.source_instance_ref,
    );
    assert_eq!(
        override_events[1].payload["sessionId"],
        json!("sole-vendor-session")
    );

    let mixed = br#"{"type":"metadata_note","uuid":"90000000-0000-4000-8000-000000000005","note":"valid leading metadata"}
{"type":"user","uuid":"90000000-0000-4000-8000-000000000006","sessionId":"vendor-a","message":{"id":"mixed-a","role":"user","content":"first vendor identity"}}
{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000007","sessionId":"vendor-b","message":{"id":"mixed-b","role":"assistant","content":"conflicting vendor identity"}}
"#;
    let (_directory, mixed_store, mixed_context) = event_store("late-mixed-session");
    assert!(
        matches!(
            import_source(
                &mixed_store,
                mixed_context,
                "mixed-vendor-session",
                mixed,
                Some("operator-selected-session"),
            ),
            Err(context_graph::CgError::SourceRecordMalformed { .. })
        ),
        "an override cannot make mixed nonempty vendor IDs safe"
    );
}

#[test]
fn item_timestamps_supply_millis_and_keep_finer_or_invalid_originals_out_of_commit_time() {
    let source = br#"{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000008","sessionId":"independent-item-time-session","message":{"id":"item-time-message","role":"assistant","content":[{"type":"text","text":"finer source time","timestamp":"2026-09-10T12:34:56.789123456Z"},{"type":"text","text":"invalid source time","timestamp":"not-a-source-timestamp"}]}}
"#;
    let expected_millis: DateTime<Utc> = "2026-09-10T12:34:56.789Z"
        .parse()
        .expect("independent expected millisecond timestamp");
    let drafts = historical_drafts(source, "independent-item-time");
    assert_eq!(drafts.len(), 2);
    assert_eq!(drafts[0].occurred_at, Some(expected_millis));
    assert_eq!(
        drafts[0].source_remainder["timestamp"],
        json!("2026-09-10T12:34:56.789123456Z")
    );
    assert_eq!(drafts[1].occurred_at, None);
    assert_eq!(
        drafts[1].source_remainder["timestamp"],
        json!("not-a-source-timestamp")
    );

    let (_directory, store, context_id) = event_store("item-time");
    let receipt = import_source(&store, context_id, "independent-item-time", source, None)
        .expect("import independent per-item source times");
    let events = stored_events(&store, context_id, &receipt.source.source_instance_ref);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].occurred_at, Some(expected_millis));
    assert_eq!(
        events[0].source_remainder["timestamp"],
        json!("2026-09-10T12:34:56.789123456Z")
    );
    assert_eq!(events[1].occurred_at, None);
    assert_eq!(
        events[1].source_remainder["timestamp"],
        json!("not-a-source-timestamp")
    );
}

#[test]
fn typed_provider_metadata_matches_historical_opaque_and_error_result_capture() {
    const SESSION: &str = "independent-typed-metadata-session";
    const SOURCE: &[u8] = br#"{"type":"assistant","uuid":"90000000-0000-4000-8000-000000000009","sessionId":"independent-typed-metadata-session","timestamp":"2026-09-10T12:34:56.123456789Z","message":{"id":"typed-metadata-message","role":"assistant","model":"claude-independent-metadata","usage":{"input_tokens":17,"output_tokens":5},"content":[{"type":"thinking","thinking":"first opaque provider message","signature":"opaque-signature"},{"type":"tool_result","tool_use_id":"typed-tool-result","content":"tool failed honestly","is_error":true}]}}
"#;
    let expected_millis: DateTime<Utc> = "2026-09-10T12:34:56.123Z"
        .parse()
        .expect("independent millisecond timestamp");

    let historical_drafts = historical_drafts(SOURCE, "independent-typed-metadata");
    assert_eq!(historical_drafts.len(), 2);
    assert_eq!(historical_drafts[0].kind, "raw:thinking");
    assert_eq!(historical_drafts[0].occurred_at, Some(expected_millis));
    assert_eq!(
        historical_drafts[0].source_remainder["message"]["model"].as_str(),
        Some("claude-independent-metadata")
    );
    assert_eq!(
        historical_drafts[0].source_remainder["message"]["usage"]["input_tokens"].as_u64(),
        Some(17)
    );
    assert_eq!(historical_drafts[1].kind, "tool_result");
    assert_eq!(historical_drafts[1].occurred_at, Some(expected_millis));
    assert_eq!(
        historical_drafts[1].source_remainder["is_error"].as_bool(),
        Some(true)
    );

    let typed_content = vec![
        ClaudeContent::Opaque(ClaudeProviderObject::new(vec![
            ClaudeProviderFact::new("type", "thinking"),
            ClaudeProviderFact::new("thinking", "first opaque provider message"),
            ClaudeProviderFact::new("signature", "opaque-signature"),
        ])),
        ClaudeContent::ToolResult {
            tool_use_id: "typed-tool-result".into(),
            content: ClaudeProviderValue::from("tool failed honestly"),
        },
    ];
    let message_extras = ClaudeProviderObject::new(vec![
        ClaudeProviderFact::new("model", "claude-independent-metadata"),
        ClaudeProviderFact::new(
            "usage",
            ClaudeProviderValue::object(vec![
                ClaudeProviderFact::new("input_tokens", 17_u64),
                ClaudeProviderFact::new("output_tokens", 5_u64),
            ]),
        ),
    ]);
    let item_remainders = vec![
        ClaudeProviderObject::default(),
        ClaudeProviderObject::new(vec![ClaudeProviderFact::new("is_error", true)]),
    ];
    assert!(
        ClaudeMessage::new(
            ClaudeMessageRole::Assistant,
            Some("typed-metadata-message".into()),
            typed_content.clone()
        )
        .with_provider_metadata(
            message_extras.clone(),
            vec![ClaudeProviderObject::default()]
        )
        .is_err(),
        "the metadata builder rejects a remainder count that does not match the content count"
    );
    let typed_message = ClaudeMessage::new(
        ClaudeMessageRole::Assistant,
        Some("typed-metadata-message".into()),
        typed_content,
    )
    .with_provider_metadata(message_extras, item_remainders)
    .expect("one typed remainder per content item");
    let typed_occurrence = ClaudeOccurrence::message(
        ClaudeOccurrenceMeta::new(
            SESSION,
            Some("90000000-0000-4000-8000-000000000009".into()),
            Some("2026-09-10T12:34:56.123456789Z".into()),
            None,
        ),
        typed_message,
    );

    let mut assembler = ClaudeOccurrenceAssembler::new();
    let mut resolver = MissingPriorRequest;
    let typed_drafts = assembler
        .assemble(
            typed_occurrence,
            ClaudeAssemblyContext {
                session_ref: SESSION.into(),
                source_instance_ref: "independent-typed-metadata".into(),
                record: 1,
                record_start_byte: 0,
            },
            &mut resolver,
        )
        .expect("assemble typed provider values without JSON construction or serialization");
    assert_eq!(
        typed_drafts, historical_drafts,
        "historical and typed paths retain identical model, usage, opaque, and error-result metadata"
    );

    let (_historical_directory, historical_store, historical_context) =
        event_store("typed-metadata-historical");
    let historical_receipt = import_source(
        &historical_store,
        historical_context,
        "independent-typed-metadata",
        SOURCE,
        None,
    )
    .expect("import historical typed-metadata source");
    let historical_events = stored_events(
        &historical_store,
        historical_context,
        &historical_receipt.source.source_instance_ref,
    );

    let (_typed_directory, typed_store, typed_context) = event_store("typed-metadata-typed");
    typed_store
        .append_event_batch(AppendEventBatch::live(
            typed_context,
            event_adapter_declaration(),
            &historical_receipt.source.source_instance_ref,
            typed_drafts,
        ))
        .expect("persist typed provider drafts");
    let typed_events = stored_events(
        &typed_store,
        typed_context,
        &historical_receipt.source.source_instance_ref,
    );
    assert_eq!(typed_events.len(), historical_events.len());
    for (typed, historical) in typed_events.iter().zip(&historical_events) {
        assert_eq!(typed.occurred_at, Some(expected_millis));
        assert_eq!(typed.occurred_at, historical.occurred_at);
        assert_eq!(typed.source_event_key, historical.source_event_key);
        assert_eq!(typed.source_position, historical.source_position);
        assert_eq!(typed.author, historical.author);
        assert_eq!(typed.kind, historical.kind);
        assert_eq!(typed.render_role, historical.render_role);
        assert_eq!(typed.interaction_ref, historical.interaction_ref);
        assert_eq!(typed.text, historical.text);
        assert_eq!(typed.payload, historical.payload);
        assert_eq!(typed.anchors, historical.anchors);
        assert_eq!(typed.source_remainder, historical.source_remainder);
        assert_eq!(typed.coverage, historical.coverage);
    }
}
