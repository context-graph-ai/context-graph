use std::ffi::OsString;
use std::io::{BufReader, Cursor};
use std::num::{NonZeroU32, NonZeroUsize};

use cg_connector_claude_code::{CONNECTOR_ID, ClaudeCodeJsonlAdapter};
use context_graph::{
    CgError, CreateContext, EventBatchLimits, EventRange, EventSourceAdapter, EventSourceIdentity,
    EventStreamSelector, ImportEventOptions, ImportEventReceipt, Store, StoreConfig,
};
use tempfile::tempdir;

use super::{
    assemble_records_from_source, assert_facts, assert_json_field_role, expected_facts,
    fixture_bytes, fixture_records, open_fixture_store, serialized_store_rows,
};

fn identify(
    file_stem: &str,
    bytes: &[u8],
    locator: &str,
    session_override: Option<&str>,
) -> EventSourceIdentity {
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from(file_stem));
    adapter
        .identify_source(
            &mut BufReader::new(Cursor::new(bytes)),
            locator,
            session_override,
        )
        .expect("identify fixture source")
}

fn limits(max_events: usize, max_bytes: usize) -> EventBatchLimits {
    EventBatchLimits {
        max_events: NonZeroUsize::new(max_events).expect("nonzero event bound"),
        max_normalized_bytes: NonZeroUsize::new(max_bytes).expect("nonzero byte bound"),
    }
}

fn import_bytes(
    store: &Store,
    context_id: context_graph::ContextId,
    file_stem: &str,
    bytes: &[u8],
    locator: &str,
    session_override: Option<&str>,
    batch_limits: EventBatchLimits,
) -> context_graph::Result<ImportEventReceipt> {
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from(file_stem));
    store.import_event_stream(
        &mut adapter,
        &mut Cursor::new(bytes.to_vec()),
        ImportEventOptions {
            context_id,
            source_locator_ref: locator.into(),
            session_override: session_override.map(str::to_owned),
            batch_limits,
        },
    )
}

fn source_events(
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
                limit: NonZeroU32::new(32).expect("nonzero event read limit"),
            },
        )
        .expect("read imported source events")
}

fn complete_record_bytes(name: &str, record: u64) -> Vec<u8> {
    let mut bytes = fixture_records(name)
        .into_iter()
        .find_map(|(ordinal, _, bytes)| (ordinal == record).then_some(bytes))
        .expect("fixture record");
    bytes.push(b'\n');
    bytes
}

#[test]
fn identity_and_nullable_time_facts_cover_the_last_primary_records_and_second_stream() {
    let primary_identity = identify(
        "compact-rich",
        fixture_bytes("compact-rich.jsonl"),
        "fixture://primary",
        None,
    );
    let second_identity = identify(
        "second-stream-same-session",
        fixture_bytes("second-stream-same-session.jsonl"),
        "fixture://second",
        None,
    );
    assert_eq!(primary_identity.session_ref, second_identity.session_ref);
    assert_ne!(
        primary_identity.source_instance_ref, second_identity.source_instance_ref,
        "same-session files with different first occurrences are separate source streams"
    );

    let primary = [10, 11, 12];
    let primary_drafts = assemble_records_from_source(
        "compact-rich.jsonl",
        &primary,
        &primary_identity.source_instance_ref,
    );
    assert_facts(
        "compact-rich.jsonl",
        &primary_drafts,
        &expected_facts("compact-rich.jsonl", &primary),
    );

    let second = [1];
    let second_drafts = assemble_records_from_source(
        "second-stream-same-session.jsonl",
        &second,
        &second_identity.source_instance_ref,
    );
    assert_facts(
        "second-stream-same-session.jsonl",
        &second_drafts,
        &expected_facts("second-stream-same-session.jsonl", &second),
    );

    let directory = tempdir().expect("source identity Store directory");
    let store = open_fixture_store(StoreConfig {
        db_path: directory.path().join("source-identities.db"),
        ..StoreConfig::default()
    });
    let context = store
        .create_context(CreateContext {
            name: "source identity".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create source identity Context");
    let primary_receipt = import_bytes(
        &store,
        context.id,
        "compact-rich",
        &complete_record_bytes("compact-rich.jsonl", 1),
        "fixture://primary-first-record",
        None,
        EventBatchLimits::default(),
    )
    .expect("core rewinds after primary identification and imports record one");
    let second_receipt = import_bytes(
        &store,
        context.id,
        "second-stream-same-session",
        fixture_bytes("second-stream-same-session.jsonl"),
        "fixture://second-first-record",
        None,
        EventBatchLimits::default(),
    )
    .expect("core rewinds after second-stream identification and imports record one");
    assert_eq!(primary_receipt.inserted, 1);
    assert_eq!(second_receipt.inserted, 1);
    assert_eq!(primary_receipt.source, primary_identity);
    assert_eq!(second_receipt.source, second_identity);
    assert_eq!(
        source_events(
            &store,
            context.id,
            &primary_receipt.source.source_instance_ref,
        )[0]
        .text
        .as_deref(),
        Some(
            "Account A-17 needs a transfer review. Is the notice period enforceable?\n\tPlease preserve this exact tab and café 東京."
        ),
        "the first record is decoded after core rewinds from identity inspection"
    );
}

#[test]
fn exact_file_override_rejects_mixed_embedded_ids_and_reader_preserves_variants() {
    let directory = tempdir().expect("recovery Store directory");
    let store = open_fixture_store(StoreConfig {
        db_path: directory.path().join("recovery.db"),
        ..StoreConfig::default()
    });
    let context = store
        .create_context(CreateContext {
            name: "connector recovery".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create recovery Context");

    let malformed_identity = identify(
        "malformed-after-valid",
        include_bytes!("../fixtures/source-oracle/malformed-after-valid.jsonl"),
        "fixture://malformed",
        None,
    );
    let malformed = import_bytes(
        &store,
        context.id,
        "malformed-after-valid",
        include_bytes!("../fixtures/source-oracle/malformed-after-valid.jsonl"),
        "fixture://malformed",
        None,
        limits(1, 1024 * 1024),
    )
    .expect_err("newline-terminated malformed record is a typed refusal");
    assert!(
        matches!(malformed, CgError::SourceRecordMalformed { connector_id, .. } if connector_id == CONNECTOR_ID)
    );
    let committed_before_malformed =
        source_events(&store, context.id, &malformed_identity.source_instance_ref);
    assert_eq!(committed_before_malformed.len(), 1);
    assert_eq!(
        committed_before_malformed[0].text.as_deref(),
        Some("This record is committable before the malformed line."),
        "the complete earlier batch remains durable"
    );

    let pending_partial = import_bytes(
        &store,
        context.id,
        "pending-tail-partial",
        include_bytes!("../fixtures/source-oracle/pending-tail-partial.jsonl"),
        "fixture://pending",
        None,
        EventBatchLimits::default(),
    )
    .expect("import complete record before pending tail");
    assert_eq!(pending_partial.inserted, 1);
    assert_eq!(pending_partial.pending_tail_records, 1);
    let checkpoint_before_resume = serialized_store_rows(&store, Some("event_source_checkpoints"));

    let pending_complete = import_bytes(
        &store,
        context.id,
        "pending-tail-complete",
        include_bytes!("../fixtures/source-oracle/pending-tail-complete.jsonl"),
        "fixture://pending",
        None,
        EventBatchLimits::default(),
    )
    .expect("fresh adapter resumes from committed checkpoint");
    assert_eq!(pending_complete.inserted, 1);
    assert_eq!(pending_complete.pending_tail_records, 0);
    assert_eq!(pending_complete.source, pending_partial.source);
    let pending_events = source_events(
        &store,
        context.id,
        &pending_complete.source.source_instance_ref,
    );
    assert_eq!(pending_events.len(), 2);
    assert_eq!(pending_events[0].source_position["record"], 1);
    assert_eq!(pending_events[1].source_position["record"], 2);
    assert_ne!(
        serialized_store_rows(&store, Some("event_source_checkpoints")),
        checkpoint_before_resume,
        "resume advances the durable source checkpoint"
    );
    let unchanged = import_bytes(
        &store,
        context.id,
        "pending-tail-complete",
        include_bytes!("../fixtures/source-oracle/pending-tail-complete.jsonl"),
        "fixture://pending",
        None,
        EventBatchLimits::default(),
    )
    .expect("unchanged completed source resumes as a no-op");
    assert_eq!(unchanged.inserted, 0);
    assert_eq!(unchanged.batch_count, 0);

    let whitespace = import_bytes(
        &store,
        context.id,
        "whitespace-only",
        include_bytes!("../fixtures/source-oracle/whitespace-only.txt"),
        "fixture://whitespace",
        None,
        EventBatchLimits::default(),
    )
    .expect("whitespace-only source checkpoints successfully");
    assert_eq!(whitespace.inserted, 0);
    assert!(whitespace.source.provisional_empty);
    let upgraded = import_bytes(
        &store,
        context.id,
        "whitespace-first-record",
        include_bytes!("../fixtures/source-oracle/whitespace-first-record.jsonl"),
        "fixture://whitespace",
        None,
        EventBatchLimits::default(),
    )
    .expect("unchanged whitespace prefix upgrades to first-record identity");
    assert_eq!(upgraded.inserted, 1);
    assert!(!upgraded.source.provisional_empty);
    assert_ne!(
        whitespace.source.source_instance_ref,
        upgraded.source.source_instance_ref
    );

    let oversized = import_bytes(
        &store,
        context.id,
        "oversized-tail",
        include_bytes!("../fixtures/source-oracle/oversized-tail.jsonl"),
        "fixture://oversized",
        None,
        limits(512, 256),
    )
    .expect("one oversized complete record remains an atomic batch before a pending tail");
    assert_eq!(oversized.inserted, 1);
    assert_eq!(oversized.batch_count, 1);
    assert_eq!(oversized.pending_tail_records, 1);
    let oversized_events = source_events(&store, context.id, &oversized.source.source_instance_ref);
    assert_eq!(oversized_events.len(), 1);
    let oversized_source: serde_json::Value =
        serde_json::from_slice(&complete_record_bytes("oversized-tail.jsonl", 1))
            .expect("independently parse the complete oversized source record");
    let exact_oversized_text = oversized_source["message"]["content"]
        .as_str()
        .expect("oversized source message is text");
    assert_eq!(
        oversized_events[0].text.as_deref(),
        Some(exact_oversized_text),
        "the complete independently parsed oversized source string survives"
    );

    let override_receipt = import_bytes(
        &store,
        context.id,
        "session-override",
        include_bytes!("../fixtures/session-override.jsonl"),
        "fixture://override",
        Some("outward-session"),
        EventBatchLimits::default(),
    )
    .expect("an exact-file override is accepted");
    assert_eq!(override_receipt.source.session_ref, "outward-session");
    let override_events = source_events(
        &store,
        context.id,
        &override_receipt.source.source_instance_ref,
    );
    assert_json_field_role(
        &override_events[0].payload,
        &[&["sessionId"], &["session_id"]],
        &serde_json::json!("embedded-session"),
        "embedded session identity under an outward override",
    );

    let mixed = import_bytes(
        &store,
        context.id,
        "mixed-embedded-session",
        include_bytes!("../fixtures/mixed-embedded-session.jsonl"),
        "fixture://mixed",
        Some("outward-session"),
        limits(1, 1024 * 1024),
    )
    .expect_err("an override cannot hide conflicting embedded session IDs");
    assert!(matches!(mixed, CgError::SourceRecordMalformed { .. }));

    let fallback_bytes = include_bytes!("../fixtures/uuid-fallback.jsonl");
    let fallback = identify(
        "uuid-fallback",
        fallback_bytes,
        "fixture://uuid-fallback",
        None,
    );
    let copied_bytes = fallback_bytes.to_vec();
    let renamed_copy = identify(
        "renamed-copy",
        &copied_bytes,
        "fixture://renamed-copy",
        None,
    );
    let mut blank_prefixed_copy = b"\n \t\n".to_vec();
    blank_prefixed_copy.extend_from_slice(fallback_bytes);
    let after_blank_prefix = identify(
        "blank-prefixed-copy",
        &blank_prefixed_copy,
        "fixture://blank-prefixed-copy",
        None,
    );
    assert_eq!(
        fallback.source_instance_ref,
        renamed_copy.source_instance_ref
    );
    assert_eq!(
        fallback.source_instance_ref,
        after_blank_prefix.source_instance_ref
    );
    assert_eq!(fallback.source_instance_ref.len(), 64);
    assert!(
        fallback
            .source_instance_ref
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "source identity is the lowercase SHA-256 of the accepted first-record identity inputs"
    );

    let changed_first_record = std::str::from_utf8(fallback_bytes)
        .expect("UTF-8 UUID-less fixture")
        .replace(
            "first nonblank raw record supplies source identity when uuid is absent",
            "different first nonblank record must produce a distinct source identity",
        )
        .into_bytes();
    let changed = identify(
        "same-name-different-first-record",
        &changed_first_record,
        "fixture://changed-first-record",
        None,
    );
    assert_ne!(
        fallback.source_instance_ref, changed.source_instance_ref,
        "the exact accepted first-record bytes participate in UUID-less source identity"
    );

    let second_record_offset = fallback_bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("first UUID-less record is newline-terminated")
        + 1;
    let fallback_directory = tempdir().expect("UUID-less fallback Store directory");
    let fallback_path = fallback_directory.path().join("uuid-fallback.db");
    let fallback_store = open_fixture_store(StoreConfig {
        db_path: fallback_path.clone(),
        ..StoreConfig::default()
    });
    let fallback_context = fallback_store
        .create_context(CreateContext {
            name: "UUID-less fallback".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create UUID-less fallback Context");
    let first_fallback_receipt = import_bytes(
        &fallback_store,
        fallback_context.id,
        "uuid-fallback",
        &fallback_bytes[..second_record_offset],
        "fixture://uuid-fallback",
        None,
        limits(1, 1024 * 1024),
    )
    .expect("forced adapter flush imports the first UUID-less record");
    assert_eq!(first_fallback_receipt.inserted, 2);
    drop(fallback_store);

    let fallback_store = open_fixture_store(StoreConfig {
        db_path: fallback_path,
        ..StoreConfig::default()
    });
    let fallback_receipt = import_bytes(
        &fallback_store,
        fallback_context.id,
        "uuid-fallback",
        fallback_bytes,
        "fixture://uuid-fallback",
        None,
        limits(1, 1024 * 1024),
    )
    .expect("reopened Store resumes UUID-less source through the real adapter lifecycle");
    assert_eq!(fallback_receipt.source, fallback);
    assert_eq!(fallback_receipt.inserted, 1);
    let fallback_events = source_events(
        &fallback_store,
        fallback_context.id,
        &fallback_receipt.source.source_instance_ref,
    );
    assert_eq!(fallback_events.len(), 3);
    for (item, event) in fallback_events[..2].iter().enumerate() {
        assert_eq!(event.source_position["record"], 1);
        assert_eq!(event.source_position["byte"], 0);
        assert_eq!(event.source_position["item"], item);
        assert_eq!(
            event.source_event_key,
            format!("source:{}:byte:0:item:{item}", fallback.source_instance_ref),
            "UUID-less event identity uses the exact source, byte, and item coordinates"
        );
    }
    let second_record = &fallback_events[2];
    assert_eq!(second_record.source_position["record"], 2);
    assert_eq!(
        second_record.source_position["byte"],
        u64::try_from(second_record_offset).expect("fixture offset fits u64")
    );
    assert_eq!(second_record.source_position["item"], 0);
    assert_eq!(
        second_record.source_event_key,
        format!(
            "source:{}:byte:{second_record_offset}:item:0",
            fallback.source_instance_ref
        ),
        "the resumed UUID-less record uses its independently measured nonzero byte offset"
    );
    assert_ne!(
        fallback_events[0].source_event_key, second_record.source_event_key,
        "the resumed UUID-less record has a distinct source identity key from the first record"
    );
}
