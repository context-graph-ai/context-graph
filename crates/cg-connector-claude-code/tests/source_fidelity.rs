#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use cg_connector_claude_code::{
    ClaudeAssemblyContext, ClaudeOccurrenceAssembler, CorrelationResolution,
    EventCorrelationResolver, PriorRequestAnchorQuery, decode_jsonl_record,
};
use chrono::{DateTime, Utc};
use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::{
    Embedder, EmbedderRegistration, EventDraft, EventRenderRole, Store, StoreConfig,
};
use serde_json::Value;

const SESSION: &str = "synthetic-session-amber";
const SOURCE: &str = "source-instance-fixture";

fn open_fixture_store(mut config: StoreConfig) -> Store {
    config.default_text_embedder = None;
    let embedder = std::sync::Arc::new(LexicalMockEmbedder::default());
    Store::open_with_embedder_registrations(
        config,
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open fixture store with lexical test embedder")
}

struct FixtureResolver;

impl EventCorrelationResolver for FixtureResolver {
    fn resolve_prior_request_anchor(
        &mut self,
        query: &PriorRequestAnchorQuery,
    ) -> context_graph::Result<CorrelationResolution> {
        Ok(
            if query.interaction_ref == "tool-edit-1"
                && query.interaction_anchor_key == "tool_use_id"
                && query.projected_anchor_key == "tool_name"
            {
                CorrelationResolution::Unique(Value::String("Edit".into()))
            } else {
                CorrelationResolution::Missing
            },
        )
    }
}

fn expected() -> Value {
    serde_json::from_slice(include_bytes!(
        "fixtures/source-oracle/expected-occurrences.json"
    ))
    .expect("immutable source oracle JSON")
}

fn fixture_bytes(name: &str) -> &'static [u8] {
    match name {
        "compact-rich.jsonl" => include_bytes!("fixtures/source-oracle/compact-rich.jsonl"),
        "second-stream-same-session.jsonl" => {
            include_bytes!("fixtures/source-oracle/second-stream-same-session.jsonl")
        }
        "oversized-tail.jsonl" => include_bytes!("fixtures/source-oracle/oversized-tail.jsonl"),
        "refused-media-valid-base64.jsonl" => {
            include_bytes!("fixtures/refused-media-valid-base64.jsonl")
        }
        "refused-media-nested-and-urls.jsonl" => {
            include_bytes!("fixtures/refused-media-nested-and-urls.jsonl")
        }
        "typed-parity-rich.jsonl" => include_bytes!("fixtures/typed-parity-rich.jsonl"),
        other => panic!("unknown fixture {other}"),
    }
}

fn fixture_records(name: &str) -> Vec<(u64, u64, Vec<u8>)> {
    let mut offset = 0_u64;
    let mut record_ordinal = 0_u64;
    fixture_bytes(name)
        .split_inclusive(|byte| *byte == b'\n')
        .filter_map(|record| {
            let start = offset;
            offset += u64::try_from(record.len()).expect("fixture length fits u64");
            if record.iter().all(u8::is_ascii_whitespace) {
                return None;
            }
            record_ordinal += 1;
            Some((
                record_ordinal,
                start,
                record.strip_suffix(b"\n").unwrap_or(record).to_vec(),
            ))
        })
        .collect()
}

fn assemble_records(name: &str, records: &[u64]) -> Vec<EventDraft> {
    assemble_records_from_source(name, records, SOURCE)
}

fn assemble_records_from_source(
    name: &str,
    records: &[u64],
    source_instance_ref: &str,
) -> Vec<EventDraft> {
    assemble_records_with_context(name, records, SESSION, source_instance_ref)
}

fn assemble_records_with_context(
    name: &str,
    records: &[u64],
    session_ref: &str,
    source_instance_ref: &str,
) -> Vec<EventDraft> {
    let mut assembler = ClaudeOccurrenceAssembler::new();
    let mut resolver = FixtureResolver;
    fixture_records(name)
        .into_iter()
        .filter(|(record, _, _)| records.contains(record))
        .flat_map(|(record, record_start_byte, raw)| {
            let occurrence = decode_jsonl_record(&raw).expect("decode fixture record");
            assembler
                .assemble(
                    occurrence,
                    ClaudeAssemblyContext {
                        session_ref: session_ref.into(),
                        source_instance_ref: source_instance_ref.into(),
                        record,
                        record_start_byte,
                    },
                    &mut resolver,
                )
                .expect("assemble fixture record")
        })
        .collect()
}

fn expected_facts(name: &str, records: &[u64]) -> Vec<Value> {
    expected()["streams"][name]["accepted_occurrences"]
        .as_array()
        .expect("oracle facts")
        .iter()
        .filter(|fact| records.contains(&fact["record"].as_u64().expect("record number")))
        .cloned()
        .collect()
}

fn expected_author(fact: &Value) -> &str {
    match fact["author"].as_str().expect("oracle author") {
        "effective user identity" => "user",
        "effective assistant identity" => "assistant",
        "connector:claude-code unless a more honest nested identity exists" => {
            "connector:claude-code"
        }
        author => author,
    }
}

fn json_matches_at(candidate: &Value, expected: &Value) -> bool {
    match (candidate, expected) {
        (Value::Object(candidate), Value::Object(expected)) => {
            expected.iter().all(|(key, value)| {
                candidate
                    .get(key)
                    .is_some_and(|candidate| json_matches_at(candidate, value))
            })
        }
        (Value::Array(candidate), Value::Array(expected)) => {
            candidate.len() == expected.len()
                && candidate
                    .iter()
                    .zip(expected)
                    .all(|(candidate, expected)| json_matches_at(candidate, expected))
        }
        _ => candidate == expected,
    }
}

fn json_contains(container: &Value, expected: &Value) -> bool {
    if json_matches_at(container, expected) {
        return true;
    }
    match container {
        Value::Array(values) => values.iter().any(|value| json_contains(value, expected)),
        Value::Object(fields) => fields.values().any(|value| json_contains(value, expected)),
        _ => false,
    }
}

fn json_matches_path_at(candidate: &Value, path: &[&str], expected: &Value) -> bool {
    let Some((field, remaining)) = path.split_first() else {
        return json_matches_at(candidate, expected);
    };
    candidate
        .as_object()
        .and_then(|fields| fields.get(*field))
        .is_some_and(|value| json_matches_path_at(value, remaining, expected))
}

fn json_contains_path(container: &Value, path: &[&str], expected: &Value) -> bool {
    if json_matches_path_at(container, path, expected) {
        return true;
    }
    match container {
        Value::Array(values) => values
            .iter()
            .any(|value| json_contains_path(value, path, expected)),
        Value::Object(fields) => fields
            .values()
            .any(|value| json_contains_path(value, path, expected)),
        _ => false,
    }
}

fn assert_json_field_role(
    container: &Value,
    allowed_paths: &[&[&str]],
    expected: &Value,
    role: &str,
) {
    assert!(
        allowed_paths
            .iter()
            .any(|path| json_contains_path(container, path, expected)),
        "{role} must retain its own field/value relation for {expected}; got {container}"
    );
}

fn assert_json_retained(container: &Value, expected: &Value, location: &str) {
    assert!(
        json_contains(container, expected),
        "{location} must retain {expected}; got {container}"
    );
}

fn assert_anchor(draft: &EventDraft, key: &str, expected: &Value) {
    assert_eq!(
        draft.anchors.get(key),
        Some(expected),
        "declared anchor `{key}` must retain its explicit source value"
    );
}

fn assert_required_source_values(draft: &EventDraft, fact: &Value, embedded_session: &Value) {
    assert_json_field_role(
        &draft.payload,
        &[&["sessionId"], &["session_id"]],
        embedded_session,
        "embedded session metadata",
    );
    if let Some(message_id) = fact.get("message_id") {
        assert_json_field_role(
            &draft.payload,
            &[&["message_id"], &["message", "id"]],
            message_id,
            "message identity metadata",
        );
    }

    let required = fact["required_source_fields"]
        .as_object()
        .expect("required source fields object");
    for (source_field, expected_value) in required {
        let descriptive_text = expected_value.as_str().unwrap_or_default();
        if source_field.ends_with(".input") {
            assert_json_field_role(
                &draft.payload,
                &[&["input"]],
                expected_value,
                "normalized tool input",
            );
            if let Some(file) = expected_value.get("file_path") {
                assert_anchor(draft, "file", file);
            }
            if let Some(command) = expected_value.get("command") {
                assert_anchor(draft, "command", command);
            }
        } else if source_field.ends_with(".content")
            && !descriptive_text.starts_with("exact string")
        {
            assert_json_field_role(
                &draft.payload,
                &[&["content"]],
                expected_value,
                "normalized tool result",
            );
            if let Some(exit_code) = expected_value.get("exit_code") {
                assert_anchor(draft, "exit_code", exit_code);
            }
        } else if source_field == "message.content[1]" || source_field == "top_level_object" {
            assert_json_retained(
                &draft.source_remainder,
                expected_value,
                "opaque source remainder",
            );
        } else if let Some(subagent_field) = source_field.strip_prefix("subagent.") {
            assert_json_field_role(
                &draft.payload,
                &[&["subagent", subagent_field]],
                expected_value,
                "nested subagent metadata",
            );
        } else if source_field.ends_with(".filename") {
            assert_json_field_role(
                &draft.payload,
                &[&["filename"]],
                expected_value,
                "attachment filename metadata",
            );
        } else if source_field.ends_with(".reference") {
            assert_json_field_role(
                &draft.payload,
                &[&["reference"]],
                expected_value,
                "attachment reference metadata",
            );
        } else if source_field.ends_with(".source.media_type") {
            assert_json_field_role(
                &draft.payload,
                &[&["source", "media_type"], &["media_type"]],
                expected_value,
                "media type metadata",
            );
        } else if source_field.ends_with(".media_type") {
            assert_json_field_role(
                &draft.payload,
                &[&["media_type"]],
                expected_value,
                "attachment media type metadata",
            );
        } else if source_field.ends_with(".source.type") {
            assert_json_field_role(
                &draft.payload,
                &[&["source", "type"]],
                expected_value,
                "media source metadata",
            );
        } else if source_field == "sessionId" {
            assert_json_field_role(
                &draft.payload,
                &[&["sessionId"], &["session_id"]],
                expected_value,
                "normalized embedded session metadata",
            );
        } else if source_field == "timestamp"
            && let Some(raw_timestamp) =
                descriptive_text.strip_suffix(" retained safely in source remainder")
        {
            assert_json_field_role(
                &draft.source_remainder,
                &[&["timestamp"]],
                &Value::String(raw_timestamp.into()),
                "invalid timestamp remainder",
            );
        }

        if source_field.contains("message.content") && source_field.ends_with(".id") {
            assert_json_field_role(
                &draft.payload,
                &[&["tool_use_id"]],
                expected_value,
                "normalized tool-use id",
            );
            assert_anchor(draft, "tool_use_id", expected_value);
        } else if source_field.ends_with(".tool_use_id") {
            assert_json_field_role(
                &draft.payload,
                &[&["tool_use_id"]],
                expected_value,
                "normalized result link",
            );
            assert_anchor(draft, "tool_use_id", expected_value);
        } else if source_field.contains("message.content") && source_field.ends_with(".name") {
            assert_json_field_role(
                &draft.payload,
                &[&["tool_name"]],
                expected_value,
                "normalized tool name",
            );
            assert_anchor(draft, "tool_name", expected_value);
        }
    }
}

fn assert_kind_and_role(draft: &EventDraft, fact: &Value) {
    let expected_kind = fact["kind"].as_str().expect("oracle kind");
    if expected_kind == "attachment or connector-declared media metadata kind" {
        assert!(
            draft.kind == "attachment"
                || (draft.render_role == EventRenderRole::Metadata
                    && !draft.kind.trim().is_empty()
                    && draft.kind != expected_kind),
            "refused media keeps the approved attachment kind or a nonempty connector-declared metadata kind"
        );
    } else {
        assert_eq!(draft.kind, expected_kind);
    }

    let expected_role = fact["render_role"].as_str().expect("oracle render role");
    if expected_role == "attachment or metadata" {
        assert!(matches!(
            draft.render_role,
            EventRenderRole::Attachment | EventRenderRole::Metadata
        ));
    } else {
        assert_eq!(
            serde_json::to_value(draft.render_role).expect("render role value"),
            fact["render_role"],
        );
    }
}

fn assert_facts(name: &str, drafts: &[EventDraft], facts: &[Value]) {
    assert_eq!(
        drafts.len(),
        facts.len(),
        "every oracle occurrence is retained"
    );
    let oracle = expected();
    let stream = &oracle["streams"][name];
    let embedded_session = &stream["session_ref"];
    let records = fixture_records(name);
    for (draft, fact) in drafts.iter().zip(facts) {
        let record = fact["record"].as_u64().expect("record number");
        let item = fact["item"].as_u64().expect("item number");
        let expected_start = records
            .iter()
            .find_map(|(candidate, start, _)| (*candidate == record).then_some(*start))
            .expect("oracle record start");

        assert_eq!(draft.session_ref, SESSION);
        assert_eq!(draft.source_position["record"], fact["record"]);
        assert_eq!(draft.source_position["item"], fact["item"]);
        assert_eq!(draft.source_position["byte"], expected_start);
        if let Some(uuid) = fact.get("uuid").and_then(Value::as_str) {
            assert_eq!(
                draft.source_event_key,
                format!("uuid:{uuid}:item:{item}"),
                "the record UUID and item ordinal are occurrence identity"
            );
        }
        assert_kind_and_role(draft, fact);
        assert_eq!(draft.author, expected_author(fact));
        assert_eq!(
            serde_json::to_value(draft.coverage).expect("coverage value"),
            fact["coverage"],
        );
        if let Some(text) = fact.get("text").and_then(Value::as_str) {
            assert_eq!(draft.text.as_deref(), Some(text));
        }
        if let Some(interaction_ref) = fact.get("interaction_ref").and_then(Value::as_str) {
            assert_eq!(draft.interaction_ref.as_deref(), Some(interaction_ref));
        }
        match fact.get("occurred_at") {
            Some(Value::String(raw_time)) => {
                let expected_time = raw_time
                    .parse::<DateTime<Utc>>()
                    .expect("oracle timestamp parses at millisecond-column precision");
                assert_eq!(draft.occurred_at, Some(expected_time));
            }
            Some(Value::Null) => assert_eq!(draft.occurred_at, None),
            _ => {}
        }
        assert_required_source_values(draft, fact, embedded_session);
    }
}

fn serialized_store_rows(store: &Store, only_table: Option<&str>) -> String {
    let database = store.sync_database();
    let mut tables = database.table_names();
    tables.sort();
    let mut serialized = String::new();
    for table in tables {
        if only_table.is_some_and(|wanted| wanted != table) {
            continue;
        }
        let rows = database
            .scan(&table, database.snapshot())
            .unwrap_or_else(|error| panic!("scan {table}: {error:?}"));
        for row in rows {
            serialized.push_str(
                &serde_json::to_string(&row.values).expect("serialize Store row for leak scan"),
            );
        }
    }
    serialized
}

#[path = "source_fidelity/capture_regressions.rs"]
mod capture_regressions;
#[path = "source_fidelity/fidelity_delta.rs"]
mod fidelity_delta;
#[path = "source_fidelity/history_and_reuse.rs"]
mod history_and_reuse;
#[path = "source_fidelity/identity_recovery.rs"]
mod identity_recovery;
#[path = "source_fidelity/mapping.rs"]
mod mapping;
#[path = "source_fidelity/opaque_media.rs"]
mod opaque_media;
#[path = "source_fidelity/timestamp_boundary.rs"]
mod timestamp_boundary;

#[path = "source_fidelity/session_alias_contract.rs"]
mod session_alias_contract;
