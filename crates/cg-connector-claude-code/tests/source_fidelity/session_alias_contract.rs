//! Synthetic fork identity regressions through the public decoder and streaming adapter.
use std::ffi::OsString;
use std::io::Cursor;

use cg_connector_claude_code::{
    ClaudeCodeJsonlAdapter, ClaudeConnectorError, ClaudeContent, ClaudeMessage, ClaudeMessageRole,
    ClaudeOccurrence, ClaudeOccurrenceMeta, ClaudeOpaqueRecord, ClaudeProviderFact,
    ClaudeProviderObject, decode_jsonl_record,
};
use context_graph::{
    AdaptedSourceRecord, CgError, CorrelationResolution, EndOfSource, EventCorrelationResolver,
    EventSourceAdapter, EventSourceRead, PriorRequestAnchorQuery,
};
use serde_json::{Value, json};

const FORK: &str = "10000000-0000-4000-8000-000000000001";
const PARENT: &str = "20000000-0000-4000-8000-000000000002";
const ANCESTOR: &str = "30000000-0000-4000-8000-000000000003";

fn record(opaque: bool) -> Value {
    if opaque {
        json!({"type": "progress", "detail": "copied prefix"})
    } else {
        json!({"type": "assistant", "message": {"role": "assistant", "content": "copied prefix"}})
    }
}

fn object(fields: Vec<ClaudeProviderFact>) -> ClaudeProviderObject {
    ClaudeProviderObject::new(fields)
}

// Compare against independently constructed typed values: no decoder/serializer round trip.
fn expected(opaque: bool, session: &str, alias: Option<&str>) -> ClaudeOccurrence {
    let facts = alias
        .map(|value| vec![ClaudeProviderFact::new("session_id", value)])
        .unwrap_or_default();
    let meta = ClaudeOccurrenceMeta::new(session, None, None, None);
    if opaque {
        let mut facts = facts;
        facts.push(ClaudeProviderFact::new("detail", "copied prefix"));
        ClaudeOccurrence::opaque(
            meta,
            ClaudeOpaqueRecord::new(Some("progress".into()), object(facts)),
        )
    } else {
        ClaudeOccurrence::message(
            meta.with_envelope_extras(object(facts)),
            ClaudeMessage::new(
                ClaudeMessageRole::Assistant,
                None,
                vec![ClaudeContent::Text("copied prefix".into())],
            ),
        )
    }
}

#[test]
fn fork_primary_and_exact_parent_alias_survive_message_and_opaque_decoding() {
    for opaque in [false, true] {
        for alias in [PARENT, " parent alias with spaces ", ""] {
            let mut row = record(opaque);
            row["sessionId"] = json!(FORK);
            row["session_id"] = json!(alias);
            assert_eq!(
                decode_jsonl_record(&serde_json::to_vec(&row).unwrap()).expect("valid fork record"),
                expected(opaque, FORK, Some(alias)),
                "raw divergent string alias must survive exactly, including empty strings"
            );
        }
    }
}

#[test]
fn absent_null_or_empty_primary_consumes_legacy_fallback() {
    for opaque in [false, true] {
        for primary in [None, Some(Value::Null), Some(json!(""))] {
            let mut row = record(opaque);
            if let Some(primary) = primary {
                row["sessionId"] = primary;
            }
            row["session_id"] = json!(PARENT);
            assert_eq!(
                decode_jsonl_record(&serde_json::to_vec(&row).unwrap()).unwrap(),
                expected(opaque, PARENT, None)
            );
        }
    }
}

#[test]
fn equal_aliases_and_primary_only_keep_existing_stripping() {
    for opaque in [false, true] {
        for legacy in [None, Some(Value::Null), Some(json!(FORK))] {
            let mut row = record(opaque);
            row["sessionId"] = json!(FORK);
            if let Some(legacy) = legacy {
                row["session_id"] = legacy;
            }
            assert_eq!(
                decode_jsonl_record(&serde_json::to_vec(&row).unwrap()).unwrap(),
                expected(opaque, FORK, None)
            );
        }
    }
}

struct NoCorrelation;
impl EventCorrelationResolver for NoCorrelation {
    fn resolve_prior_request_anchor(
        &mut self,
        _: &PriorRequestAnchorQuery,
    ) -> context_graph::Result<CorrelationResolution> {
        panic!("plain synthetic records must not request tool correlation")
    }
}

fn bytes(rows: &[Value]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut bytes, row).unwrap();
        bytes.push(b'\n');
    }
    bytes
}

fn next(adapter: &mut ClaudeCodeJsonlAdapter, reader: &mut Cursor<&[u8]>) -> AdaptedSourceRecord {
    match adapter
        .read_next_record(reader, &mut NoCorrelation)
        .expect("read valid synthetic record")
    {
        EventSourceRead::Record(record) => record,
        other => panic!("expected a record, got {other:?}"),
    }
}

#[test]
fn nonstring_aliases_are_refused_even_when_the_other_alias_is_active() {
    for opaque in [false, true] {
        for bad in [json!(7), json!(false), json!([]), json!({})] {
            for key in ["sessionId", "session_id"] {
                let mut row = record(opaque);
                row["sessionId"] = json!(FORK);
                row["session_id"] = json!(PARENT);
                row[key] = bad.clone();
                let reason = format!("top-level {key} must be a string when present");
                assert!(
                    matches!(decode_jsonl_record(&serde_json::to_vec(&row).unwrap()), Err(ClaudeConnectorError::MalformedRecord { reason: actual }) if actual == reason)
                );
                let raw = bytes(&[row]);
                let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from("synthetic"));
                adapter
                    .identify_source(
                        &mut Cursor::new(raw.as_slice()),
                        "synthetic://invalid",
                        Some("outward"),
                    )
                    .unwrap();
                assert!(
                    matches!(adapter.read_next_record(&mut Cursor::new(raw.as_slice()), &mut NoCorrelation), Err(CgError::SourceRecordMalformed { reason: actual, .. }) if actual.contains(&reason))
                );
            }
        }
    }
}

#[test]
fn copied_prefix_stream_infers_fork_identity_and_preserves_each_ancestor() {
    let mut rows = Vec::new();
    for (index, alias) in [ANCESTOR, PARENT, FORK].into_iter().enumerate() {
        let mut row = record(index == 1);
        row["sessionId"] = json!(FORK);
        row["session_id"] = json!(alias);
        row["provider"] = json!({"session_id": "nested untouched", "flags": [true, null, 9]});
        if index != 1 {
            row["message"]["vendor"] = json!({"session_id": "message nested"});
        }
        rows.push(row);
    }
    let raw = bytes(&rows);
    for stem in ["renamed-pathless-synthetic", PARENT] {
        let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from(stem));
        let identity = adapter
            .identify_source(&mut Cursor::new(raw.as_slice()), "synthetic://fork", None)
            .unwrap();
        assert_eq!(
            identity.session_ref, FORK,
            "embedded primary beats the file stem"
        );
        let mut reader = Cursor::new(raw.as_slice());
        for (index, alias) in [ANCESTOR, PARENT, FORK].into_iter().enumerate() {
            let decoded = next(&mut adapter, &mut reader);
            assert_eq!(decoded.checkpoint_after_record["embedded_session_id"], FORK);
            assert_eq!(decoded.drafts.len(), 1);
            let draft = &decoded.drafts[0];
            assert_eq!(draft.session_ref, FORK);
            assert_eq!(draft.payload["sessionId"], FORK);
            assert_eq!(
                draft.source_remainder["provider"],
                json!({"session_id": "nested untouched", "flags": [true, null, 9]})
            );
            if alias == FORK {
                assert!(draft.source_remainder.get("session_id").is_none());
            } else {
                assert_eq!(draft.source_remainder["session_id"], alias);
            }
            if index != 1 {
                assert_eq!(
                    draft.source_remainder["message"]["vendor"],
                    json!({"session_id": "message nested"})
                );
            }
        }
        assert!(matches!(
            adapter
                .read_next_record(&mut reader, &mut NoCorrelation)
                .unwrap(),
            EventSourceRead::EndOfSource(EndOfSource::Complete)
        ));
    }
}

#[test]
fn stream_fallback_identity_accepts_absent_null_and_empty_primary() {
    for primary in [None, Some(Value::Null), Some(json!(""))] {
        let mut row = record(false);
        if let Some(primary) = primary {
            row["sessionId"] = primary;
        }
        row["session_id"] = json!(PARENT);
        let raw = bytes(&[row]);
        let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from("unrelated-name"));
        let identity = adapter
            .identify_source(
                &mut Cursor::new(raw.as_slice()),
                "synthetic://fallback",
                None,
            )
            .unwrap();
        assert_eq!(identity.session_ref, PARENT);
        let decoded = next(&mut adapter, &mut Cursor::new(raw.as_slice()));
        assert_eq!(decoded.drafts[0].payload["sessionId"], PARENT);
        assert!(
            decoded.drafts[0]
                .source_remainder
                .get("session_id")
                .is_none()
        );
    }
}

#[test]
fn changed_primary_still_fails_after_a_valid_record_even_with_outward_override() {
    let mut first = record(false);
    first["sessionId"] = json!(FORK);
    first["session_id"] = json!(FORK);
    let mut second = record(true);
    second["sessionId"] = json!(PARENT);
    second["session_id"] = json!(FORK);
    let raw = bytes(&[first, second]);
    for session_override in [None, Some("outward")] {
        let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from("synthetic"));
        adapter
            .identify_source(
                &mut Cursor::new(raw.as_slice()),
                "synthetic://changed",
                session_override,
            )
            .unwrap();
        let mut reader = Cursor::new(raw.as_slice());
        next(&mut adapter, &mut reader);
        assert!(
            matches!(adapter.read_next_record(&mut reader, &mut NoCorrelation), Err(CgError::SourceRecordMalformed { reason, position, .. }) if reason.contains("differs from the source's first embedded session") && position["record"] == 2)
        );
    }
}
