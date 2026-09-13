use cg_connector_claude_code::{
    CONNECTOR_ID, ClaudeAssemblyContext, ClaudeCodeJsonlAdapter, ClaudeContent, ClaudeMessage,
    ClaudeMessageRole, ClaudeOccurrence, ClaudeOccurrenceAssembler, ClaudeOccurrenceMeta,
    ClaudeProviderFact, ClaudeProviderValue, CorrelationResolution, EventCorrelationResolver,
    PriorRequestAnchorQuery,
};
use context_graph::{
    ContextId, CreateContext, EventBatchLimits, EventRange, EventStreamSelector,
    ImportEventOptions, Store, StoreConfig,
};
use serde_json::{Value, json};
use std::ffi::OsString;
use std::io::Cursor;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::Path;
use tempfile::tempdir;

use super::{SESSION, SOURCE, assemble_records, assert_facts, expected_facts, open_fixture_store};

struct RecordingResolver {
    response: Option<CorrelationResolution>,
    calls: Vec<(String, String, String)>,
}

impl RecordingResolver {
    fn returning(response: CorrelationResolution) -> Self {
        Self {
            response: Some(response),
            calls: Vec::new(),
        }
    }
}

impl EventCorrelationResolver for RecordingResolver {
    fn resolve_prior_request_anchor(
        &mut self,
        query: &PriorRequestAnchorQuery,
    ) -> context_graph::Result<CorrelationResolution> {
        self.calls.push((
            query.interaction_ref.clone(),
            query.interaction_anchor_key.clone(),
            query.projected_anchor_key.clone(),
        ));
        Ok(self
            .response
            .take()
            .expect("one resolver response per standalone result"))
    }
}

fn standalone_result_author(
    interaction_ref: &str,
    response: CorrelationResolution,
    record: u64,
) -> (String, Vec<(String, String, String)>) {
    let mut assembler = ClaudeOccurrenceAssembler::new();
    let mut resolver = RecordingResolver::returning(response);
    let occurrence = ClaudeOccurrence::message(
        ClaudeOccurrenceMeta::new(
            SESSION,
            Some(format!("80000000-0000-4000-8000-{record:012}")),
            None,
            None,
        ),
        ClaudeMessage::new(
            ClaudeMessageRole::User,
            Some(format!("resolver-result-{record}")),
            vec![ClaudeContent::ToolResult {
                tool_use_id: interaction_ref.into(),
                content: "resolver result".into(),
            }],
        ),
    );
    let drafts = assembler
        .assemble(
            occurrence,
            ClaudeAssemblyContext {
                session_ref: SESSION.into(),
                source_instance_ref: SOURCE.into(),
                record,
                record_start_byte: record * 100,
            },
            &mut resolver,
        )
        .expect("assemble standalone tool result");
    assert_eq!(drafts.len(), 1);
    (drafts[0].author.clone(), resolver.calls)
}

fn assert_resolver_outcomes_and_occurrence_local_correlation() {
    let cases = [
        (
            "unique-string",
            CorrelationResolution::Unique(serde_json::json!("Bash")),
            "tool:Bash",
        ),
        ("missing", CorrelationResolution::Missing, "tool"),
        ("ambiguous", CorrelationResolution::Ambiguous, "tool"),
        (
            "unique-non-string",
            CorrelationResolution::Unique(serde_json::json!({"unexpected": "shape"})),
            "tool",
        ),
    ];
    for (index, (interaction_ref, response, expected_author)) in cases.into_iter().enumerate() {
        let record = u64::try_from(index + 20).expect("small record ordinal");
        let (author, calls) = standalone_result_author(interaction_ref, response, record);
        assert_eq!(author, expected_author);
        assert_eq!(
            calls,
            vec![(
                interaction_ref.into(),
                "tool_use_id".into(),
                "tool_name".into(),
            )],
            "attribution depends on the exact resolver query and returned evidence"
        );
    }

    let mut assembler = ClaudeOccurrenceAssembler::new();
    let mut resolver = RecordingResolver::returning(CorrelationResolution::Unique(
        serde_json::json!("resolver-must-not-win"),
    ));
    let same_occurrence = ClaudeOccurrence::message(
        ClaudeOccurrenceMeta::new(
            SESSION,
            Some("80000000-0000-4000-8000-000000000099".into()),
            None,
            None,
        ),
        ClaudeMessage::new(
            ClaudeMessageRole::Assistant,
            Some("same-occurrence-correlation".into()),
            vec![
                ClaudeContent::ToolUse {
                    id: "same-occurrence-tool".into(),
                    name: "Write".into(),
                    input: ClaudeProviderValue::object(vec![ClaudeProviderFact::new(
                        "file_path",
                        "/workspace/same-occurrence.md",
                    )]),
                },
                ClaudeContent::ToolResult {
                    tool_use_id: "same-occurrence-tool".into(),
                    content: "written".into(),
                },
            ],
        ),
    );
    let drafts = assembler
        .assemble(
            same_occurrence,
            ClaudeAssemblyContext {
                session_ref: SESSION.into(),
                source_instance_ref: SOURCE.into(),
                record: 99,
                record_start_byte: 9_900,
            },
            &mut resolver,
        )
        .expect("assemble occurrence-local request and result");
    assert_eq!(drafts.len(), 2);
    assert_eq!(drafts[1].author, "tool:Write");
    assert!(
        resolver.calls.is_empty(),
        "an earlier request in the current occurrence resolves before Store correlation"
    );
}

fn correlation_source(records: &[Value]) -> Vec<u8> {
    let mut source = Vec::new();
    for record in records {
        source.extend(serde_json::to_vec(record).expect("serialize independent correlation input"));
        source.push(b'\n');
    }
    source
}

fn tool_request(uuid: &str, session: &str, interaction_ref: &str, name: &str) -> Value {
    json!({
        "type": "assistant",
        "uuid": uuid,
        "sessionId": session,
        "timestamp": "2026-09-08T10:00:00Z",
        "message": {
            "id": format!("message-{uuid}"),
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": interaction_ref,
                "name": name,
                "input": {"command": format!("run-{name}")}
            }]
        }
    })
}

fn tool_result(uuid: &str, session: &str, interaction_ref: &str) -> Value {
    json!({
        "type": "user",
        "uuid": uuid,
        "sessionId": session,
        "timestamp": "2026-09-08T10:01:00Z",
        "message": {
            "id": format!("message-{uuid}"),
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": interaction_ref,
                "content": {"stdout": format!("result-{interaction_ref}"), "exit_code": 0}
            }]
        }
    })
}

fn correlation_limits(max_events: usize) -> EventBatchLimits {
    EventBatchLimits {
        max_events: NonZeroUsize::new(max_events).expect("nonzero correlation event limit"),
        max_normalized_bytes: NonZeroUsize::new(1024 * 1024)
            .expect("nonzero correlation byte limit"),
    }
}

fn open_correlation_store(path: &Path, context_name: &str) -> (Store, ContextId) {
    let store = open_fixture_store(StoreConfig {
        db_path: path.to_path_buf(),
        ..StoreConfig::default()
    });
    let context = store
        .create_context(CreateContext {
            name: context_name.into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create correlation Context");
    (store, context.id)
}

fn import_correlation_source(
    store: &Store,
    context_id: ContextId,
    file_stem: &str,
    locator: &str,
    source: &[u8],
    max_events: usize,
) -> context_graph::ImportEventReceipt {
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from(file_stem));
    store
        .import_event_stream(
            &mut adapter,
            &mut Cursor::new(source.to_vec()),
            ImportEventOptions {
                context_id,
                source_locator_ref: locator.into(),
                session_override: None,
                batch_limits: correlation_limits(max_events),
            },
        )
        .expect("import correlation fixture through the Store and Claude adapter")
}

fn correlation_events(
    store: &Store,
    context_id: ContextId,
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
                limit: NonZeroU32::new(16).expect("nonzero correlation read limit"),
            },
        )
        .expect("read correlation events")
}

fn assert_result_author(
    store: &Store,
    context_id: ContextId,
    source_instance_ref: &str,
    interaction_ref: &str,
    expected_author: &str,
) {
    let events = correlation_events(store, context_id, source_instance_ref);
    let result = events
        .iter()
        .find(|event| {
            event.render_role == context_graph::EventRenderRole::ToolResult
                && event.interaction_ref.as_deref() == Some(interaction_ref)
        })
        .expect("linked result event");
    assert_eq!(result.author, expected_author);
}

fn assert_store_correlation_lifecycle() {
    const SESSION: &str = "correlation-session";
    let linked = correlation_source(&[
        tool_request(
            "81000000-0000-4000-8000-000000000001",
            SESSION,
            "linked-operation",
            "Edit",
        ),
        tool_result(
            "81000000-0000-4000-8000-000000000002",
            SESSION,
            "linked-operation",
        ),
    ]);
    let first_record_end = linked
        .iter()
        .position(|byte| *byte == b'\n')
        .map(|offset| offset + 1)
        .expect("first complete correlation record");

    let pending_dir = tempdir().expect("pending correlation Store directory");
    let (pending_store, pending_context) = open_correlation_store(
        &pending_dir.path().join("pending.db"),
        "pending correlation",
    );
    let pending = import_correlation_source(
        &pending_store,
        pending_context,
        "pending",
        "fixture://correlation/pending",
        &linked,
        8,
    );
    assert_eq!(
        pending.batch_count, 1,
        "request and result stayed in one pending batch"
    );
    assert_result_author(
        &pending_store,
        pending_context,
        &pending.source.source_instance_ref,
        "linked-operation",
        "tool:Edit",
    );

    let flushed_dir = tempdir().expect("flushed correlation Store directory");
    let (flushed_store, flushed_context) = open_correlation_store(
        &flushed_dir.path().join("flushed.db"),
        "flushed correlation",
    );
    let flushed = import_correlation_source(
        &flushed_store,
        flushed_context,
        "flushed",
        "fixture://correlation/flushed",
        &linked,
        1,
    );
    assert_eq!(
        flushed.batch_count, 2,
        "the result follows a forced request flush"
    );
    assert_result_author(
        &flushed_store,
        flushed_context,
        &flushed.source.source_instance_ref,
        "linked-operation",
        "tool:Edit",
    );

    let reopened_dir = tempdir().expect("reopened correlation Store directory");
    let reopened_path = reopened_dir.path().join("reopened.db");
    let (reopened_store, reopened_context) =
        open_correlation_store(&reopened_path, "reopened correlation");
    let first = import_correlation_source(
        &reopened_store,
        reopened_context,
        "reopened",
        "fixture://correlation/reopened",
        &linked[..first_record_end],
        8,
    );
    assert_eq!(first.inserted, 1);
    drop(reopened_store);
    let reopened_store = open_fixture_store(StoreConfig {
        db_path: reopened_path,
        ..StoreConfig::default()
    });
    let resumed = import_correlation_source(
        &reopened_store,
        reopened_context,
        "renamed-after-reopen",
        "fixture://correlation/reopened",
        &linked,
        8,
    );
    assert_eq!(resumed.inserted, 1, "resume imports only the linked result");
    assert_result_author(
        &reopened_store,
        reopened_context,
        &resumed.source.source_instance_ref,
        "linked-operation",
        "tool:Edit",
    );

    let union = correlation_source(&[
        tool_request(
            "82000000-0000-4000-8000-000000000001",
            SESSION,
            "union-operation",
            "Bash",
        ),
        tool_request(
            "82000000-0000-4000-8000-000000000002",
            SESSION,
            "union-operation",
            "Edit",
        ),
        tool_result(
            "82000000-0000-4000-8000-000000000003",
            SESSION,
            "union-operation",
        ),
    ]);
    let union_first_end = union
        .iter()
        .position(|byte| *byte == b'\n')
        .map(|offset| offset + 1)
        .expect("first complete union record");
    let union_dir = tempdir().expect("union correlation Store directory");
    let (union_store, union_context) =
        open_correlation_store(&union_dir.path().join("union.db"), "union correlation");
    import_correlation_source(
        &union_store,
        union_context,
        "union",
        "fixture://correlation/union",
        &union[..union_first_end],
        8,
    );
    let union_receipt = import_correlation_source(
        &union_store,
        union_context,
        "union",
        "fixture://correlation/union",
        &union,
        8,
    );
    assert_eq!(union_receipt.inserted, 2);
    assert_result_author(
        &union_store,
        union_context,
        &union_receipt.source.source_instance_ref,
        "union-operation",
        "tool",
    );
}

#[test]
fn mapping_preserves_oracle_facts_one_through_nine_in_order() {
    let records = [1, 2, 3, 4, 5, 6];
    let drafts = assemble_records("compact-rich.jsonl", &records);
    let facts = expected_facts("compact-rich.jsonl", &records);
    assert_facts("compact-rich.jsonl", &drafts[..9], &facts[..9]);
    assert_resolver_outcomes_and_occurrence_local_correlation();
    assert_store_correlation_lifecycle();
}
