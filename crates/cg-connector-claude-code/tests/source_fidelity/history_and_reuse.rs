#[cfg(context_graph_internal_test_support)]
use cg_connector_claude_code::internal_test_support::{
    ConnectorOperationCounts, connector_operation_counts, reset_connector_operation_counts,
};
use cg_connector_claude_code::{
    CONNECTOR_ID, ClaudeAssemblyContext, ClaudeCodeJsonlAdapter, ClaudeContent,
    ClaudeHistoryLocator, ClaudeMessage, ClaudeMessageRole, ClaudeOccurrence,
    ClaudeOccurrenceAssembler, ClaudeOccurrenceMeta, ClaudeOpaqueRecord, ClaudeProviderFact,
    ClaudeProviderObject, ClaudeProviderValue, ClaudeSubagent, CorrelationResolution,
    EventCorrelationResolver, PriorRequestAnchorQuery, decode_jsonl_record, effective_history_root,
    event_adapter_declaration,
};
use context_graph::{
    AppendEventBatch, ContextId, ConversationEvent, CreateContext, EventBatchLimits, EventDraft,
    EventRange, EventStreamSelector, ImportEventOptions, Store, StoreConfig,
};
use std::ffi::OsString;
use std::io::Cursor;
use std::num::NonZeroU32;
use std::path::Path;
use tempfile::tempdir;

use super::{
    assert_json_retained, fixture_bytes, fixture_records, json_contains, open_fixture_store,
};

const PROGRAMMATIC_SESSION: &str = "synthetic-session-programmatic";
const PROGRAMMATIC_SOURCE: &str = "programmatic-source";

struct MissingResolver;

impl EventCorrelationResolver for MissingResolver {
    fn resolve_prior_request_anchor(
        &mut self,
        _query: &PriorRequestAnchorQuery,
    ) -> context_graph::Result<CorrelationResolution> {
        Ok(CorrelationResolution::Missing)
    }
}

fn object(fields: Vec<ClaudeProviderFact>) -> ClaudeProviderObject {
    ClaudeProviderObject::new(fields)
}

fn historical_occurrences() -> Vec<(u64, u64, ClaudeOccurrence)> {
    fixture_records("typed-parity-rich.jsonl")
        .into_iter()
        .map(|(record, record_start_byte, raw)| {
            (
                record,
                record_start_byte,
                decode_jsonl_record(&raw).expect("decode historical parity record"),
            )
        })
        .collect()
}

fn typed_provider_occurrences() -> Vec<(u64, u64, ClaudeOccurrence)> {
    let message_meta = ClaudeOccurrenceMeta::new(
        PROGRAMMATIC_SESSION,
        Some("70000000-0000-4000-8000-000000000001".into()),
        Some("2026-09-07T10:01:00Z".into()),
        None,
    )
    .with_working_directory("/workspace/typed-parity")
    .with_subagent(ClaudeSubagent::new(
        "typed-subagent-river",
        Some("River".into()),
    ))
    .with_envelope_extras(object(vec![ClaudeProviderFact::new(
        "envelope_extension",
        ClaudeProviderValue::object(vec![
            ClaudeProviderFact::new("mode", "delegated"),
            ClaudeProviderFact::new("attempt", 2_u64),
        ]),
    )]));
    let message = ClaudeOccurrence::message(
        message_meta,
        ClaudeMessage::new(
            ClaudeMessageRole::Assistant,
            Some("typed-message-1".into()),
            vec![
                ClaudeContent::Text("programmatic source stays typed".into()),
                ClaudeContent::ToolUse {
                    id: "typed-tool-1".into(),
                    name: "Bash".into(),
                    input: ClaudeProviderValue::object(vec![
                        ClaudeProviderFact::new("command", "printf 'typed parity'"),
                        ClaudeProviderFact::new(
                            "options",
                            ClaudeProviderValue::object(vec![ClaudeProviderFact::new(
                                "timeout_ms",
                                2_500_u64,
                            )]),
                        ),
                    ]),
                },
                ClaudeContent::Opaque(object(vec![
                    ClaudeProviderFact::new("type", "future_delta"),
                    ClaudeProviderFact::new("signature", "typed-signature"),
                    ClaudeProviderFact::new(
                        "value",
                        ClaudeProviderValue::object(vec![ClaudeProviderFact::new(
                            "note",
                            "typed opaque block survives",
                        )]),
                    ),
                ])),
            ],
        ),
    );

    let opaque_meta = ClaudeOccurrenceMeta::new(
        PROGRAMMATIC_SESSION,
        Some("70000000-0000-4000-8000-000000000002".into()),
        Some("bad-typed-time".into()),
        None,
    )
    .with_working_directory("/workspace/typed-parity")
    .with_envelope_extras(object(vec![ClaudeProviderFact::new(
        "envelope_extension",
        ClaudeProviderValue::object(vec![
            ClaudeProviderFact::new("mode", "opaque"),
            ClaudeProviderFact::new("attempt", 3_u64),
        ]),
    )]));
    let opaque = ClaudeOccurrence::opaque(
        opaque_meta,
        ClaudeOpaqueRecord::new(
            Some("telemetry_zeta".into()),
            object(vec![
                ClaudeProviderFact::new("opaque_note", "typed opaque record survives"),
                ClaudeProviderFact::new(
                    "nested",
                    ClaudeProviderValue::object(vec![ClaudeProviderFact::new("answer", 42_u64)]),
                ),
            ]),
        ),
    );

    vec![(1, 0, message), (2, 1, opaque)]
}

fn assemble_occurrences(occurrences: Vec<(u64, u64, ClaudeOccurrence)>) -> Vec<Vec<EventDraft>> {
    let mut assembler = ClaudeOccurrenceAssembler::new();
    let mut resolver = MissingResolver;
    occurrences
        .into_iter()
        .map(|(record, record_start_byte, occurrence)| {
            assembler
                .assemble(
                    occurrence,
                    ClaudeAssemblyContext {
                        session_ref: PROGRAMMATIC_SESSION.into(),
                        source_instance_ref: PROGRAMMATIC_SOURCE.into(),
                        record,
                        record_start_byte,
                    },
                    &mut resolver,
                )
                .expect("assemble canonical occurrence")
        })
        .collect()
}

fn without_transport_byte(position: &serde_json::Value) -> serde_json::Value {
    let mut position = position.clone();
    if let Some(fields) = position.as_object_mut() {
        fields.remove("byte");
    }
    position
}

fn assert_draft_groups_equal(historical: &[Vec<EventDraft>], typed: &[Vec<EventDraft>]) {
    assert_eq!(historical.len(), typed.len(), "same occurrence groups");
    for (historical_group, typed_group) in historical.iter().zip(typed) {
        assert_eq!(
            historical_group.len(),
            typed_group.len(),
            "same item boundary"
        );
        for (historical, typed) in historical_group.iter().zip(typed_group) {
            assert_eq!(historical.session_ref, typed.session_ref);
            assert_eq!(historical.source_event_key, typed.source_event_key);
            assert_eq!(
                without_transport_byte(&historical.source_position),
                without_transport_byte(&typed.source_position),
            );
            assert_eq!(historical.occurred_at, typed.occurred_at);
            assert_eq!(historical.author, typed.author);
            assert_eq!(historical.kind, typed.kind);
            assert_eq!(historical.render_role, typed.render_role);
            assert_eq!(historical.interaction_ref, typed.interaction_ref);
            assert_eq!(historical.text, typed.text);
            assert_eq!(historical.payload, typed.payload);
            assert_eq!(historical.anchors, typed.anchors);
            assert_eq!(historical.source_remainder, typed.source_remainder);
            assert_eq!(historical.coverage, typed.coverage);
        }
    }
}

fn append_groups(
    store: &Store,
    context_id: ContextId,
    source_instance_ref: &str,
    groups: Vec<Vec<EventDraft>>,
) -> Vec<ConversationEvent> {
    groups
        .into_iter()
        .flat_map(|group| {
            store
                .append_event_batch(AppendEventBatch::live(
                    context_id,
                    event_adapter_declaration(),
                    source_instance_ref,
                    group,
                ))
                .expect("append independently assembled group")
                .events
                .into_iter()
                .map(|result| result.event)
        })
        .collect()
}

fn import_historical_source(
    store: &Store,
    context_id: ContextId,
) -> (String, Vec<ConversationEvent>) {
    let mut adapter = ClaudeCodeJsonlAdapter::new(OsString::from("typed-parity-rich"));
    let receipt = store
        .import_event_stream(
            &mut adapter,
            &mut Cursor::new(fixture_bytes("typed-parity-rich.jsonl").to_vec()),
            ImportEventOptions {
                context_id,
                source_locator_ref: "fixture://typed-parity-rich".into(),
                session_override: None,
                batch_limits: EventBatchLimits::default(),
            },
        )
        .expect("import historical parity source through the Claude adapter");
    let events = store
        .list_stream_events(
            EventStreamSelector {
                context_id,
                connector_id: CONNECTOR_ID.into(),
                source_instance_ref: receipt.source.source_instance_ref.clone(),
            },
            EventRange::Count {
                after_seq: None,
                limit: NonZeroU32::new(16).expect("nonzero parity event limit"),
            },
        )
        .expect("read historically imported parity events");
    (receipt.source.source_instance_ref, events)
}

fn assert_persisted_events_equal(historical: &[ConversationEvent], typed: &[ConversationEvent]) {
    assert_eq!(historical.len(), typed.len());
    for (historical, typed) in historical.iter().zip(typed) {
        assert_eq!(historical.connector_id, typed.connector_id);
        assert_eq!(historical.adapter_version, typed.adapter_version);
        assert_eq!(
            historical.anchor_declaration_version,
            typed.anchor_declaration_version
        );
        assert_eq!(historical.source_instance_ref, typed.source_instance_ref);
        assert_eq!(historical.session_ref, typed.session_ref);
        assert_eq!(historical.source_event_key, typed.source_event_key);
        assert_eq!(
            without_transport_byte(&historical.source_position),
            without_transport_byte(&typed.source_position),
        );
        assert_eq!(historical.seq, typed.seq);
        assert_eq!(historical.anchor_count, typed.anchor_count);
        assert_eq!(historical.anchor_set_hash, typed.anchor_set_hash);
        assert_eq!(historical.occurred_at, typed.occurred_at);
        assert_eq!(historical.author, typed.author);
        assert_eq!(historical.kind, typed.kind);
        assert_eq!(historical.render_role, typed.render_role);
        assert_eq!(historical.interaction_ref, typed.interaction_ref);
        assert_eq!(historical.text, typed.text);
        assert_eq!(historical.payload, typed.payload);
        assert_eq!(historical.anchors, typed.anchors);
        assert_eq!(historical.source_remainder, typed.source_remainder);
        assert_eq!(historical.coverage, typed.coverage);
    }
}

fn claude_default_project_key(project: &Path) -> String {
    let source = project.to_str().expect("UTF-8 synthetic project path");
    let encoded: String = source
        .encode_utf16()
        .map(|unit| match unit {
            0x41..=0x5a | 0x61..=0x7a | 0x30..=0x39 => {
                char::from_u32(u32::from(unit)).expect("ASCII project-key character")
            }
            _ => '-',
        })
        .collect();
    if encoded.len() <= 200 {
        return encoded;
    }

    let hash = source.encode_utf16().fold(0_i32, |hash, unit| {
        hash.wrapping_shl(5)
            .wrapping_sub(hash)
            .wrapping_add(i32::from(unit))
    });
    let mut magnitude = i64::from(hash).abs();
    let mut suffix = String::new();
    loop {
        let digit = u8::try_from(magnitude % 36).expect("base-36 digit");
        suffix.push(char::from(if digit < 10 {
            b'0' + digit
        } else {
            b'a' + (digit - 10)
        }));
        magnitude /= 36;
        if magnitude == 0 {
            break;
        }
    }
    let suffix: String = suffix.chars().rev().collect();
    format!("{}-{suffix}", &encoded[..200])
}

fn write_header(path: &Path, cwd: &Path) {
    std::fs::create_dir_all(path.parent().expect("history fixture parent"))
        .expect("history fixture parent");
    std::fs::write(
        path,
        format!(
            "{}\n",
            serde_json::json!({"cwd": cwd.display().to_string()})
        ),
    )
    .expect("history fixture JSONL header");
}

#[test]
fn claude_project_key_uses_the_verified_punctuation_utf16_and_long_suffix_encoding() {
    assert_eq!(
        claude_default_project_key(Path::new("/srv/acme/a.b_caf\u{e9}")),
        "-srv-acme-a-b-caf-"
    );
    assert_eq!(
        claude_default_project_key(Path::new("/srv/acme/a\u{1f600}")),
        "-srv-acme-a--",
        "a non-BMP character occupies two UTF-16 code units in Claude's JavaScript key"
    );
    assert_eq!(
        claude_default_project_key(Path::new(concat!(
            "/srv/acme/",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "a",
            "\u{1f600}"
        ))),
        concat!(
            "-srv-acme-",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "aaaaaaaaaa",
            "-e0jczu"
        )
    );
}

#[test]
fn history_locator_selects_only_matching_immediate_regular_jsonl_in_name_order() {
    let sandbox = tempdir().expect("temporary source root");
    let configured = sandbox.path().join("claude-config");
    let project = sandbox.path().join("project.a_b\u{e9}\u{1f600}");
    let child = project.join("child-project");
    let alias = sandbox.path().join("project-alias");
    std::fs::create_dir_all(&project).expect("project directory");
    std::fs::create_dir_all(&child).expect("child project directory");
    std::fs::create_dir_all(configured.join("projects")).expect("history root");
    std::os::unix::fs::symlink(&project, &alias).expect("project alias");

    let canonical_project = project.canonicalize().expect("canonical project");
    let canonical_child = child.canonicalize().expect("canonical child project");
    let root = effective_history_root(sandbox.path(), Some(&configured));
    assert_eq!(root, configured.join("projects"));
    let project_store = root.join(claude_default_project_key(&canonical_project));
    let child_store = root.join(claude_default_project_key(&canonical_child));

    write_header(&project_store.join("z-selected.jsonl"), &canonical_project);
    write_header(&project_store.join("b-selected.jsonl"), &canonical_project);
    write_header(&project_store.join("a-child-cwd.jsonl"), &canonical_child);
    std::fs::write(project_store.join("c-missing-cwd.jsonl"), "{}\n").expect("missing-cwd fixture");
    write_header(
        &project_store.join("nested").join("not-immediate.jsonl"),
        &canonical_project,
    );
    std::os::unix::fs::symlink(
        project_store.join("b-selected.jsonl"),
        project_store.join("d-linked.jsonl"),
    )
    .expect("linked JSONL fixture");

    std::fs::create_dir_all(project_store.join("b-selected").join("nested"))
        .expect("first selected source helper directory");
    std::fs::create_dir_all(project_store.join("z-selected"))
        .expect("second selected source helper directory");
    std::fs::write(
        project_store.join("b-selected").join("helper-one.jsonl"),
        "{}\n",
    )
    .expect("first regular helper transcript");
    std::fs::write(
        project_store
            .join("b-selected")
            .join("nested")
            .join("helper-two.jsonl"),
        "{}\n",
    )
    .expect("nested regular helper transcript");
    std::os::unix::fs::symlink(
        project_store.join("b-selected").join("helper-one.jsonl"),
        project_store.join("b-selected").join("linked-helper.jsonl"),
    )
    .expect("linked helper transcript");
    std::fs::write(
        project_store.join("z-selected").join("helper-three.jsonl"),
        "{}\n",
    )
    .expect("second selected source helper transcript");
    write_header(
        &child_store.join("a-child-selected.jsonl"),
        &canonical_child,
    );

    let history = ClaudeHistoryLocator::new(root)
        .resolve_project(&project)
        .expect("matching synthetic history resolves");
    assert_eq!(history.canonical_project, canonical_project);
    assert_eq!(
        history
            .sources
            .iter()
            .map(|source| source.file_stem.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        vec!["b-selected", "z-selected"],
        "only immediate regular JSONL whose first complete header identifies this project import"
    );
    assert_eq!(history.helper_transcripts_not_imported, 3);

    let aliased_history = ClaudeHistoryLocator::new(configured.join("projects"))
        .resolve_project(&alias)
        .expect("canonical project alias resolves the same history");
    assert_eq!(aliased_history.canonical_project, canonical_project);
    assert_eq!(
        aliased_history
            .sources
            .iter()
            .map(|source| source.path.clone())
            .collect::<Vec<_>>(),
        history
            .sources
            .iter()
            .map(|source| source.path.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn history_locator_uses_the_verified_long_default_project_key_suffix() {
    let sandbox = tempdir().expect("temporary source root");
    let project = sandbox
        .path()
        .join("long-project")
        .join("a".repeat(80))
        .join("b".repeat(80))
        .join("c".repeat(80));
    std::fs::create_dir_all(&project).expect("long project directory");
    let canonical_project = project.canonicalize().expect("canonical long project");
    let root = sandbox.path().join("claude-config").join("projects");
    let key = claude_default_project_key(&canonical_project);
    assert!(
        key.len() > 200,
        "long keys retain a hash suffix after 200 units"
    );
    write_header(&root.join(&key).join("selected.jsonl"), &canonical_project);

    let history = ClaudeHistoryLocator::new(root)
        .resolve_project(&project)
        .expect("long-key synthetic history resolves");
    assert_eq!(history.canonical_project, canonical_project);
    assert_eq!(history.sources.len(), 1);
    assert_eq!(
        history.sources[0].file_stem.to_string_lossy(),
        "selected",
        "the selected source comes from Claude's suffix-bearing project directory"
    );
}

#[test]
fn typed_occurrences_use_the_same_assembler_then_append_to_fresh_stores() {
    let directory = tempdir().expect("fresh parity stores");
    let historical_store = open_fixture_store(StoreConfig {
        db_path: directory.path().join("historical.db"),
        ..StoreConfig::default()
    });
    let typed_store = open_fixture_store(StoreConfig {
        db_path: directory.path().join("typed.db"),
        ..StoreConfig::default()
    });
    let historical_context = historical_store
        .create_context(CreateContext {
            name: "historical parity".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create historical Context");
    let typed_context = typed_store
        .create_context(CreateContext {
            name: "typed parity".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create typed Context");

    #[cfg(context_graph_internal_test_support)]
    {
        reset_connector_operation_counts();
        let serialization_probe = ClaudeProviderValue::object(vec![ClaudeProviderFact::new(
            "probe",
            "typed-item-json-boundary",
        )]);
        serde_json::to_value(&serialization_probe)
            .expect_err("the observed typed-item JSON serialization boundary refuses a round trip");
        assert_eq!(
            connector_operation_counts().typed_item_json_serializations,
            1,
            "the typed-item serialization observer has a positive control"
        );
        reset_connector_operation_counts();
    }

    let historical_occurrences = historical_occurrences();
    let historical_groups = assemble_occurrences(historical_occurrences.clone());
    #[cfg(context_graph_internal_test_support)]
    reset_connector_operation_counts();
    let observed_history_root = effective_history_root(
        Path::new("/synthetic/home"),
        Some(Path::new("/synthetic/claude-config")),
    );
    assert_eq!(
        observed_history_root,
        Path::new("/synthetic/claude-config/projects")
    );
    let (source_instance_ref, historical_events) =
        import_historical_source(&historical_store, historical_context.id);

    #[cfg(context_graph_internal_test_support)]
    {
        let historical = connector_operation_counts();
        assert!(historical.history_path_resolutions > 0);
        assert!(historical.historical_source_reads > 0);
        assert!(historical.historical_json_decodes > 0);
        assert_eq!(historical.typed_item_json_serializations, 0);
        reset_connector_operation_counts();
    }

    let typed_occurrences = typed_provider_occurrences();
    assert_eq!(historical_occurrences.len(), typed_occurrences.len());
    for (historical, typed) in historical_occurrences.iter().zip(&typed_occurrences) {
        assert_eq!(
            historical.2, typed.2,
            "historical and typed decoders construct the same rich canonical occurrence"
        );
    }

    let typed_groups = assemble_occurrences(typed_occurrences);
    assert_draft_groups_equal(&historical_groups, &typed_groups);
    assert_eq!(historical_groups[0].len(), 3);
    for draft in &historical_groups[0] {
        assert_eq!(
            draft.anchors.get("working_directory"),
            Some(&serde_json::json!("/workspace/typed-parity"))
        );
        assert_json_retained(
            &draft.payload,
            &serde_json::json!("typed-subagent-river"),
            "typed subagent id",
        );
        assert_json_retained(
            &draft.payload,
            &serde_json::json!("River"),
            "typed subagent name",
        );
    }
    assert_json_retained(
        &historical_groups[0][0].source_remainder,
        &serde_json::json!({"envelope_extension": {"mode": "delegated", "attempt": 2}}),
        "record-envelope extras on the first item",
    );
    assert!(
        historical_groups[0][1..]
            .iter()
            .all(|draft| !json_contains(&draft.source_remainder, &serde_json::json!("delegated"))),
        "record-envelope extras are retained once rather than repeated on every item"
    );
    assert_json_retained(
        &historical_groups[0][1].payload,
        &serde_json::json!({
            "command": "printf 'typed parity'",
            "options": {"timeout_ms": 2500}
        }),
        "typed nested tool input",
    );
    assert_json_retained(
        &historical_groups[0][2].source_remainder,
        &serde_json::json!({
            "type": "future_delta",
            "signature": "typed-signature",
            "value": {"note": "typed opaque block survives"}
        }),
        "typed opaque content block",
    );
    assert_eq!(historical_groups[1].len(), 1);
    let opaque_source: serde_json::Value = serde_json::from_slice(
        &fixture_records("typed-parity-rich.jsonl")
            .into_iter()
            .find_map(|(record, _, raw)| (record == 2).then_some(raw))
            .expect("historical opaque parity record"),
    )
    .expect("historical opaque parity object");
    assert_json_retained(
        &historical_groups[1][0].source_remainder,
        &opaque_source,
        "complete typed opaque record and envelope extras",
    );
    assert_json_retained(
        &historical_groups[1][0].source_remainder,
        &serde_json::json!("bad-typed-time"),
        "typed invalid source timestamp",
    );

    let typed_events = append_groups(
        &typed_store,
        typed_context.id,
        &source_instance_ref,
        typed_groups,
    );
    #[cfg(context_graph_internal_test_support)]
    assert_eq!(
        connector_operation_counts(),
        ConnectorOperationCounts::default(),
        "the complete typed assembly and append performs no source read, JSON decode, path resolution, or item JSON serialization; Store persistence is outside the connector observer"
    );
    assert_persisted_events_equal(&historical_events, &typed_events);
}
