//! CLI checks for importing history and reading it after capture.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

const CONTEXT: &str = "amber-context";
const FORBIDDEN_MEDIA: &str = "FORBIDDEN_MEDIA_MARKER_6f31";
const FORBIDDEN_MEDIA_BASE64: &str = "Rk9SQklEREVOX01FRElBX01BUktFUl82ZjMx";
const FORBIDDEN_MEDIA_HEX: &str = "464f5242494444454e5f4d454449415f4d41524b45525f36663331";
const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
const NOMIC_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

struct CommandOutput {
    code: i32,
    stdout: String,
    stderr: String,
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/event_import")
        .join(name)
}

fn run_cg(data_root: &Path, model_cache_root: Option<&Path>, args: &[String]) -> CommandOutput {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cg"));
    command
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("HOME", data_root)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("CG_USE_TEST_LEXICAL_EMBEDDER")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(model_cache_root) = model_cache_root {
        command.env("CG_MODEL_CACHE_ROOT", model_cache_root);
    } else {
        command.env_remove("CG_MODEL_CACHE_ROOT");
    }
    let output = command.output().expect("run the compiled cg binary");
    CommandOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn existing_nomic_cache_root() -> PathBuf {
    let cache_root = cg_catalog::default_cache_root();
    let model_dir = cache_root.join(NOMIC_MODEL_ID);
    for file in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "REVISION",
    ] {
        assert!(
            model_dir.join(file).is_file(),
            "the real Nomic smoke requires the existing pinned artifact at {}; missing {}. \
             No lexical or downloaded embedding fallback is allowed",
            model_dir.display(),
            model_dir.join(file).display(),
        );
    }
    let revision = fs::read_to_string(model_dir.join("REVISION"))
        .expect("read the existing Nomic revision marker");
    assert_eq!(
        revision.trim(),
        NOMIC_REVISION,
        "the real Nomic smoke requires pinned revision {NOMIC_REVISION} at {}",
        model_dir.display(),
    );
    cache_root
}

fn json(output: &CommandOutput, invocation: &str) -> Value {
    serde_json::from_str(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "{invocation} must use cg's default JSON output, not an unstructured substitute: \
             {error}; stdout={} stderr={}",
            output.stdout, output.stderr
        )
    })
}

fn json_at<'a>(value: &'a Value, pointer: &str, description: &str) -> &'a Value {
    value.pointer(pointer).unwrap_or_else(|| {
        panic!("{description} must expose {pointer} in cg's JSON result; actual JSON: {value}")
    })
}

fn json_string_at<'a>(value: &'a Value, pointer: &str, description: &str) -> &'a str {
    json_at(value, pointer, description)
        .as_str()
        .unwrap_or_else(|| {
            panic!("{description} at {pointer} must be a string; actual JSON: {value}")
        })
}

fn json_u64_at(value: &Value, pointer: &str, description: &str) -> u64 {
    json_at(value, pointer, description)
        .as_u64()
        .unwrap_or_else(|| {
            panic!("{description} at {pointer} must be a non-negative count; actual JSON: {value}")
        })
}

fn import_args(source: &Path, search: &str) -> Vec<String> {
    vec![
        "import".to_owned(),
        source.display().to_string(),
        "--context".to_owned(),
        CONTEXT.to_owned(),
        "--search".to_owned(),
        search.to_owned(),
    ]
}

fn status_args() -> Vec<String> {
    vec![
        "status".to_owned(),
        "--context".to_owned(),
        CONTEXT.to_owned(),
    ]
}

fn stage_roundtrip_source(source: &Path, nonce: &str) {
    fs::copy(fixture("roundtrip.jsonl"), source).expect("stage source fixture");
    let mut source_file = fs::OpenOptions::new()
        .append(true)
        .open(source)
        .expect("open staged source for the runtime-varied occurrence");
    writeln!(
        source_file,
        "{{\"type\":\"assistant\",\"uuid\":\"00000000-0000-4000-8000-000000000012\",\"sessionId\":\"synthetic-session-amber\",\"timestamp\":\"2026-09-01T09:10:00Z\",\"message\":{{\"id\":\"runtime-varied-message\",\"role\":\"assistant\",\"content\":\"Runtime-varied retained source fact: {nonce}.\"}}}}"
    )
    .expect("append independently known source content");
}

fn assert_no_forbidden_persisted_bytes(root: &Path) {
    let mut pending = vec![root.to_path_buf()];
    let mut artifacts = 0usize;
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path).unwrap_or_else(|error| {
            panic!(
                "read persisted data-root artifact directory {}: {error}",
                path.display()
            )
        }) {
            let entry = entry.expect("read persisted data-root entry");
            let entry_path = entry.path();
            if entry
                .file_type()
                .expect("inspect data-root entry type")
                .is_dir()
            {
                pending.push(entry_path);
                continue;
            }
            artifacts += 1;
            let bytes = fs::read(&entry_path).unwrap_or_else(|error| {
                panic!(
                    "read persisted data-root artifact {}: {error}",
                    entry_path.display()
                )
            });
            for forbidden in [
                FORBIDDEN_MEDIA.as_bytes(),
                FORBIDDEN_MEDIA_BASE64.as_bytes(),
                FORBIDDEN_MEDIA_HEX.as_bytes(),
            ] {
                assert!(
                    !bytes
                        .windows(forbidden.len())
                        .any(|window| window == forbidden),
                    "persisted artifact {} retains forbidden inline-media bytes or an encoded form",
                    entry_path.display()
                );
            }
        }
    }
    assert!(
        artifacts > 0,
        "the metadata-only media event must create persisted Store artifacts"
    );
}

fn event_id(event: &Value) -> String {
    let id = json_string_at(event, "/id", "a shown event identity");
    assert!(!id.is_empty(), "shown event identities are not empty");
    id.to_owned()
}

fn event_semantics(event: &Value) -> Value {
    for field in [
        "id",
        "seq",
        "author",
        "kind",
        "render_role",
        "interaction_ref",
        "text",
        "payload",
        "anchors",
        "coverage",
        "source_remainder",
    ] {
        assert!(
            event.get(field).is_some(),
            "cg show must expose captured {field} for each linked event: {event}"
        );
    }
    event.clone()
}

fn payload_matches_structure(payload: &Value, expected: &Value) -> bool {
    match (payload, expected) {
        (Value::Object(actual), Value::Object(expected)) => expected.iter().all(|(key, value)| {
            actual
                .get(key)
                .is_some_and(|actual_value| payload_matches_structure(actual_value, value))
        }),
        (Value::Array(actual), Value::Array(expected)) => {
            actual.len() == expected.len()
                && actual
                    .iter()
                    .zip(expected)
                    .all(|(actual_value, expected_value)| {
                        payload_matches_structure(actual_value, expected_value)
                    })
        }
        _ => payload == expected,
    }
}

fn payload_contains(payload: &Value, expected: &Value) -> bool {
    payload_matches_structure(payload, expected)
        || match payload {
            Value::Object(values) => values
                .values()
                .any(|value| payload_contains(value, expected)),
            Value::Array(values) => values.iter().any(|value| payload_contains(value, expected)),
            _ => false,
        }
}

fn event_matches_expected(event: &Value, expected: &Value) -> bool {
    let Some(expected_fields) = expected.as_object() else {
        return false;
    };
    expected_fields.iter().all(|(field, value)| {
        if field == "required_payload_values" {
            let Some(payload) = event.get("payload") else {
                return false;
            };
            value
                .as_array()
                .is_some_and(|values| values.iter().all(|value| payload_contains(payload, value)))
        } else if field == "requires_decoded_newline_and_tab" {
            value == &Value::Bool(true)
                && event
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| {
                        text.contains('\n')
                            && text.contains('\t')
                            && !text.contains("\\\\n")
                            && !text.contains("\\\\t")
                    })
        } else {
            event.get(field) == Some(value)
        }
    })
}

fn assert_tool_event_matcher_cases() {
    let expected = serde_json::json!({
        "kind": "file_edit",
        "render_role": "tool_request",
        "interaction_ref": "tool-edit-1",
        "required_payload_values": [
            {"tool_use_id": "tool-edit-1"},
            {"tool_name": "Edit"},
            {"input": {
                "file_path": "/workspace/notice.md",
                "old_string": "seven days",
                "new_string": "fourteen days",
                "reason": "retain the negotiated notice period"
            }},
            "synthetic-session-amber"
        ],
        "coverage": "complete"
    });
    let event = serde_json::json!({
        "kind": "file_edit",
        "render_role": "tool_request",
        "interaction_ref": "tool-edit-1",
        "coverage": "complete",
        "payload": {
            "content": [{
                "tool_use_id": "tool-edit-1",
                "tool_name": "Edit",
                "input": {
                    "file_path": "/workspace/notice.md",
                    "old_string": "seven days",
                    "new_string": "fourteen days",
                    "reason": "retain the negotiated notice period"
                }
            }],
            "metadata": {
                "session_id": "synthetic-session-amber",
                "retained_extra": "additional metadata"
            }
        }
    });

    assert!(
        event_matches_expected(&event, &expected),
        "a complete nested tool event retains required values while permitting additional metadata"
    );

    let mut missing_session = event.clone();
    missing_session
        .pointer_mut("/payload/metadata")
        .and_then(Value::as_object_mut)
        .expect("matcher negative-control metadata object")
        .remove("session_id");
    assert!(
        !event_matches_expected(&missing_session, &expected),
        "a nested tool event without required session metadata must not match"
    );

    let mut wrong_input = event;
    let wrong_input_value = wrong_input
        .pointer_mut("/payload/content/0/input/new_string")
        .expect("matcher negative-control nested tool value");
    *wrong_input_value = Value::String("seven days".to_owned());
    assert!(
        !event_matches_expected(&wrong_input, &expected),
        "a nested tool event with a wrong required value must not match"
    );
}

#[test]
fn explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt() {
    let workspace = tempfile::tempdir().expect("isolated CLI workspace");
    let data_root = workspace.path().join("data-root");
    let missing_model_cache = workspace.path().join("empty-model-cache");
    let source = workspace.path().join("amber.jsonl");
    stage_roundtrip_source(&source, "defer-only-runtime-sentinel");

    let imported = run_cg(
        &data_root,
        Some(&missing_model_cache),
        &import_args(&source, "defer"),
    );
    assert_eq!(
        imported.code, 0,
        "an explicit Context and --search defer must capture without a model download or hosted \
         call: stdout={} stderr={}",
        imported.stdout, imported.stderr
    );
    let receipt = json(&imported, "cg import --search defer");
    assert_eq!(receipt.get("ok").and_then(Value::as_bool), Some(true));
    assert_eq!(
        json_string_at(&receipt, "/state", "deferred import state"),
        "captured_search_deferred"
    );
    assert_eq!(
        json_string_at(&receipt, "/capture/status", "capture receipt status"),
        "complete"
    );
    assert_eq!(
        json_string_at(&receipt, "/search/status", "deferred search receipt status"),
        "pending"
    );
    assert_eq!(
        json_u64_at(&receipt, "/capture/events/inserted", "captured event count"),
        8
    );
    assert_eq!(
        json_u64_at(
            &receipt,
            "/capture/local_embedding_calls",
            "deferred embedding calls"
        ),
        0
    );
    assert_eq!(
        json_u64_at(
            &receipt,
            "/capture/understanding_calls",
            "deferred understanding calls"
        ),
        0
    );
    assert_eq!(
        json_u64_at(
            &receipt,
            "/capture/hosted_provider_calls",
            "deferred hosted calls"
        ),
        0
    );
    assert!(
        !imported.stdout.contains("defer-only-runtime-sentinel")
            && !imported.stdout.contains(&source.display().to_string()),
        "the receipt is progress metadata, never a source echo or source path: {}",
        imported.stdout
    );

    fs::remove_file(&source)
        .expect("remove the imported source before the fresh-process status read");
    let status = run_cg(&data_root, Some(&missing_model_cache), &status_args());
    assert_eq!(
        status.code, 0,
        "fresh cg status reads durable deferred capture: {}",
        status.stderr
    );
    let status = json(&status, "cg status after deferred import");
    assert_eq!(
        json_u64_at(
            &status,
            "/history/captured_events",
            "durable captured event count"
        ),
        8
    );
    assert_eq!(
        json_string_at(
            &status,
            "/history/search/status",
            "missing-artifact deferred search status"
        ),
        "blocked"
    );
    assert_eq!(
        json_u64_at(
            &status,
            "/history/search/pending_events",
            "durable pending search count"
        ),
        8
    );
}

#[test]
fn search_wait_round_trip_reads_linked_exact_events_from_the_store_with_real_nomic() {
    assert_tool_event_matcher_cases();

    let workspace = tempfile::tempdir().expect("isolated real-Nomic CLI workspace");
    let data_root = workspace.path().join("data-root");
    let source = workspace.path().join("amber.jsonl");
    let model_cache_root = existing_nomic_cache_root();
    let oracle: Value = serde_json::from_slice(
        &fs::read(fixture("roundtrip-source-oracle.json")).expect("read source oracle"),
    )
    .expect("source oracle is JSON");
    let nonce = format!("runtime-sentinel-{}", uuid::Uuid::new_v4());
    stage_roundtrip_source(&source, &nonce);

    for (control, cache_root) in [
        ("missing", workspace.path().join("missing-nomic-cache")),
        ("invalid", workspace.path().join("invalid-nomic-cache")),
    ] {
        if control == "invalid" {
            let invalid_model_dir = cache_root.join(NOMIC_MODEL_ID);
            fs::create_dir_all(&invalid_model_dir)
                .expect("create isolated invalid cache directory");
            fs::write(
                invalid_model_dir.join("REVISION"),
                "not-the-pinned-nomic-revision\n",
            )
            .expect("write invalid cache revision marker");
        }
        let control_source = workspace.path().join(format!("{control}-cache.jsonl"));
        stage_roundtrip_source(
            &control_source,
            &format!("{control}-cache-must-not-retrieve"),
        );
        let control_output = run_cg(
            &workspace.path().join(format!("{control}-cache-data-root")),
            Some(&cache_root),
            &import_args(&control_source, "wait"),
        );
        assert_eq!(
            control_output.code, 1,
            "a {control} isolated cache captures but refuses to fabricate a projection: stdout={} stderr={}",
            control_output.stdout, control_output.stderr
        );
        let control_receipt = json(&control_output, "cg import --search wait cache control");
        assert_eq!(
            json_string_at(&control_receipt, "/state", "cache-control state"),
            "captured_search_behind"
        );
        assert_eq!(
            json_u64_at(
                &control_receipt,
                "/capture/events/inserted",
                "cache-control capture count"
            ),
            8
        );
        assert_eq!(
            json_u64_at(
                &control_receipt,
                "/capture/local_embedding_calls",
                "cache-control embedding calls"
            ),
            0
        );
    }

    let imported = run_cg(
        &data_root,
        Some(&model_cache_root),
        &import_args(&source, "wait"),
    );
    assert_eq!(
        imported.code, 0,
        "--search wait must capture and complete projection with the existing offline Nomic cache: \
         stdout={} stderr={}",
        imported.stdout, imported.stderr
    );
    let import_receipt = json(&imported, "cg import --search wait");
    assert_eq!(
        import_receipt.get("ok").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        json_string_at(&import_receipt, "/state", "wait import state"),
        "complete"
    );
    assert_eq!(
        json_string_at(&import_receipt, "/search/status", "wait search state"),
        "ready"
    );
    assert!(
        json_u64_at(
            &import_receipt,
            "/search/local_embedding_calls",
            "wait embedding calls"
        ) > 0
    );
    assert_eq!(
        json_u64_at(
            &import_receipt,
            "/capture/hosted_provider_calls",
            "wait hosted calls"
        ),
        0
    );

    fs::remove_file(&source).expect("remove the imported source before database reads");
    assert!(
        !source.exists(),
        "find and show must not be able to reopen the original source path"
    );

    let empty_store = run_cg(
        &workspace.path().join("empty-data-root"),
        Some(&model_cache_root),
        &[
            "find".to_owned(),
            nonce.clone(),
            "--context".to_owned(),
            CONTEXT.to_owned(),
        ],
    );
    assert_ne!(
        empty_store.code, 0,
        "an empty data root cannot return a fabricated history hit"
    );
    let empty_store = json(&empty_store, "cg find against an empty data root");
    assert!(
        empty_store
            .get("hits")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
    );

    let mut observed_events = BTreeMap::<String, Value>::new();
    let mut expected_ids = BTreeSet::<String>::new();
    let mut expected_sequences = Vec::new();
    for check in oracle
        .get("checks")
        .and_then(Value::as_array)
        .expect("source oracle checks")
    {
        let query = json_string_at(check, "/query", "oracle query");
        let found = run_cg(
            &data_root,
            Some(&model_cache_root),
            &[
                "find".to_owned(),
                query.to_owned(),
                "--context".to_owned(),
                CONTEXT.to_owned(),
            ],
        );
        assert_eq!(
            found.code, 0,
            "cg find reads projection after source removal: {}",
            found.stderr
        );
        let find_receipt = json(&found, "cg find");
        let hits = json_at(&find_receipt, "/hits", "matching event-history hits")
            .as_array()
            .expect("find hits is an array");
        assert!(
            !hits.is_empty(),
            "the independently authored query returns a stored passage"
        );
        for hit in hits {
            let passage_id = json_string_at(hit, "/passage_id", "find passage handle").to_owned();
            let source_links = json_at(hit, "/source_links", "ordered find source links")
                .as_array()
                .expect("find source_links is an array");
            let linked_ids = source_links
                .iter()
                .map(|link| {
                    json_string_at(link, "/event_id", "find source-link event identity").to_owned()
                })
                .collect::<Vec<_>>();
            assert!(
                !linked_ids.is_empty(),
                "the matching hit names its source events"
            );

            let shown = run_cg(
                &data_root,
                Some(&model_cache_root),
                &[
                    "show".to_owned(),
                    passage_id,
                    "--context".to_owned(),
                    CONTEXT.to_owned(),
                ],
            );
            assert_eq!(
                shown.code, 0,
                "cg show expands the find handle: {}",
                shown.stderr
            );
            let shown = json(&shown, "cg show");
            let events = json_at(&shown, "/events", "ordered shown passage events")
                .as_array()
                .expect("show events is an array");
            let shown_ids = events.iter().map(event_id).collect::<Vec<_>>();
            assert_eq!(
                shown_ids, linked_ids,
                "show expands exactly the ordered events linked by find"
            );
            let shown_sequences = events
                .iter()
                .map(|event| json_u64_at(event, "/seq", "shown event sequence"))
                .collect::<Vec<_>>();
            assert!(
                shown_sequences.windows(2).all(|pair| pair[0] < pair[1]),
                "show returns linked events in source sequence order: {shown_sequences:?}"
            );
            for event in events {
                let event = event_semantics(event);
                observed_events.insert(event_id(&event), event);
            }
        }
        for expected in check
            .get("expected_events")
            .and_then(Value::as_array)
            .expect("oracle events")
        {
            let matching_ids = observed_events
                .iter()
                .filter_map(|(id, event)| {
                    event_matches_expected(event, expected).then_some(id.clone())
                })
                .collect::<Vec<_>>();
            assert_eq!(
                matching_ids.len(),
                1,
                "expected event must appear once through a hit-linked show result; expected={expected} observed={observed_events:?}"
            );
            expected_ids.insert(matching_ids[0].clone());
            expected_sequences.push(json_u64_at(
                &observed_events[&matching_ids[0]],
                "/seq",
                "expected source occurrence sequence",
            ));
        }
    }
    assert!(
        expected_sequences.windows(2).all(|pair| pair[0] < pair[1]),
        "source-derived occurrences retain their order with monotonic sequences: {expected_sequences:?}"
    );
    assert_eq!(
        expected_ids.len(),
        oracle
            .get("event_count")
            .and_then(Value::as_u64)
            .expect("source oracle event count") as usize
    );

    let varied = run_cg(
        &data_root,
        Some(&model_cache_root),
        &[
            "find".to_owned(),
            nonce.clone(),
            "--context".to_owned(),
            CONTEXT.to_owned(),
        ],
    );
    assert_eq!(
        varied.code, 0,
        "runtime-varied source query succeeds: {}",
        varied.stderr
    );
    let varied = json(&varied, "cg find for runtime-varied source content");
    let expected_varied_text = format!("Runtime-varied retained source fact: {nonce}.");
    let mut nonce_shown = false;
    for hit in json_at(&varied, "/hits", "runtime-varied find hits")
        .as_array()
        .expect("runtime-varied hits array")
    {
        let passage_id = json_string_at(hit, "/passage_id", "runtime-varied passage handle");
        let linked_ids = json_at(hit, "/source_links", "runtime-varied source links")
            .as_array()
            .expect("runtime-varied source links array")
            .iter()
            .map(|link| json_string_at(link, "/event_id", "runtime-varied linked event").to_owned())
            .collect::<Vec<_>>();
        let shown = run_cg(
            &data_root,
            Some(&model_cache_root),
            &[
                "show".to_owned(),
                passage_id.to_owned(),
                "--context".to_owned(),
                CONTEXT.to_owned(),
            ],
        );
        assert_eq!(
            shown.code, 0,
            "cg show expands the nonce hit: {}",
            shown.stderr
        );
        let shown = json(&shown, "cg show for runtime-varied source content");
        let events = json_at(&shown, "/events", "runtime-varied shown events")
            .as_array()
            .expect("runtime-varied shown events array");
        assert_eq!(
            events.iter().map(event_id).collect::<Vec<_>>(),
            linked_ids,
            "show expands exactly the events linked by the nonce hit"
        );
        nonce_shown |= events.iter().any(|event| {
            event.get("text").and_then(Value::as_str) == Some(expected_varied_text.as_str())
        });
    }
    assert!(
        nonce_shown,
        "the exact shown event retains the independently staged runtime source fact"
    );
}

#[test]
fn malformed_input_reports_source_partial_at_the_failed_position_without_silently_skipping_it() {
    let workspace = tempfile::tempdir().expect("isolated malformed-input workspace");
    let data_root = workspace.path().join("data-root");
    let source = workspace.path().join("malformed.jsonl");
    fs::copy(fixture("malformed-after-valid.jsonl"), &source).expect("stage malformed source");

    let imported = run_cg(&data_root, None, &import_args(&source, "defer"));
    assert_eq!(
        imported.code, 1,
        "source_partial is a failed import receipt"
    );
    let receipt = json(&imported, "cg import malformed source");
    assert_eq!(
        json_string_at(&receipt, "/state", "malformed import state"),
        "source_partial"
    );
    let failed = json_at(&receipt, "/capture/failed_sources/0", "failed source entry");
    assert_eq!(
        json_u64_at(failed, "/position/line", "failed source line"),
        2
    );
    assert_eq!(
        json_u64_at(failed, "/position/byte", "failed source byte"),
        261
    );
    assert!(!json_string_at(failed, "/reason", "failed source reason").is_empty());
}

#[test]
fn inline_media_is_captured_as_metadata_only_without_persisting_the_forbidden_payload() {
    let workspace = tempfile::tempdir().expect("isolated media-refusal workspace");
    let data_root = workspace.path().join("data-root");
    let missing_model_cache = workspace.path().join("empty-model-cache");
    let source = workspace.path().join("forbidden-media.jsonl");
    fs::copy(fixture("forbidden-media.jsonl"), &source).expect("stage media source");

    let imported = run_cg(
        &data_root,
        Some(&missing_model_cache),
        &import_args(&source, "defer"),
    );
    assert_eq!(
        imported.code, 0,
        "inline media is metadata-only history: {}",
        imported.stderr
    );
    let receipt = json(&imported, "cg import media source");
    assert_eq!(
        json_string_at(&receipt, "/state", "media import state"),
        "captured_search_deferred"
    );
    assert_eq!(
        json_u64_at(&receipt, "/capture/events/inserted", "media event count"),
        1
    );
    assert_eq!(
        json_u64_at(
            &receipt,
            "/capture/coverage/payload_refused",
            "refused-payload coverage count"
        ),
        1
    );
    assert_eq!(
        json_u64_at(
            &receipt,
            "/capture/local_embedding_calls",
            "media defer embedding calls"
        ),
        0
    );
    assert!(!format!("{}{}", imported.stdout, imported.stderr).contains(FORBIDDEN_MEDIA));

    fs::remove_file(&source).expect("remove the media source before the fresh-process status read");
    let status = run_cg(&data_root, Some(&missing_model_cache), &status_args());
    assert_eq!(
        status.code, 0,
        "the metadata-only event creates a durable Store: {}",
        status.stderr
    );
    let status = json(&status, "cg status after media capture");
    assert_eq!(
        json_u64_at(
            &status,
            "/history/captured_events",
            "durable media event count"
        ),
        1
    );
    assert_eq!(
        json_u64_at(
            &status,
            "/history/coverage/payload_refused",
            "durable refused-payload count"
        ),
        1
    );
    assert_eq!(
        json_string_at(
            &status,
            "/history/search/status",
            "media missing-artifact search status"
        ),
        "blocked"
    );
    assert_no_forbidden_persisted_bytes(&data_root);
}
