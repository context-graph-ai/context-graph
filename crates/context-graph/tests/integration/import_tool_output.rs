//! RED contract tests for tool-output preservation: `cg import` must stop deleting
//! decision-bearing tool output at jsonl parse. Authored and committed BEFORE implementation;
//! implementation commits may not touch this file.
//!
//! Adversarial construction (the rename-fixture lesson): the decisive detail exists ONLY in the
//! tool_result block; the following assistant dialogue references the mechanism by name but never
//! restates the detail — so these tests cannot pass unless tool content actually survives.

use context_graph::ingest::{
    ClaudeJsonlParser, ImportParser, ToolEvent, TranscriptItem, render_import_transcript,
};

/// The decisive detail: present ONLY inside the tool_result. The assistant's follow-up names
/// `SyncServer::new` / "shared hub component" (token overlap = the "assistant then discussed it"
/// signal) but never repeats this phrase.
const DECISIVE_DETAIL: &str = "zero enterprise edits";

fn assistant_text_line(text: &str) -> String {
    serde_json::json!({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}]
        }
    })
    .to_string()
}

fn assistant_tool_use_line(id: &str, name: &str, input: serde_json::Value) -> String {
    serde_json::json!({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": id, "name": name, "input": input}]
        }
    })
    .to_string()
}

fn user_tool_result_line(id: &str, content: &str) -> String {
    serde_json::json!({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": id, "content": content}]
        }
    })
    .to_string()
}

/// A session where the relay-mode placement decision's rationale lives in a plan-file read —
/// the real capture-loss class this test closes (the relay-hub session is the live-smoke
/// counterpart, assertion S2).
fn decision_bearing_session() -> String {
    [
        assistant_text_line(
            "Let me check the sync design plan before deciding where relay mode belongs.",
        ),
        assistant_tool_use_line(
            "toolu_01",
            "Read",
            serde_json::json!({"file_path": "/plans/sync-design.md"}),
        ),
        user_tool_result_line(
            "toolu_01",
            "Relay mode must be enabled inside SyncServer::new, the shared hub component, so the \
             OSS and enterprise binaries both inherit it with zero enterprise edits.\nThe rest of \
             this document covers unrelated formatting conventions.",
        ),
        assistant_text_line(
            "Given the plan places relay on the shared hub component SyncServer::new, I will \
             enable it there.",
        ),
    ]
    .join("\n")
}

/// A large tool result nothing ever discusses: build noise with a unique marker. Selection must
/// drop its content (keeping at most a one-line prose header naming the target).
fn undiscussed_noise_session() -> (String, usize) {
    let noise: String = (0..800)
        .map(|i| format!("noise-marker-31415 irrelevant build output line {i}\n"))
        .collect();
    let raw = [
        assistant_text_line("Kicking off the build to warm the cache."),
        assistant_tool_use_line(
            "toolu_02",
            "Bash",
            serde_json::json!({"command": "cargo build --release 2>&1 | tee build.log"}),
        ),
        user_tool_result_line("toolu_02", &noise),
        assistant_text_line("While that runs, back to the design discussion about chunk sizing."),
    ]
    .join("\n");
    let len = raw.len();
    (raw, len)
}

#[test]
fn import_preserves_decision_bearing_tool_output_as_prose() {
    let parser = ClaudeJsonlParser;
    let transcript = parser
        .parse(decision_bearing_session().as_bytes())
        .expect("session parses");
    assert!(
        transcript.contains(DECISIVE_DETAIL),
        "the decision's rationale lived only in tool output and was deleted at parse; \
         transcript was:\n{transcript}"
    );
    assert!(
        transcript.contains("/plans/sync-design.md"),
        "the kept tool content must say where it came from (prose header naming the target)"
    );
}

#[test]
fn import_renders_kept_tool_content_without_tool_tags() {
    let parser = ClaudeJsonlParser;
    let transcript = parser
        .parse(decision_bearing_session().as_bytes())
        .expect("session parses");
    // Content must be present (otherwise this test would pass vacuously pre-fix)...
    assert!(transcript.contains(DECISIVE_DETAIL));
    // ...and rendered as plain prose: tag-wrapped content would be invisible to retrieval
    // (tier-0's salience pass strips tool tags from the digest the embedding is built from).
    for tag in ["<tool_use", "<tool_result", "tool_use_id"] {
        assert!(
            !transcript.contains(tag),
            "kept tool content must be plain prose, found {tag:?} in:\n{transcript}"
        );
    }
}

#[test]
fn import_selection_is_deterministic_and_bounded() {
    let parser = ClaudeJsonlParser;
    let (raw, raw_len) = undiscussed_noise_session();
    let first = parser.parse(raw.as_bytes()).expect("session parses");
    let second = parser.parse(raw.as_bytes()).expect("session parses");
    assert_eq!(first, second, "selection must be deterministic");
    assert!(
        !first.contains("noise-marker-31415"),
        "undiscussed tool noise must not survive into the transcript"
    );
    assert!(
        first.len() < raw_len / 10,
        "wholesale keeping is infeasible (~10x chunk volume); undiscussed results are dropped"
    );
    assert!(
        first.contains("cargo build --release"),
        "every tool event keeps a one-line prose header naming what was done, even when its \
         result content is dropped; transcript was:\n{first}"
    );
}

#[test]
fn import_selection_policy_is_shared_not_vendor_specific() {
    // The selection + rendering policy is ONE deterministic library rule over vendor-neutral
    // items — callable with no vendor parser at all, so future connectors (Codex/Gemini,
    // Plan-22 formats) reuse it instead of re-inventing it.
    let items = vec![
        TranscriptItem::Dialogue {
            role: "assistant".to_string(),
            text: "Let me check the sync design plan before deciding where relay mode belongs."
                .to_string(),
        },
        TranscriptItem::Tool(ToolEvent {
            tool_name: "Read".to_string(),
            target: "/plans/sync-design.md".to_string(),
            result_text:
                "Relay mode must be enabled inside SyncServer::new, the shared hub component, so \
                 the OSS and enterprise binaries both inherit it with zero enterprise edits."
                    .to_string(),
        }),
        TranscriptItem::Dialogue {
            role: "assistant".to_string(),
            text: "Given the plan places relay on the shared hub component SyncServer::new, I \
                   will enable it there."
                .to_string(),
        },
        TranscriptItem::Tool(ToolEvent {
            tool_name: "Bash".to_string(),
            target: "cargo build --release".to_string(),
            result_text: "noise-marker-31415 nothing downstream ever references this".to_string(),
        }),
    ];
    let transcript = render_import_transcript(&items);
    assert!(
        transcript.contains(DECISIVE_DETAIL),
        "discussed tool content survives"
    );
    assert!(
        !transcript.contains("noise-marker-31415"),
        "undiscussed tool content is dropped"
    );
    assert!(
        transcript.contains("/plans/sync-design.md"),
        "kept event names its target"
    );
    assert!(
        transcript.contains("cargo build --release"),
        "dropped event keeps its header"
    );
    for tag in ["<tool_use", "<tool_result"] {
        assert!(
            !transcript.contains(tag),
            "prose rendering, never tool tags"
        );
    }
}
