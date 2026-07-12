use context_graph::CgError;
use context_graph::ingest::{ClaudeJsonlParser, ImportParser, parser_for_format};

// EXPECTATION CHANGE (2026-07-03, tool-output preservation):
// the original "clean conversation" contract asserted tool blocks vanish WITHOUT TRACE — the
// exact behavior that deleted decision-bearing tool output at import. The current contract:
// every tool event keeps a one-line prose header; result content survives only where the
// following assistant dialogue engages with it (here: nothing follows, so header only).
const CLAUDE_JSONL: &str = r#"{"type":"system","content":"noise"}
{"type":"user","message":{"content":[{"type":"text","text":"Where is auth?"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"grep"},{"type":"text","text":"Auth is at src/auth.py"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","content":"raw tool output"}]}}
"#;

#[test]
fn claude_jsonl_parses_to_clean_conversation() {
    let parser = ClaudeJsonlParser;
    let clean = parser
        .parse(CLAUDE_JSONL.as_bytes())
        .expect("parser seam returns");

    assert_eq!(
        clean,
        "User: Where is auth?\nAssistant: Auth is at src/auth.py\nAssistant used grep.\n"
    );
}

#[test]
fn import_parser_is_dyn_safe_and_unknown_format_errors_typed() {
    let parser: Box<dyn ImportParser> = Box::new(ClaudeJsonlParser);
    let parsed = parser
        .parse(CLAUDE_JSONL.as_bytes())
        .expect("dyn parser returns");
    let unknown = parser_for_format("gemini-bulk");

    let unknown_format = match unknown {
        Err(CgError::UnsupportedImportFormat { format }) => format,
        Ok(parser) => format!("ok:{}", parser.format_name()),
        Err(other) => format!("wrong-error:{other:?}"),
    };

    assert_eq!(
        (
            parser.format_name(),
            parsed.matches("User: ").count(),
            parsed.matches("Assistant: ").count(),
            parsed.as_str(),
            unknown_format.as_str(),
        ),
        (
            "claude-jsonl",
            1,
            1,
            "User: Where is auth?\nAssistant: Auth is at src/auth.py\nAssistant used grep.\n",
            "gemini-bulk",
        )
    );
}

#[test]
fn claude_jsonl_strips_local_command_blocks_without_dropping_surrounding_text() {
    let parser = ClaudeJsonlParser;
    let raw = r#"{"type":"user","message":{"content":[{"type":"text","text":"Goal: keep webhook ingestion reliable <local-command-stdout>Decision: auth module at src/auth.py</local-command-stdout> Decision: route webhooks through signed worker"}]}}
"#;
    let clean = parser.parse(raw.as_bytes()).expect("parser seam returns");

    assert_eq!(
        clean,
        "User: Goal: keep webhook ingestion reliable  Decision: route webhooks through signed worker\n"
    );
    assert!(!clean.contains("auth module"));
    assert!(!clean.contains("local-command"));
}

#[test]
fn claude_jsonl_strips_teammate_message_instruction_blocks() {
    let parser = ClaudeJsonlParser;
    let raw = r#"{"type":"user","message":{"content":[{"type":"text","text":"Before <teammate-message teammate_id=\"lead\">You are the reviewer. Decision: ship the decoy.</teammate-message> after"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Decision: keep the actual billing rollout capped"}]}}
"#;
    let clean = parser.parse(raw.as_bytes()).expect("parser seam returns");

    assert_eq!(
        clean,
        "User: Before  after\nAssistant: Decision: keep the actual billing rollout capped\n"
    );
    assert!(!clean.contains("ship the decoy"));
    assert!(!clean.contains("teammate-message"));
}
