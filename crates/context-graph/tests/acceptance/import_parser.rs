use context_graph::CgError;
use context_graph::ingest::{ClaudeJsonlParser, ImportParser, parser_for_format};

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
        "User: Where is auth?\nAssistant: Auth is at src/auth.py\n"
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
            "User: Where is auth?\nAssistant: Auth is at src/auth.py\n",
            "gemini-bulk",
        )
    );
}
