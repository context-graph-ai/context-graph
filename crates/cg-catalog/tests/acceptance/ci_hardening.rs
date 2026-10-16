#[path = "../../../context-graph/tests/common/ci_stages.rs"]
mod ci_stages;
use ci_stages::{
    REQUIRED_PR_JOB_AGGREGATE_RUN, VETTED_DISK_RECLAMATION_LINES, VETTED_JOB_RECEIPT_RUN,
};

use std::fs;
use std::process::Command;

use serde_yaml_ng::{Mapping, Value};

const BANNED_CRATES: &[&str] = &[
    "candle-embeddinggemma",
    "pyo3",
    "cpython",
    "tch",
    "ort",
    "tensorflow",
    "tflite",
    "onnxruntime",
    "libtorch",
    "reqwest",
    "openai",
    "anthropic-sdk",
    "anthropic",
    "cohere",
    "openssl-sys",
    "mkl-sys",
    "cuda",
    "cudnn",
    "tensorrt",
];
const ALLOWED_ADVISORY_IGNORES: &[&str] = &[
    "RUSTSEC-2024-0436",
    "RUSTSEC-2025-0141",
    // atomic-polyfill (unmaintained): embedded-target-only, reachable solely
    // via the dial-by-key sync transport's relay wire codec (iroh-relay ->
    // postcard -> heapless 0.7); never compiled for shipped targets; no safe
    // upgrade until postcard moves to heapless 0.8.
    "RUSTSEC-2023-0089",
    // proc-macro-error (unmaintained): build-time proc-macro helper reached
    // solely via the dial-by-key sync transport's blob plane (iroh-blobs ->
    // bao-tree -> genawaiter -> genawaiter-proc-macro); compiled only on the
    // host at build time, never linked into a shipped artifact; no upgrade path
    // until bao-tree/genawaiter drop it.
    "RUSTSEC-2024-0370",
    "RUSTSEC-2024-0449",
];
const CARGO_AUDIT_ALIASES: &[&str] = &["deny", "expand"];
const COMMAND_LOOKUP_ENV_KEYS: &[&str] = &["PATH", "BASH_ENV", "ENV", "SHELLOPTS", "IFS"];
const AUDIT_TOOL_PATH_MARKERS: &[&str] = &[
    ".cargo",
    ".cargo/bin",
    ".local/bin",
    "CARGO_HOME",
    "/usr/local",
    "/usr/local/bin",
    "/usr",
    "/usr/bin",
    "/bin",
    "RUNNER_TOOL_CACHE",
];
const AUDIT_CAPTURE_PRODUCER_LINES: &[&str] = &[
    "cargo expand --lib -p context-graph | tee \"$RUNNER_TEMP/context-graph-expanded.rs\"",
    "cargo tree -p context-graph -e normal,build | tee \"$RUNNER_TEMP/context-graph-tree.txt\"",
    "cargo tree -p context-graph --edges normal,build | tee \"$RUNNER_TEMP/context-graph-tree.txt\"",
];
const AUDIT_CAPTURE_PATHS: &[&str] = &[
    "$RUNNER_TEMP/context-graph-expanded.rs",
    "$RUNNER_TEMP/context-graph-tree.txt",
];
const COMMAND_LOOKUP_SAFE_ACTIONS: &[&str] = &[
    "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
    "actions/checkout@v4",
    "actions/cache@v4",
    "dtolnay/rust-toolchain@stable",
    "taiki-e/install-action@v2",
];
const OPAQUE_DELEGATION_COMMANDS: &[&str] = &[
    "make", "gmake", "just", "npm", "npx", "pnpm", "yarn", "bun", "go", "gradle", "mvn", "bazel",
    "cmake", "ninja", "tox",
];
const OPAQUE_CARGO_SUBCOMMANDS: &[&str] = &["run", "make", "xtask", "script"];
const AUDIT_CARGO_SUBCOMMANDS: &[&str] = &["deny", "expand", "tree"];
const OPAQUE_INTERPRETER_COMMANDS: &[&str] = &[
    "awk",
    "busybox",
    "cmd",
    "cmd.exe",
    "dash",
    "deno",
    "fish",
    "julia",
    "lua",
    "node",
    "perl",
    "php",
    "powershell",
    "pwsh",
    "ruby",
    "sed",
    "wscript",
    "cscript",
    "zsh",
];
const SAFE_GITHUB_HOSTED_RUNNERS: &[&str] = &["ubuntu-latest", "macos-latest", "windows-latest"];
const CONTEXT_GRAPH_WORKING_DIRECTORY: &str = "workspace/context-graph";
const CONTEXT_GRAPH_CHECKOUT_PATH: &str = "workspace/context-graph";
const CONTEXT_GRAPH_CHECKOUT_FETCH_DEPTH: i64 = 0;
const MODEL_CACHE_ROOT_REFERENCE: &str = "${{ env.CG_MODEL_CACHE_ROOT }}";
const AUDIT_TOOL_INSTALL_SPECS: &[&str] = &[
    "cargo-deny@0.20.2",
    "cargo-expand@1.0.124",
    "cargo-nextest@0.9.140",
];
const VETTED_AUDIT_RUN_LINES: &[&str] = &[
    "require_file_without_match() {",
    "local pattern=\"$1\"",
    "local file=\"$2\"",
    "local grep_status=0",
    "grep -q -E \"$pattern\" \"$file\" || grep_status=$?",
    "case \"$grep_status\" in",
    "0)",
    "echo \"forbidden dependency marker found in $file\" >&2",
    "return 1",
    ";;",
    "1) return 0 ;;",
    "*)",
    "echo \"dependency-marker scan failed for $file\" >&2",
    "return \"$grep_status\"",
    ";;",
    "esac",
    "}",
    "cargo deny --config deny.toml check",
    "cargo expand --lib -p context-graph | tee \"$RUNNER_TEMP/context-graph-expanded.rs\"",
    "test -s \"$RUNNER_TEMP/context-graph-expanded.rs\"",
    "require_file_without_match 'embeddinggemma|candle_embeddinggemma' \"$RUNNER_TEMP/context-graph-expanded.rs\"",
    "cargo tree -p context-graph -e normal,build | tee \"$RUNNER_TEMP/context-graph-tree.txt\"",
    "test -s \"$RUNNER_TEMP/context-graph-tree.txt\"",
    "require_file_without_match '(^|[[:space:]])candle-embeddinggemma([[:space:]]|$)' \"$RUNNER_TEMP/context-graph-tree.txt\"",
];
const EXACT_REVISION_VERIFIER_LINES: &[&str] = &[
    "context_graph_sha=\"$(git rev-parse HEAD)\"",
    "contextdb_sha=\"$(grep -o -m 1 -E 'contextdb[?]rev=[0-9a-f]{40}' Cargo.lock)\"",
    "contextdb_sha=\"${contextdb_sha##*=}\"",
    "if [[ ! \"$context_graph_sha\" =~ ^[0-9a-f]{40}$ ]] || [[ ! \"$contextdb_sha\" =~ ^[0-9a-f]{40}$ ]]; then",
    "echo \"Context Graph checkout and locked ContextDB revision must be immutable 40-character commit SHAs\" >&2",
    "exit 1",
    "fi",
    "if [[ \"$context_graph_sha\" != \"$TESTED_SHA\" ]]; then",
    "echo \"checked-out Context Graph SHA does not match selected tested SHA\" >&2",
    "exit 1",
    "fi",
    "if [[ -n \"$REQUESTED_CONTEXTDB_SHA\" ]] && [[ \"$contextdb_sha\" != \"$REQUESTED_CONTEXTDB_SHA\" ]]; then",
    "echo \"locked ContextDB SHA does not match contextdb_sha\" >&2",
    "exit 1",
    "fi",
    "printf 'Context Graph SHA: %s\\n' \"$context_graph_sha\"",
    "printf 'ContextDB SHA: %s\\n' \"$contextdb_sha\"",
    "printf 'Base SHA: %s\\nHead SHA: %s\\nEvent SHA: %s\\n' \"$BASE_SHA\" \"$HEAD_SHA\" \"$EVENT_SHA\"",
];
const EXACT_CONTEXTDB_SHA_INPUT_VALIDATION_LINES: &[&str] = &[
    "if [[ ! \"$CONTEXTDB_SHA\" =~ ^[0-9a-f]{40}$ ]]; then",
    "echo \"contextdb_sha must be an exact 40-character lowercase hexadecimal commit SHA\" >&2",
    "exit 1",
    "fi",
];
const VETTED_COLD_MISS_MODEL_SEED_LINES: &[&str] = &[
    "rustup run 1.94.1 cargo run -p context-graph-cli --bin cg -- model fetch --defaults",
    "rustup run 1.94.1 cargo test -p context-graph --test nomic_document_preparer_adds_exactly_one_system_task_prefix",
];
const VETTED_OFFLINE_REAL_MODEL_PROOF_LINES: &[&str] = &[
    "rustup run 1.94.1 cargo test -p context-graph \\",
    "--test nomic_document_preparer_adds_exactly_one_system_task_prefix \\",
    "--test nomic_query_preparer_adds_exactly_one_system_task_prefix \\",
    "--test event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged",
];
const VETTED_DOCUMENTATION_CLAIM_VALIDATION_LINES: &[&str] = &[
    "base=\"${DOCUMENTATION_BASE_SHA:-}\"",
    "if [[ -z \"$base\" ]] || [[ \"$base\" =~ ^0+$ ]] || ! git rev-parse --verify --quiet \"HEAD^\" > /dev/null || ! git rev-parse --verify --quiet \"${base}^{commit}\" > /dev/null; then",
    "echo \"initial state, no comparison: auditing all documentation\"",
    "python3 scripts/validate-doc-coverage.py --all",
    "else",
    "printf 'comparing the touched doc files against %s\\n' \"$base\"",
    "python3 scripts/validate-doc-coverage.py --base \"$base\"",
    "fi",
    "bash scripts/check-doc-validator-negative-control.sh",
];
const VETTED_GATE_WALL_CLOCK_BASELINE_DIRECTION_LINES: &[&str] = &[
    "base=\"${DOCUMENTATION_BASE_SHA:-}\"",
    "if [[ -z \"$base\" ]] || [[ \"$base\" =~ ^0+$ ]] || ! git rev-parse --verify --quiet \"HEAD^\" > /dev/null || ! git rev-parse --verify --quiet \"${base}^{commit}\" > /dev/null; then",
    "echo \"initial state, no comparison: gate wall-clock baseline\"",
    "else",
    "printf 'comparing the gate wall-clock baseline against %s\\n' \"$base\"",
    "python3 scripts/check-gate-wall-clock.py --base \"$base\"",
    "fi",
];

fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn read_workflow_yaml() -> Value {
    let workflow = fs::read_to_string(workspace_root().join(".github/workflows/ci.yml"))
        .expect("workflow readable");
    let mut yaml = serde_yaml_ng::from_str(&workflow).expect("workflow yaml");
    yaml = ci_stages::platform_coverage(
        yaml,
        serde_yaml_ng::from_str(
            &std::fs::read_to_string(workspace_root().join(".github/workflows/macos-advisory.yml"))
                .expect("advisory readable"),
        )
        .expect("advisory yaml"),
    );
    ci_stages::effective(
        &mut yaml,
        &fs::read_to_string(workspace_root().join("scripts/gate.sh")).expect("shared stages"),
    );
    yaml
}

fn assert_no_cargo_audit_aliases() {
    let aliases = cargo_config_audit_aliases();
    assert!(
        aliases.is_empty(),
        "cargo config aliases can shadow cargo-deny/cargo-expand while CI text stays exact; remove audit-tool aliases: {aliases:?}"
    );
}

fn cargo_config_audit_aliases() -> Vec<String> {
    [".cargo/config.toml", ".cargo/config"]
        .into_iter()
        .flat_map(|relative| {
            let path = workspace_root().join(relative);
            let Ok(contents) = fs::read_to_string(&path) else {
                return Vec::new();
            };
            let mut in_alias_table = false;
            let mut multiline_string = None;
            contents
                .lines()
                .filter_map(|raw_line| {
                    let line = strip_unquoted_toml_comment(raw_line, &mut multiline_string);
                    let line = line.trim();
                    if line.is_empty() {
                        return None;
                    }
                    if line.starts_with('[') {
                        in_alias_table = toml_table_name(line) == Some("alias".to_string());
                        return None;
                    }
                    cargo_audit_alias_on_line(line, in_alias_table)
                        .map(|alias| format!("{} defines cargo alias `{alias}`", path.display()))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn cargo_audit_alias_on_line(line: &str, in_alias_table: bool) -> Option<&'static str> {
    let (raw_key, value) = line.split_once('=')?;
    let key_parts = toml_key_parts(raw_key);
    CARGO_AUDIT_ALIASES.iter().copied().find(|alias| {
        (in_alias_table && key_parts.as_slice() == [*alias])
            || key_parts.as_slice() == ["alias", *alias]
            || (key_parts.as_slice() == ["alias"] && toml_inline_table_mentions_key(value, alias))
    })
}

fn toml_inline_table_mentions_key(value: &str, key: &str) -> bool {
    value
        .split([',', '{', '}'])
        .filter_map(|entry| entry.split_once('='))
        .any(|(raw_key, _)| toml_key_parts(raw_key).as_slice() == [key])
}

fn toml_table_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?.trim();
    let inner = inner.strip_prefix('[').unwrap_or(inner);
    let inner = inner.strip_suffix(']').unwrap_or(inner);
    Some(toml_key_parts(inner).join("."))
}

fn toml_key_parts(key: &str) -> Vec<String> {
    key.split('.')
        .map(|part| toml_key_part(part.trim()))
        .filter(|part| !part.is_empty())
        .collect()
}

fn toml_key_part(part: &str) -> String {
    let stripped = part.trim_matches('"').trim_matches('\'');
    decode_toml_key_escapes(stripped)
}

fn decode_toml_key_escapes(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut decoded = String::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '\\' {
            decoded.push(chars[index]);
            index += 1;
            continue;
        }
        if index + 1 >= chars.len() {
            decoded.push('\\');
            break;
        }
        match chars[index + 1] {
            'u' if index + 5 < chars.len() => {
                let hex: String = chars[index + 2..index + 6].iter().collect();
                if let Ok(codepoint) = u32::from_str_radix(&hex, 16)
                    && let Some(ch) = char::from_u32(codepoint)
                {
                    decoded.push(ch);
                    index += 6;
                    continue;
                }
                decoded.push('\\');
                index += 1;
            }
            'U' if index + 9 < chars.len() => {
                let hex: String = chars[index + 2..index + 10].iter().collect();
                if let Ok(codepoint) = u32::from_str_radix(&hex, 16)
                    && let Some(ch) = char::from_u32(codepoint)
                {
                    decoded.push(ch);
                    index += 10;
                    continue;
                }
                decoded.push('\\');
                index += 1;
            }
            escaped => {
                decoded.push(escaped);
                index += 2;
            }
        }
    }
    decoded
}

#[derive(Clone, Copy)]
enum TomlMultilineString {
    Basic,
    Literal,
}

impl TomlMultilineString {
    fn delimiter(self) -> &'static str {
        match self {
            TomlMultilineString::Basic => "\"\"\"",
            TomlMultilineString::Literal => "'''",
        }
    }
}

fn strip_unquoted_toml_comment(line: &str, multiline: &mut Option<TomlMultilineString>) -> String {
    let mut code = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut index = 0;
    while index < line.len() {
        if let Some(kind) = *multiline {
            let delimiter = kind.delimiter();
            let delimiter_closes = line[index..].starts_with(delimiter)
                && (!matches!(kind, TomlMultilineString::Basic)
                    || !toml_basic_delimiter_escaped(line, index));
            if delimiter_closes {
                *multiline = None;
                index += delimiter.len();
                continue;
            }
            let ch = line[index..].chars().next().expect("char boundary");
            index += ch.len_utf8();
            continue;
        }
        if !in_single && !in_double && line[index..].starts_with("\"\"\"") {
            *multiline = Some(TomlMultilineString::Basic);
            index += 3;
            continue;
        }
        if !in_single && !in_double && line[index..].starts_with("'''") {
            *multiline = Some(TomlMultilineString::Literal);
            index += 3;
            continue;
        }
        let ch = line[index..].chars().next().expect("char boundary");
        if escaped {
            code.push(ch);
            escaped = false;
            index += ch.len_utf8();
            continue;
        }
        if in_double && ch == '\\' {
            code.push(ch);
            escaped = true;
            index += ch.len_utf8();
            continue;
        }
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                code.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                code.push(ch);
            }
            '#' if !in_single && !in_double => {
                break;
            }
            _ => {
                code.push(ch);
            }
        }
        index += ch.len_utf8();
    }
    code
}

fn toml_basic_delimiter_escaped(line: &str, delimiter_index: usize) -> bool {
    let mut backslashes = 0;
    let mut cursor = delimiter_index;
    while cursor > 0 {
        let Some(previous) = line[..cursor].chars().next_back() else {
            break;
        };
        if previous != '\\' {
            break;
        }
        backslashes += 1;
        cursor -= previous.len_utf8();
    }
    backslashes % 2 == 1
}

fn value_mapping_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    let key = Value::String(key.to_string());
    value.as_mapping()?.get(&key)
}

fn mapping_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    let key = Value::String(key.to_string());
    mapping.get(&key)
}

fn yaml_string(value: &Value) -> Option<&str> {
    value.as_str()
}

fn yaml_explicit_false(value: &Value) -> bool {
    match value {
        Value::Bool(value) => !value,
        Value::String(value) => value.trim().eq_ignore_ascii_case("false"),
        _ => false,
    }
}

fn yaml_explicit_integer(value: &Value, expected: i64) -> bool {
    match value {
        Value::Number(value) => value.as_i64() == Some(expected),
        Value::String(value) => value.trim().parse::<i64>().ok() == Some(expected),
        _ => false,
    }
}

fn yaml_field_contains(value: &Value, needle: &str) -> bool {
    match value {
        Value::String(value) => value.contains(needle),
        Value::Sequence(values) => values
            .iter()
            .any(|value| yaml_field_contains(value, needle)),
        Value::Mapping(values) => values
            .values()
            .any(|value| yaml_field_contains(value, needle)),
        _ => false,
    }
}

fn active_shell_line(line: &str) -> Option<&str> {
    let line = strip_unquoted_comment(line).trim();
    if line.is_empty() { None } else { Some(line) }
}

fn heredoc_delimiter(line: &str) -> Option<String> {
    let line = strip_unquoted_comment(line).trim();
    let marker = find_unquoted(line, "<<")?;
    let after_marker = line[marker + 2..].trim_start();
    let after_marker = after_marker.strip_prefix('-').unwrap_or(after_marker);
    let token = after_marker.split_whitespace().next()?;
    let token = token
        .trim_matches('\'')
        .trim_matches('"')
        .trim_matches('\'')
        .trim_matches('"');
    (!token.is_empty()).then(|| token.to_string())
}

fn active_run_lines(run: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut heredoc_end = None::<String>;
    for raw_line in run.lines() {
        let trimmed = raw_line.trim();
        if heredoc_end.as_deref() == Some(trimmed) {
            heredoc_end = None;
            continue;
        }
        if heredoc_end.is_some() {
            continue;
        }
        if let Some(line) = active_shell_line(raw_line) {
            if let Some(delimiter) = heredoc_delimiter(line) {
                heredoc_end = Some(delimiter);
            }
            lines.push(line);
        }
    }
    lines
}

fn all_run_shell_lines(run: &str) -> Vec<&str> {
    run.lines().filter_map(active_shell_line).collect()
}

fn failure_neutered(line: &str) -> bool {
    line.contains("||")
        || line.contains("; true")
        || line.contains("&& true")
        || line.contains("; :")
        || line.contains("&& :")
        || line.contains("; exit 0")
        || line.contains("&& exit 0")
        || line_contains_unquoted(line, "| true")
        || line_contains_unquoted(line, "| false")
        || line_contains_unquoted(line, "| :")
}

fn set_command_masks_failures(segment: &str) -> bool {
    let words = segment
        .split_whitespace()
        .map(|word| word.replace(['\'', '"'], "").replace('\\', ""))
        .collect::<Vec<_>>();
    let Some(set_index) = first_set_word_index(&words) else {
        return false;
    };
    let words = &words[set_index + 1..];
    words
        .iter()
        .any(|word| word.starts_with('+') && word.chars().skip(1).any(|option| option == 'e'))
        || words
            .windows(2)
            .any(|window| window == ["+o", "pipefail"] || window == ["+o", "errexit"])
}

fn first_set_word_index(words: &[String]) -> Option<usize> {
    let mut index = 0;
    let mut after_time = false;
    while let Some(word) = words.get(index) {
        if word == "!" || word == "time" {
            after_time = word == "time";
            index += 1;
            continue;
        }
        if after_time && word.starts_with('-') {
            index += 1;
            continue;
        }
        return (word == "set").then_some(index);
    }
    None
}

fn run_block_masks_failures(run: &str) -> bool {
    active_run_lines(run).into_iter().any(|line| {
        failure_neutered(line)
            || shell_line_uses_unmodeled_control_flow(line)
            || line.split(';').any(|segment| {
                let segment = segment.trim();
                set_command_masks_failures(segment)
                    || segment == "true"
                    || segment == ":"
                    || segment == "exit 0"
            })
    })
}

fn line_contains_unquoted(line: &str, needle: &str) -> bool {
    find_unquoted(line, needle).is_some()
}

fn line_contains_shell_word(line: &str, word: &str) -> bool {
    let line = line.replace(['\'', '"'], "");
    line.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.'))
        .any(|token| token == word)
}

fn first_shell_word(line: &str) -> String {
    line.split_whitespace()
        .next()
        .unwrap_or("")
        .replace(['\'', '"'], "")
}

fn shell_command_words_use_expansion_escape_or_glob(line: &str) -> bool {
    shell_command_segments(line)
        .into_iter()
        .filter_map(first_command_word_in_segment)
        .any(command_word_uses_expansion_escape_or_glob)
}

fn shell_command_segments(line: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (index, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_double && ch == '\\' {
            escaped = true;
            continue;
        }
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                continue;
            }
            '"' if !in_single => {
                in_double = !in_double;
                continue;
            }
            ';' | '|' | '&' if !in_single && !in_double => {
                segments.push(&line[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    segments.push(&line[start..]);
    segments
}

fn first_command_word_in_segment(segment: &str) -> Option<&str> {
    let mut rest = segment.trim_start();
    let mut after_time = false;
    loop {
        let word = rest.split_whitespace().next()?;
        let normalized = word.replace(['\'', '"'], "");
        if normalized == "!" || normalized == "time" {
            rest = rest[word.len()..].trim_start();
            after_time = normalized == "time";
            continue;
        }
        if after_time && normalized.starts_with('-') {
            rest = rest[word.len()..].trim_start();
            continue;
        }
        return Some(word);
    }
}

fn command_word_uses_expansion_escape_or_glob(word: &str) -> bool {
    word.contains('$')
        || word.contains('\\')
        || word.contains('*')
        || word.contains('?')
        || word.contains('[')
        || word.contains(']')
}

fn strip_unquoted_comment(line: &str) -> &str {
    find_unquoted_comment_start(line)
        .map(|index| &line[..index])
        .unwrap_or(line)
}

fn find_unquoted_comment_start(line: &str) -> Option<usize> {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (index, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_double && ch == '\\' {
            escaped = true;
            continue;
        }
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                continue;
            }
            '"' if !in_single => {
                in_double = !in_double;
                continue;
            }
            '#' if !in_single && !in_double && starts_shell_word(line, index) => {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

fn starts_shell_word(line: &str, index: usize) -> bool {
    match line[..index].chars().next_back() {
        None => true,
        Some(ch) => ch.is_whitespace() || matches!(ch, ';' | '&' | '|' | '(' | ')' | '<' | '>'),
    }
}

fn find_unquoted(line: &str, needle: &str) -> Option<usize> {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (index, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_double && ch == '\\' {
            escaped = true;
            continue;
        }
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                continue;
            }
            '"' if !in_single => {
                in_double = !in_double;
                continue;
            }
            _ => {}
        }
        if !in_single && !in_double && line[index..].starts_with(needle) {
            return Some(index);
        }
    }
    None
}

fn shell_line_uses_unmodeled_control_flow(line: &str) -> bool {
    let first_word = first_shell_word(line);
    line_contains_unquoted(line, "<<")
        || line_contains_unquoted(line, ">")
        || line_contains_unquoted(line, "<")
        || line_contains_unquoted(line, "&")
        || line_contains_unquoted(line, "{")
        || line_contains_unquoted(line, "}")
        || line_contains_unquoted(line, "(")
        || line_contains_unquoted(line, ")")
        || line_contains_unquoted(line, "&&")
        || line_contains_unquoted(line, "$(")
        || line.trim_end().ends_with('\\')
        || line
            .split_whitespace()
            .any(|word| word.replace(['\'', '"'], "") == "trap")
        || line_mutates_command_lookup(line)
        || line_uses_unexpected_audit_cargo_subcommand(line)
        || line.starts_with("function ")
        || line.contains("() {")
        || line.contains("(){")
        || matches!(
            first_word.as_str(),
            "if" | "then"
                | "else"
                | "elif"
                | "fi"
                | "for"
                | "while"
                | "until"
                | "do"
                | "done"
                | "case"
                | "esac"
                | "select"
                | "trap"
                | "alias"
                | "unalias"
                | "shopt"
                | "command"
                | "builtin"
                | "export"
                | "env"
                | "source"
                | "."
                | "eval"
                | "hash"
                | "enable"
                | "declare"
                | "typeset"
                | "readonly"
        )
}

fn line_mutates_command_lookup(line: &str) -> bool {
    let quote_stripped = line.replace(['\'', '"'], "");
    line.contains("${!")
        || line.contains("${")
        || line.contains("$'")
        || line.contains("$\"")
        || line.contains("$[")
        || line.contains("$(")
        || line.contains('`')
        || shell_command_words_use_expansion_escape_or_glob(line)
        || line_contains_shell_word(line, "printenv")
        || ["sh", "bash", "xargs"]
            .iter()
            .any(|command| line_contains_shell_word(line, command))
        || line_uses_interpreter_delegation(line)
        || line_assigns_shell_variable(line)
        || line_printf_assigns_command_lookup(line)
        || line_uses_unsafe_tee(line)
        || line_touches_audit_capture_unsafely(line)
        || line_writes_audit_tool_path(line)
        || line_invokes_path_executable(line)
        || line_invokes_cargo_install(line)
        || line_uses_opaque_delegation(line)
        || quote_stripped.contains("GITHUB_PATH")
        || quote_stripped.contains("GITHUB_ENV")
        // GitHub file-command names are too easy to split-spell and resolve
        // indirectly, so this audit workflow rejects command substitution
        // and executable GITHUB_* use. Env introspection and audit-tool
        // shadow writes are rejected for the same command-lookup reason.
        || quote_stripped.contains("GITHUB_")
        || line
            .split(|ch: char| ch == ';' || ch.is_whitespace())
            .any(|segment| {
                let segment = segment.trim_start();
                COMMAND_LOOKUP_ENV_KEYS.iter().any(|key| {
                    segment.starts_with(&format!("{key}="))
                        || segment.starts_with(&format!("{key}+="))
                })
            })
}

fn line_writes_audit_tool_path(line: &str) -> bool {
    let quote_stripped = line.replace(['\'', '"'], "").replace('\\', "");
    AUDIT_TOOL_PATH_MARKERS
        .iter()
        .any(|marker| quote_stripped.contains(marker))
}

fn line_invokes_path_executable(line: &str) -> bool {
    shell_command_segments(line)
        .into_iter()
        .filter_map(first_command_word_in_segment)
        .map(|word| word.replace(['\'', '"'], "").replace('\\', ""))
        .any(|word| word.contains('/'))
}

fn line_invokes_cargo_install(line: &str) -> bool {
    shell_command_segments(line).into_iter().any(|segment| {
        let words = shell_segment_words(segment);
        let Some(cargo_index) = first_cargo_word_index(&words) else {
            return false;
        };
        cargo_words_after_cargo_use_expansion_escape_or_glob(&words, cargo_index)
            || cargo_subcommand_after_cargo(&words, cargo_index) == Some("install")
    })
}

fn line_uses_opaque_delegation(line: &str) -> bool {
    shell_command_segments(line).into_iter().any(|segment| {
        let words = shell_segment_words(segment);
        let Some(first_word) = first_command_word_in_segment(segment) else {
            return false;
        };
        let first_word = first_word.replace(['\'', '"'], "").replace('\\', "");
        if OPAQUE_DELEGATION_COMMANDS.contains(&first_word.as_str()) {
            return true;
        }
        let Some(cargo_index) = first_cargo_word_index(&words) else {
            return false;
        };
        cargo_words_after_cargo_use_expansion_escape_or_glob(&words, cargo_index)
            || cargo_subcommand_after_cargo(&words, cargo_index)
                .is_some_and(|word| OPAQUE_CARGO_SUBCOMMANDS.contains(&word))
    })
}

fn line_uses_unexpected_audit_cargo_subcommand(line: &str) -> bool {
    shell_command_segments(line).into_iter().any(|segment| {
        let words = shell_segment_words(segment);
        let Some(cargo_index) = first_cargo_word_index(&words) else {
            return false;
        };
        cargo_words_after_cargo_use_expansion_escape_or_glob(&words, cargo_index)
            || !cargo_subcommand_after_cargo(&words, cargo_index)
                .is_some_and(|word| AUDIT_CARGO_SUBCOMMANDS.contains(&word))
    })
}

fn shell_segment_words(segment: &str) -> Vec<String> {
    segment
        .split_whitespace()
        .map(|word| word.replace(['\'', '"'], "").replace('\\', ""))
        .collect()
}

fn first_cargo_word_index(words: &[String]) -> Option<usize> {
    let mut index = 0;
    let mut after_time = false;
    while let Some(word) = words.get(index) {
        if word == "!" || word == "time" {
            after_time = word == "time";
            index += 1;
            continue;
        }
        if after_time && word.starts_with('-') {
            index += 1;
            continue;
        }
        return (word == "cargo").then_some(index);
    }
    None
}

fn cargo_subcommand_after_cargo(words: &[String], cargo_index: usize) -> Option<&str> {
    words
        .iter()
        .skip(cargo_index + 1)
        .find(|word| !word.starts_with('+') && !word.starts_with('-'))
        .map(String::as_str)
}

fn cargo_words_after_cargo_use_expansion_escape_or_glob(
    words: &[String],
    cargo_index: usize,
) -> bool {
    words
        .iter()
        .skip(cargo_index + 1)
        .any(|word| command_word_uses_expansion_escape_or_glob(word))
}

fn line_uses_interpreter_delegation(line: &str) -> bool {
    let line = line.replace(['\'', '"'], "");
    line.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.'))
        .any(|token| {
            let token = token.to_ascii_lowercase();
            OPAQUE_INTERPRETER_COMMANDS.contains(&token.as_str())
                || token == "python"
                || token.starts_with("python2")
                || token.starts_with("python3")
                || token == "rscript"
        })
}

fn line_touches_audit_capture_unsafely(line: &str) -> bool {
    let quote_stripped = line.replace(['\'', '"'], "");
    let mentions_capture = AUDIT_CAPTURE_PATHS
        .iter()
        .any(|path| quote_stripped.contains(path));
    mentions_capture && !line_uses_audit_capture_safely(line)
}

fn line_uses_audit_capture_safely(line: &str) -> bool {
    AUDIT_CAPTURE_PRODUCER_LINES.contains(&line)
        || line == "test -s \"$RUNNER_TEMP/context-graph-expanded.rs\""
        || line == "test -s \"$RUNNER_TEMP/context-graph-tree.txt\""
        || (line.starts_with("rg --files-without-match ")
            && AUDIT_CAPTURE_PATHS
                .iter()
                .any(|path| line.ends_with(&format!("\"{path}\""))))
}

fn line_assigns_shell_variable(line: &str) -> bool {
    line.split(|ch: char| ch == ';' || ch.is_whitespace())
        .map(str::trim_start)
        .any(|segment| {
            let Some((name, _value)) = segment.split_once('=') else {
                return false;
            };
            let name = name.trim_end_matches('+');
            let mut chars = name.chars();
            chars
                .next()
                .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
                && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
        })
}

fn line_printf_assigns_command_lookup(line: &str) -> bool {
    let quote_stripped = line.replace(['\'', '"'], "").replace('\\', "");
    quote_stripped.split(';').any(|segment| {
        let words: Vec<_> = segment.split_whitespace().collect();
        let Some(printf_index) = first_printf_word_index(&words) else {
            return false;
        };
        let mut words = words.into_iter().skip(printf_index + 1);
        while let Some(word) = words.next() {
            if word == "-v" {
                return words
                    .next()
                    .is_some_and(printf_v_target_can_mutate_command_lookup);
            }
        }
        false
    })
}

fn printf_v_target_can_mutate_command_lookup(name: &str) -> bool {
    let name = name.trim_end_matches('+');
    COMMAND_LOOKUP_ENV_KEYS.contains(&name) || command_word_uses_expansion_escape_or_glob(name)
}

fn first_printf_word_index(words: &[&str]) -> Option<usize> {
    let mut index = 0;
    let mut after_time = false;
    while let Some(word) = words.get(index) {
        if *word == "!" || *word == "time" {
            after_time = *word == "time";
            index += 1;
            continue;
        }
        if after_time && word.starts_with('-') {
            index += 1;
            continue;
        }
        return (*word == "printf").then_some(index);
    }
    None
}

fn line_uses_unsafe_tee(line: &str) -> bool {
    if !line_contains_shell_word(line, "tee") {
        return false;
    }
    !AUDIT_CAPTURE_PRODUCER_LINES.contains(&line)
}

fn shell_line_mutates_command_lookup(line: &str) -> bool {
    line_mutates_command_lookup(line)
        || shell_command_segments(line)
            .into_iter()
            .filter_map(first_command_word_in_segment)
            .map(|word| word.replace(['\'', '"'], ""))
            .any(|word| {
                matches!(
                    word.as_str(),
                    "alias"
                        | "unalias"
                        | "shopt"
                        | "command"
                        | "builtin"
                        | "export"
                        | "env"
                        | "source"
                        | "."
                        | "eval"
                        | "hash"
                        | "enable"
                        | "declare"
                        | "typeset"
                        | "readonly"
                        | "cd"
                        | "pushd"
                        | "popd"
                )
            })
}

fn run_block_prunes_deny_graph(run: &str) -> bool {
    active_run_lines(run)
        .into_iter()
        .any(|line| line.contains("--exclude"))
}

fn run_invokes_deny_against_workspace_config(run: &str) -> bool {
    if run_is_vetted_audit_run(run) {
        return true;
    }
    if run_block_masks_failures(run) || run_block_prunes_deny_graph(run) {
        return false;
    }
    active_run_lines(run).into_iter().any(|line| {
        line == "cargo deny --config deny.toml check" || line == "cargo deny -c deny.toml check"
    })
}

fn run_invokes_negative_symbol_check(run: &str) -> bool {
    if run_is_vetted_audit_run(run) {
        return true;
    }
    const PATTERN_SINGLE: &str = "'embeddinggemma|candle_embeddinggemma'";
    const PATTERN_DOUBLE: &str = "\"embeddinggemma|candle_embeddinggemma\"";
    if run_block_masks_failures(run) {
        return false;
    }
    [
        format!(
            "rg --files-without-match {PATTERN_SINGLE} \"$RUNNER_TEMP/context-graph-expanded.rs\""
        ),
        format!(
            "rg --files-without-match {PATTERN_DOUBLE} \"$RUNNER_TEMP/context-graph-expanded.rs\""
        ),
    ]
    .iter()
    .any(|rg_line| {
        run_contains_consecutive_lines(
            run,
            &[
                "cargo expand --lib -p context-graph | tee \"$RUNNER_TEMP/context-graph-expanded.rs\"",
                "test -s \"$RUNNER_TEMP/context-graph-expanded.rs\"",
                rg_line.as_str(),
            ],
        )
    })
}

fn run_invokes_negative_context_graph_dep_tree_check(run: &str) -> bool {
    if run_is_vetted_audit_run(run) {
        return true;
    }
    const PATTERN: &str = "(^|[[:space:]])candle-embeddinggemma([[:space:]]|$)";
    if run_block_masks_failures(run) {
        return false;
    }
    [
        "cargo tree -p context-graph -e normal,build | tee \"$RUNNER_TEMP/context-graph-tree.txt\"",
        "cargo tree -p context-graph --edges normal,build | tee \"$RUNNER_TEMP/context-graph-tree.txt\"",
    ]
    .iter()
    .any(|tree_line| {
        [
            format!("rg --files-without-match '{PATTERN}' \"$RUNNER_TEMP/context-graph-tree.txt\""),
            format!("rg --files-without-match \"{PATTERN}\" \"$RUNNER_TEMP/context-graph-tree.txt\""),
        ]
        .iter()
        .any(|rg_line| {
            run_contains_consecutive_lines(
                run,
                &[
                    *tree_line,
                    "test -s \"$RUNNER_TEMP/context-graph-tree.txt\"",
                    rg_line.as_str(),
                ],
            )
        })
    })
}

fn run_contains_consecutive_lines(run: &str, expected: &[&str]) -> bool {
    active_run_lines(run)
        .windows(expected.len())
        .any(|window| window == expected)
}

fn run_is_vetted_audit_run(run: &str) -> bool {
    active_run_lines(run) == VETTED_AUDIT_RUN_LINES
}

fn run_is_exact_revision_verifier(run: &str) -> bool {
    active_run_lines(run) == EXACT_REVISION_VERIFIER_LINES
}

fn workflow_steps(yaml: &Value) -> Vec<&Mapping> {
    let jobs = value_mapping_get(yaml, "jobs")
        .and_then(Value::as_mapping)
        .expect("workflow jobs map");
    jobs.values()
        .flat_map(|job| {
            value_mapping_get(job, "steps")
                .and_then(Value::as_sequence)
                .into_iter()
                .flatten()
        })
        .filter_map(Value::as_mapping)
        .collect()
}

fn workflow_step_contexts(yaml: &Value) -> Vec<(&Mapping, &Mapping)> {
    let jobs = value_mapping_get(yaml, "jobs")
        .and_then(Value::as_mapping)
        .expect("workflow jobs map");
    let mut contexts = Vec::new();
    for job_value in jobs.values() {
        let Some(job) = job_value.as_mapping() else {
            continue;
        };
        if let Some(steps) = value_mapping_get(job_value, "steps").and_then(Value::as_sequence) {
            for step_value in steps {
                if let Some(step) = step_value.as_mapping() {
                    contexts.push((job, step));
                }
            }
        }
    }
    contexts
}

fn workflow_job_steps(yaml: &Value) -> Vec<(&Mapping, Vec<&Mapping>)> {
    let jobs = value_mapping_get(yaml, "jobs")
        .and_then(Value::as_mapping)
        .expect("workflow jobs map");
    let mut job_steps = Vec::new();
    for (name, job_value) in jobs {
        if ci_stages::is_reusable_job(name, job_value) {
            continue;
        }
        let Some(job) = job_value.as_mapping() else {
            continue;
        };
        let steps = value_mapping_get(job_value, "steps")
            .and_then(Value::as_sequence)
            .into_iter()
            .flatten()
            .filter_map(Value::as_mapping)
            .collect();
        job_steps.push((job, steps));
    }
    job_steps
}

fn value_default_run_shell(value: &Value) -> Option<&str> {
    value_mapping_get(value, "defaults")
        .and_then(Value::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(Value::as_mapping)
        .and_then(|run| mapping_get(run, "shell"))
        .and_then(yaml_string)
}

fn mapping_default_run_shell(mapping: &Mapping) -> Option<&str> {
    mapping_get(mapping, "defaults")
        .and_then(Value::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(Value::as_mapping)
        .and_then(|run| mapping_get(run, "shell"))
        .and_then(yaml_string)
}

fn value_default_run_working_directory(value: &Value) -> Option<&Value> {
    value_mapping_get(value, "defaults")
        .and_then(Value::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(Value::as_mapping)
        .and_then(|run| mapping_get(run, "working-directory"))
}

fn mapping_default_run_working_directory(mapping: &Mapping) -> Option<&Value> {
    mapping_get(mapping, "defaults")
        .and_then(Value::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(Value::as_mapping)
        .and_then(|run| mapping_get(run, "working-directory"))
}

fn env_overrides_command_lookup(value: &Value) -> bool {
    value.as_mapping().is_some_and(|env| {
        env.keys().filter_map(Value::as_str).any(|key| {
            COMMAND_LOOKUP_ENV_KEYS
                .iter()
                .any(|lookup_key| key.eq_ignore_ascii_case(lookup_key))
        })
    })
}

fn step_uses(step: &Mapping) -> Option<&str> {
    let uses = mapping_get(step, "uses").and_then(yaml_string)?;
    // Exact immutable revisions retain the action identity and its declared version.
    match uses {
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" => {
            assert!(
                include_str!("../../../../.github/workflows/ci.yml").contains(
                    "uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4"
                )
            );
            Some("actions/checkout@v4")
        }
        "dtolnay/rust-toolchain@686976e191b89faba57d3206551f0f330d8cb249" => {
            assert!(
                include_str!("../../../../.github/workflows/ci.yml").contains(
                    "uses: dtolnay/rust-toolchain@686976e191b89faba57d3206551f0f330d8cb249 # stable"
                )
            );
            Some("dtolnay/rust-toolchain@stable")
        }
        "Swatinem/rust-cache@49a0bdc70d2e1b713ca9e2869b211fcce03d3c1c" => {
            assert!(
                include_str!("../../../../.github/workflows/ci.yml").contains(
                    "uses: Swatinem/rust-cache@49a0bdc70d2e1b713ca9e2869b211fcce03d3c1c # v2"
                )
            );
            Some("Swatinem/rust-cache@v2")
        }
        "taiki-e/install-action@1168f6810a3cb30a3514bf5756a3b7f0422c6df8" => {
            assert!(
                include_str!("../../../../.github/workflows/ci.yml").contains(
                    "uses: taiki-e/install-action@1168f6810a3cb30a3514bf5756a3b7f0422c6df8 # v2"
                )
            );
            Some("taiki-e/install-action@v2")
        }
        "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830" => {
            assert!(
                include_str!("../../../../.github/workflows/ci.yml")
                    .contains("uses: actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830 # v4")
            );
            Some("actions/cache@v4")
        }
        _ => Some(uses),
    }
}

fn step_run(step: &Mapping) -> Option<&str> {
    mapping_get(step, "run").and_then(yaml_string)
}

fn step_is_exact_contextdb_sha_input_validation(step: &Mapping) -> bool {
    mapping_get(step, "if").and_then(yaml_string) == Some("github.event_name == 'workflow_dispatch'")
        && mapping_get(step, "working-directory").and_then(yaml_string) == Some(".")
        && !step_continue_on_error_can_mask_failures(step)
        && mapping_get(step, "env")
            .and_then(Value::as_mapping)
            .and_then(|env| mapping_get(env, "CONTEXTDB_SHA"))
            .and_then(yaml_string)
            == Some("${{ inputs.contextdb_sha }}")
        && step_run(step).is_some_and(|run| {
            run_contains_consecutive_lines(
                run,
                &[
                    "if [[ ! \"$CONTEXTDB_SHA\" =~ ^[0-9a-f]{40}$ ]]; then",
                    "echo \"contextdb_sha must be an exact 40-character lowercase hexadecimal commit SHA\" >&2",
                    "exit 1",
                    "fi",
                ],
            )
        })
}

fn job_verifies_selected_checkout(job: &Mapping) -> bool {
    let steps = mapping_get(job, "steps")
        .and_then(Value::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(Value::as_mapping)
        .collect::<Vec<_>>();
    let checkout = steps
        .iter()
        .position(|step| step_is_credential_free_context_graph_checkout(step));
    let verification = steps
        .iter()
        .position(|step| step_is_exact_revision_verification(step));
    matches!((checkout, verification), (Some(c), Some(v)) if c < v)
}

fn assert_selected_checkout_verification(workflow: &Value) {
    for job in [
        "source",
        "lint",
        "docs",
        "test",
        "features",
        "controls",
        "feature-qualification-core",
        "feature-qualification-sync",
    ] {
        assert!(
            job_verifies_selected_checkout(
                workflow["jobs"][job].as_mapping().expect("required job")
            ),
            "{job} must verify its selected checkout before running its commands"
        );
        let mut missing = workflow.clone();
        let steps = missing["jobs"][job]["steps"]
            .as_sequence_mut()
            .expect("job steps");
        let before = steps.len();
        steps.retain(|step| {
            !step
                .as_mapping()
                .is_some_and(step_is_exact_revision_verification)
        });
        assert_eq!(
            steps.len() + 1,
            before,
            "remove exactly the revision verifier"
        );
        assert!(
            !job_verifies_selected_checkout(
                missing["jobs"][job].as_mapping().expect("required job")
            ),
            "removing verification from {job} must fail"
        );
        let mut wrong = workflow.clone();
        let mut changed = false;
        for step in wrong["jobs"][job]["steps"]
            .as_sequence_mut()
            .expect("job steps")
        {
            if step
                .as_mapping()
                .is_some_and(step_is_exact_revision_verification)
            {
                step["env"]["TESTED_SHA"] = Value::String("${{ github.sha }}".to_string());
                changed = true;
            }
        }
        assert!(changed);
        assert!(
            !job_verifies_selected_checkout(wrong["jobs"][job].as_mapping().expect("required job")),
            "verifying the merge event instead of the selected head must fail"
        );
    }
    let root = workspace_root();
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .expect("read checkout revision");
    assert!(head.status.success());
    let head = String::from_utf8(head.stdout).expect("SHA text");
    let head = head.trim();
    let verifier = EXACT_REVISION_VERIFIER_LINES.join("\n");
    for (tested, event, expected) in [
        (head, "0".repeat(40), true),
        (head, head.to_string(), true),
        (
            "0000000000000000000000000000000000000000",
            head.to_string(),
            false,
        ),
    ] {
        let output = Command::new("bash")
            .args(["-e", "-c", &verifier])
            .current_dir(&root)
            .env("TESTED_SHA", tested)
            .env("EVENT_SHA", event)
            .env("BASE_SHA", "")
            .env("HEAD_SHA", tested)
            .env("REQUESTED_CONTEXTDB_SHA", "")
            .output()
            .expect("execute the shipped revision verifier");
        assert_eq!(
            output.status.success(),
            expected,
            "the checkout must match the selected tested SHA for PR and push events: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn step_is_exact_revision_verification(step: &Mapping) -> bool {
    mapping_get(step, "if").is_none()
        && !step_continue_on_error_can_mask_failures(step)
        && mapping_get(step, "env")
            .and_then(Value::as_mapping)
            .and_then(|env| mapping_get(env, "REQUESTED_CONTEXTDB_SHA"))
            .and_then(yaml_string)
            == Some("${{ inputs.contextdb_sha }}")
        && mapping_get(step, "env")
            .and_then(Value::as_mapping)
            .and_then(|env| mapping_get(env, "TESTED_SHA"))
            .and_then(yaml_string)
            == Some("${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}")
        && step_run(step).is_some_and(run_is_exact_revision_verifier)
}

fn step_uses_can_mask_command_lookup(step: &Mapping) -> bool {
    match step_uses(step) {
        None => false,
        Some("actions/checkout@v4") => checkout_inputs_can_shadow_workspace(step),
        Some("taiki-e/install-action@v2") => {
            !step_tool_set_equals(step, &["cargo-deny", "cargo-expand", "cargo-nextest"])
        }
        Some("actions/cache@v4") => cache_inputs_can_shadow_workspace_or_command_lookup(step),
        Some("Swatinem/rust-cache@v2") => rust_cache_inputs_can_shadow_command_lookup(step),
        Some("actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093") => {
            ci_stages::artifact_download_can_shadow_command_lookup(step)
        }
        Some(uses) => !COMMAND_LOOKUP_SAFE_ACTIONS.contains(&uses),
    }
}

fn checkout_inputs_can_shadow_workspace(step: &Mapping) -> bool {
    !step_is_credential_free_context_graph_checkout(step)
}

fn step_is_credential_free_context_graph_checkout(step: &Mapping) -> bool {
    step_uses(step) == Some("actions/checkout@v4")
        && step_with_fields(step).is_some_and(|with| {
            with.len() == 4
                && mapping_get(with, "ref").and_then(yaml_string)
                    == Some(
                        "${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}",
                    )
                && mapping_get(with, "path").and_then(yaml_string)
                    == Some(CONTEXT_GRAPH_CHECKOUT_PATH)
                && mapping_get(with, "fetch-depth").is_some_and(|value| {
                    yaml_explicit_integer(value, CONTEXT_GRAPH_CHECKOUT_FETCH_DEPTH)
                })
                && mapping_get(with, "persist-credentials").is_some_and(yaml_explicit_false)
        })
}

fn cache_inputs_can_shadow_workspace_or_command_lookup(step: &Mapping) -> bool {
    let Some(path) = step_with_field(step, "path") else {
        return true;
    };
    field_mentions_command_lookup_target(path) || !cache_path_is_model_cache_only(path)
}

fn rust_cache_inputs_can_shadow_command_lookup(step: &Mapping) -> bool {
    let Some(with) = step_with_fields(step) else {
        return true;
    };
    if !step_with_field(step, "cache-bin").is_some_and(yaml_explicit_false) {
        return true;
    }
    with.iter().any(|(key, value)| {
        let key = key.as_str().unwrap_or_default();
        match key {
            "cache-bin" => !yaml_explicit_false(value),
            "cache-directories" => true,
            _ => field_mentions_command_lookup_target(value),
        }
    })
}

fn cache_path_is_model_cache_only(value: &Value) -> bool {
    let mut lines = Vec::new();
    collect_yaml_string_lines(value, &mut lines);
    !lines.is_empty()
        && lines.iter().all(|line| {
            let normalized = line.replace('\\', "/");
            allowed_model_cache_path(&normalized)
                && !normalized.contains("github.workspace")
                && !normalized.contains("GITHUB_WORKSPACE")
        })
}

fn allowed_model_cache_path(path: &str) -> bool {
    path == MODEL_CACHE_ROOT_REFERENCE
}

fn collect_yaml_string_lines<'a>(value: &'a Value, lines: &mut Vec<&'a str>) {
    match value {
        Value::String(value) => {
            lines.extend(value.lines().map(str::trim).filter(|line| !line.is_empty()))
        }
        Value::Sequence(values) => values
            .iter()
            .for_each(|value| collect_yaml_string_lines(value, lines)),
        Value::Mapping(values) => values
            .values()
            .for_each(|value| collect_yaml_string_lines(value, lines)),
        _ => {}
    }
}

fn field_mentions_command_lookup_target(value: &Value) -> bool {
    AUDIT_TOOL_PATH_MARKERS
        .iter()
        .any(|marker| yaml_field_contains(value, marker))
        || COMMAND_LOOKUP_ENV_KEYS
            .iter()
            .any(|key| yaml_field_contains(value, key))
}

fn step_continue_on_error_can_mask_failures(step: &Mapping) -> bool {
    mapping_get(step, "continue-on-error").is_some_and(|value| !yaml_explicit_false(value))
}

fn job_continue_on_error_can_mask_failures(job: &Mapping) -> bool {
    mapping_get(job, "continue-on-error").is_some_and(|value| !yaml_explicit_false(value))
}

fn step_if_field_can_disable_guard(step: &Mapping) -> bool {
    mapping_get(step, "if").is_some()
}

fn job_if_field_can_disable_guard(job: &Mapping) -> bool {
    mapping_get(job, "if").is_some()
}

fn step_shell_can_mask_failures(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    let Some(shell) = mapping_get(step, "shell")
        .and_then(yaml_string)
        .or_else(|| mapping_default_run_shell(job))
        .or_else(|| value_default_run_shell(workflow))
    else {
        return true;
    };
    shell.trim() != "bash"
}

fn step_working_directory_can_shadow_workspace(
    workflow: &Value,
    job: &Mapping,
    step: &Mapping,
) -> bool {
    mapping_get(step, "working-directory").is_some()
        || mapping_default_run_working_directory(job).is_some()
        || value_default_run_working_directory(workflow).and_then(yaml_string)
            != Some(CONTEXT_GRAPH_WORKING_DIRECTORY)
}

/// Whether anything in the workflow can change what the shell resolves as `cargo` — anywhere, in
/// any job, at any position.
///
/// A command-lookup mutation is not local to the step that carries it: once `GITHUB_PATH`, `PATH`
/// or `BASH_ENV` is redirected, every command that runs afterwards resolves through the redirected
/// directory, so format, lint and `cargo test --workspace` can all be faked green by a mutation
/// dropped after the step this audit is guarding. The scan therefore covers the whole document and
/// exempts only the exact vetted runs, matched line for line — never by name, prefix or substring.
fn workflow_command_lookup_can_mask_failures(workflow: &Value) -> bool {
    if value_mapping_get(workflow, "env").is_some_and(env_overrides_command_lookup) {
        return true;
    }

    workflow_job_steps(workflow).iter().any(|(job, steps)| {
        job_runtime_can_mask_command_lookup(job)
            || mapping_get(job, "env").is_some_and(env_overrides_command_lookup)
            || steps.iter().any(|step| {
                mapping_get(step, "env").is_some_and(env_overrides_command_lookup)
                    || step_uses_can_mask_command_lookup(step)
                    || step_run(step).is_some_and(|run| {
                        !run_is_vetted_command_lookup_run(run)
                            && all_run_shell_lines(run)
                                .into_iter()
                                .any(shell_line_mutates_command_lookup)
                    })
            })
    })
}

const VETTED_GATE_WALL_CLOCK_MEASUREMENT_LINES: &[&str] = &[
    "export CG_GATE_JUNIT=\"$repository_root/target/nextest/gate/junit.xml\"",
    "default_baseline",
    "rustup run 1.94.1 cargo test -p context-graph --test test_estate_audit gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact",
    "python3 scripts/check-gate-wall-clock.py --junit \"$CG_GATE_JUNIT\"",
];

fn assert_exact_python_blocks(workflow: &Value) {
    ci_stages::assert_exact_shared_commands(workflow, workflow_command_lookup_can_mask_failures);
    for block in [VETTED_JOB_RECEIPT_RUN, REQUIRED_PR_JOB_AGGREGATE_RUN] {
        assert!(run_is_vetted_command_lookup_run(block));
        for altered in [
            block.replace(
                "import json, os",
                "import json, os; os.environ['PATH'] = '/tmp/shadow'",
            ),
            format!("{block}python3 -c 'print(123)'\n"),
        ] {
            assert_ne!(
                altered, block,
                "the control must alter exact executable content"
            );
            assert!(
                !run_is_vetted_command_lookup_run(&altered),
                "changed Python bodies are not vetted"
            );
            let mut mutated = workflow.clone();
            let mut changed = 0;
            for (name, job) in mutated["jobs"].as_mapping_mut().expect("jobs map") {
                if ci_stages::is_reusable_job(name, job) {
                    continue;
                }
                for step in job["steps"].as_sequence_mut().expect("job steps") {
                    if step["run"].as_str() == Some(block) {
                        step["run"] = Value::String(altered.clone());
                        changed += 1;
                    }
                }
            }
            assert!(
                changed > 0,
                "the control must mutate a shipped Python block"
            );
            assert!(
                workflow_command_lookup_can_mask_failures(&mutated),
                "changed receipt or aggregate bodies must be rejected"
            );
        }
    }
    let mut unknown = workflow.clone();
    let mut interpreter_step = Mapping::new();
    interpreter_step.insert(
        Value::String("run".to_string()),
        Value::String("python3 -c 'print(123)'".to_string()),
    );
    unknown["jobs"]["lint"]["steps"]
        .as_sequence_mut()
        .expect("lint steps")
        .push(Value::Mapping(interpreter_step));
    assert!(
        workflow_command_lookup_can_mask_failures(&unknown),
        "an unrelated interpreter step remains rejected"
    );
    let disk = VETTED_DISK_RECLAMATION_LINES.join("\n");
    assert!(run_is_vetted_command_lookup_run(&disk));
    assert!(!run_is_vetted_command_lookup_run(&format!(
        "{disk}\npython3 -c 'print(123)'"
    )));
}

fn run_is_vetted_command_lookup_run(run: &str) -> bool {
    if ci_stages::exact_shared_command(run) {
        return true;
    }
    if run == VETTED_JOB_RECEIPT_RUN || run == REQUIRED_PR_JOB_AGGREGATE_RUN {
        return true;
    }
    let lines = active_run_lines(run);
    lines == EXACT_CONTEXTDB_SHA_INPUT_VALIDATION_LINES
        || lines == EXACT_REVISION_VERIFIER_LINES
        || lines == VETTED_AUDIT_RUN_LINES
        || lines == VETTED_COLD_MISS_MODEL_SEED_LINES
        || lines == VETTED_OFFLINE_REAL_MODEL_PROOF_LINES
        || lines == VETTED_DOCUMENTATION_CLAIM_VALIDATION_LINES
        || lines == VETTED_GATE_WALL_CLOCK_BASELINE_DIRECTION_LINES
        || lines == VETTED_GATE_WALL_CLOCK_MEASUREMENT_LINES
        || lines == VETTED_DISK_RECLAMATION_LINES
}

fn job_runtime_can_mask_command_lookup(job: &Mapping) -> bool {
    mapping_get(job, "container").is_some()
        || mapping_get(job, "services").is_some()
        || !job_runs_on_known_github_hosted_runner(job)
}

fn job_runs_on_known_github_hosted_runner(job: &Mapping) -> bool {
    let Some(runs_on) = mapping_get(job, "runs-on") else {
        return false;
    };
    match runs_on {
        Value::String(runner) => {
            let runner = runner.trim();
            is_safe_github_hosted_runner(runner)
                || (runner.replace(' ', "") == "${{matrix.os}}"
                    && job_matrix_os_values_are_known_github_hosted(job))
        }
        Value::Sequence(runners) => {
            !runners.is_empty()
                && runners
                    .iter()
                    .all(|runner| runner.as_str().is_some_and(is_safe_github_hosted_runner))
        }
        _ => false,
    }
}

fn job_matrix_os_values_are_known_github_hosted(job: &Mapping) -> bool {
    let Some(matrix) = job_matrix(job) else {
        return false;
    };
    ci_stages::exact_physical_matrix(matrix)
        || (mapping_get(matrix, "include").is_none()
            && mapping_get(matrix, "exclude").is_none()
            && job_matrix_values(job, "os").is_some_and(|values| {
                values
                    .iter()
                    .all(|value| is_safe_github_hosted_runner(value))
            }))
}

fn job_matrix(job: &Mapping) -> Option<&Mapping> {
    mapping_get(job, "strategy")
        .and_then(Value::as_mapping)
        .and_then(|strategy| mapping_get(strategy, "matrix"))
        .and_then(Value::as_mapping)
}

fn job_matrix_values(job: &Mapping, key: &str) -> Option<Vec<String>> {
    let values = job_matrix(job)
        .and_then(|matrix| mapping_get(matrix, key))
        .and_then(Value::as_sequence)?;
    let values = values
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| values.into_iter().map(ToOwned::to_owned).collect())
}

fn is_safe_github_hosted_runner(runner: &str) -> bool {
    SAFE_GITHUB_HOSTED_RUNNERS.contains(&runner.trim())
}

fn step_can_enforce_audit_failure(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    !step_continue_on_error_can_mask_failures(step)
        && !job_continue_on_error_can_mask_failures(job)
        && !step_if_field_can_disable_guard(step)
        && !job_if_field_can_disable_guard(job)
        && !step_shell_can_mask_failures(workflow, job, step)
        && !step_working_directory_can_shadow_workspace(workflow, job, step)
        && !workflow_command_lookup_can_mask_failures(workflow)
}

fn step_enforcing_run<'a>(workflow: &Value, job: &Mapping, step: &'a Mapping) -> Option<&'a str> {
    step_can_enforce_audit_failure(workflow, job, step).then(|| step_run(step))?
}

fn step_with_field<'a>(step: &'a Mapping, field: &str) -> Option<&'a Value> {
    step_with_fields(step).and_then(|with| mapping_get(with, field))
}

fn step_with_fields(step: &Mapping) -> Option<&Mapping> {
    mapping_get(step, "with").and_then(Value::as_mapping)
}

fn action_step_can_enforce_failure(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    !step_continue_on_error_can_mask_failures(step)
        && !job_continue_on_error_can_mask_failures(job)
        && !step_if_field_can_disable_guard(step)
        && !job_if_field_can_disable_guard(job)
        && !workflow_command_lookup_can_mask_failures(workflow)
}

fn step_installs_tools(
    workflow: &Value,
    job: &Mapping,
    step: &Mapping,
    required_tools: &[&str],
) -> bool {
    step_uses(step) == Some("taiki-e/install-action@v2")
        && step_tool_set_equals(step, required_tools)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_is_context_graph_checkout(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    step_is_credential_free_context_graph_checkout(step)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_pins_stable_rust_toolchain(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    step_uses(step) == Some("dtolnay/rust-toolchain@stable")
        && step_with_field(step, "toolchain").and_then(yaml_string) == Some("1.94.1")
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_uses_safe_rust_cache(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    step_uses(step) == Some("Swatinem/rust-cache@v2")
        && !rust_cache_inputs_can_shadow_command_lookup(step)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_uses_safe_model_cache(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    step_uses(step) == Some("actions/cache@v4")
        && !cache_inputs_can_shadow_workspace_or_command_lookup(step)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn prior_steps_prepare_audit_tools(
    workflow: &Value,
    job: &Mapping,
    prior_steps: &[&Mapping],
) -> bool {
    prior_steps
        .iter()
        .any(|step| step_is_context_graph_checkout(workflow, job, step))
        && prior_steps
            .iter()
            .any(|step| step_pins_stable_rust_toolchain(workflow, job, step))
        && prior_steps
            .iter()
            .any(|step| step_uses_safe_rust_cache(workflow, job, step))
        && prior_steps.iter().any(|step| {
            step_installs_tools(
                workflow,
                job,
                step,
                &["cargo-deny", "cargo-expand", "cargo-nextest"],
            )
        })
        && prior_steps
            .iter()
            .all(|step| prior_step_can_precede_audit(workflow, job, step))
}

fn prior_step_can_precede_audit(workflow: &Value, job: &Mapping, step: &Mapping) -> bool {
    (step_run(step).is_some_and(|run| active_run_lines(run) == VETTED_DISK_RECLAMATION_LINES)
        && mapping_get(step, "if").and_then(yaml_string) == Some("runner.os == 'Linux'"))
        || step_is_exact_contextdb_sha_input_validation(step)
        || step_is_exact_revision_verification(step)
        || step_is_context_graph_checkout(workflow, job, step)
        || step_pins_stable_rust_toolchain(workflow, job, step)
        || step_uses_safe_rust_cache(workflow, job, step)
        || step_uses_safe_model_cache(workflow, job, step)
        || step_installs_tools(
            workflow,
            job,
            step,
            &["cargo-deny", "cargo-expand", "cargo-nextest"],
        )
}

fn workflow_has_enforcing_audit_run_with_setup(
    workflow: &Value,
    run_predicate: impl Fn(&str) -> bool,
) -> bool {
    workflow_job_steps(workflow).iter().any(|(job, steps)| {
        steps.iter().enumerate().any(|(index, step)| {
            step_enforcing_run(workflow, job, step).is_some_and(|run| {
                run_predicate(run)
                    && prior_steps_prepare_audit_tools(workflow, job, &steps[..index])
            })
        })
    })
}

fn step_tool_set_equals(step: &Mapping, required_tools: &[&str]) -> bool {
    let Some(tool_field) = step_with_field(step, "tool").and_then(Value::as_str) else {
        return false;
    };
    let tools = tool_field
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .map(str::trim)
        .filter(|tool| !tool.is_empty())
        .collect::<std::collections::BTreeSet<_>>();
    let tool_names = tools
        .iter()
        .map(|tool| tool.split_once('@').map_or(*tool, |(name, _)| name))
        .collect::<std::collections::BTreeSet<_>>();
    let expected = required_tools
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let exact_installs = AUDIT_TOOL_INSTALL_SPECS
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    tool_names == expected && tools == exact_installs
}

#[test]
fn oss_purity_no_embeddinggemma_symbol_leakage_in_main_crate() {
    assert_no_cargo_audit_aliases();
    assert!(
        which::which("cargo-expand").is_ok(),
        "install cargo-expand so this guard can inspect the expanded context-graph library"
    );
    let output = Command::new("cargo")
        .args(["expand", "--lib", "-p", "context-graph"])
        .current_dir(workspace_root())
        .output()
        .expect("cargo expand starts");
    assert!(
        output.status.success(),
        "cargo expand --lib -p context-graph failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let expanded_source = String::from_utf8_lossy(&output.stdout).to_lowercase();
    assert!(
        !expanded_source.contains("embeddinggemma")
            && !expanded_source.contains("candle_embeddinggemma"),
        "context-graph expanded library source must not include sibling model runtime symbols"
    );
    let tree = Command::new("cargo")
        .args(["tree", "-p", "context-graph", "-e", "normal,build"])
        .current_dir(workspace_root())
        .output()
        .expect("cargo tree starts");
    assert!(
        tree.status.success(),
        "cargo tree -p context-graph -e normal,build failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&tree.stdout),
        String::from_utf8_lossy(&tree.stderr)
    );
    let dep_tree = String::from_utf8_lossy(&tree.stdout).to_lowercase();
    assert!(
        !dep_tree.contains("candle-embeddinggemma") && !dep_tree.contains("candle_embeddinggemma"),
        "context-graph normal/build dependency tree must not include sibling model runtime"
    );
    let yaml = read_workflow_yaml();
    assert!(
        workflow_has_enforcing_audit_run_with_setup(&yaml, run_invokes_negative_symbol_check),
        "ci must keep the expanded-source OSS purity guard as an unconditional failure-loud workflow step with same-job checkout/toolchain/cache/tool setup"
    );
    assert!(
        workflow_has_enforcing_audit_run_with_setup(
            &yaml,
            run_invokes_negative_context_graph_dep_tree_check
        ),
        "ci must keep the context-graph normal/build dependency-tree OSS purity guard as an unconditional failure-loud workflow step with same-job checkout/toolchain/cache/tool setup"
    );
}

#[test]
fn ci_workflow_uses_dtolnay_rust_toolchain_stable_pin_parsed_yaml() {
    let yaml = read_workflow_yaml();
    let steps = workflow_steps(&yaml);
    assert!(
        steps
            .iter()
            .filter_map(|step| step_uses(step))
            .any(|uses| uses == "dtolnay/rust-toolchain@stable")
    );
}

/// Every required job verifies the selected checkout and the locked dependency revision.
#[test]
fn ci_workflow_validates_and_verifies_the_exact_contextdb_revision() {
    assert_selected_checkout_verification(&read_workflow_yaml());
    let yaml = read_workflow_yaml();
    let dispatch_inputs = value_mapping_get(&yaml, "on")
        .and_then(Value::as_mapping)
        .and_then(|trigger| mapping_get(trigger, "workflow_dispatch"))
        .and_then(Value::as_mapping)
        .and_then(|dispatch| mapping_get(dispatch, "inputs"))
        .and_then(Value::as_mapping)
        .expect("workflow_dispatch contextdb_sha input");
    let contextdb_sha = mapping_get(dispatch_inputs, "contextdb_sha")
        .and_then(Value::as_mapping)
        .expect("contextdb_sha input");
    assert_eq!(
        mapping_get(contextdb_sha, "required").and_then(Value::as_bool),
        Some(true),
        "manual coordinated runs must require the immutable ContextDB SHA"
    );
    assert_eq!(
        mapping_get(contextdb_sha, "type").and_then(yaml_string),
        Some("string"),
        "manual coordinated runs must receive the ContextDB SHA as text"
    );

    let contexts = workflow_step_contexts(&yaml);
    assert!(
        contexts
            .iter()
            .any(|(_, step)| step_is_exact_contextdb_sha_input_validation(step)),
        "manual coordinated runs must reject anything but a lowercase 40-character ContextDB SHA before checkout"
    );
    assert!(
        contexts
            .iter()
            .any(|(_, step)| step_is_credential_free_context_graph_checkout(step)),
        "CI must check out Context Graph only into its declared workspace path without persisted credentials"
    );
    assert!(
        !contexts.iter().any(|(_, step)| {
            step_with_fields(step).is_some_and(|with| {
                mapping_get(with, "repository").and_then(yaml_string)
                    == Some("context-graph-ai/contextdb")
            })
        }),
        "the build compiles the ContextDB revision locked in Cargo.lock; a separate ContextDB checkout would report an engine the build never uses"
    );
    assert!(
        contexts
            .iter()
            .any(|(_, step)| step_is_exact_revision_verification(step)),
        "every matrix job must print and verify the checked-out Context Graph SHA and the locked ContextDB SHA before building"
    );
}

/// Every git revision of `github.com/context-graph-ai/contextdb` named by a line of `text`, taken
/// from `rev = "<sha>"` (manifests) or `?rev=<sha>#<sha>` (lockfile source).
fn contextdb_revisions_named_in(text: &str) -> Vec<String> {
    let mut revisions = Vec::new();
    for line in text.lines() {
        if !line.contains("github.com/context-graph-ai/contextdb") {
            continue;
        }
        let rev = line
            .split_once("rev = \"")
            .map(|(_, rest)| rest.split('"').next().unwrap_or_default())
            .or_else(|| {
                line.split_once("contextdb?rev=")
                    .map(|(_, rest)| rest.split(['#', '"']).next().unwrap_or_default())
            })
            .unwrap_or_else(|| panic!("ContextDB dependency without an exact revision: {line}"));
        if let Some((_, fragment)) = line.split_once('#') {
            let locked = fragment.trim_end_matches('"');
            assert_eq!(
                locked, rev,
                "Cargo.lock must resolve ContextDB to the revision it requested: {line}"
            );
        }
        revisions.push(rev.to_owned());
    }
    revisions
}

#[test]
fn manifests_lockfile_and_readme_name_the_same_exact_contextdb_revision() {
    let root = workspace_root();
    let mut manifest_paths = vec![root.join("Cargo.toml")];
    for entry in std::fs::read_dir(root.join("crates")).expect("read crates directory") {
        let manifest = entry.expect("crate entry").path().join("Cargo.toml");
        if manifest.is_file() {
            manifest_paths.push(manifest);
        }
    }
    let mut revisions = std::collections::BTreeSet::new();
    for manifest in &manifest_paths {
        let text = std::fs::read_to_string(manifest).expect("read manifest");
        revisions.extend(contextdb_revisions_named_in(&text));
    }
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");
    let locked: std::collections::BTreeSet<String> =
        contextdb_revisions_named_in(&lock).into_iter().collect();
    assert_eq!(
        revisions.len(),
        1,
        "every manifest must pin ContextDB to one exact revision: {revisions:?}"
    );
    let pinned = revisions.into_iter().next().expect("one revision");
    assert!(
        pinned.len() == 40
            && pinned
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "the ContextDB pin must be a full lowercase 40-character commit SHA: {pinned}"
    );
    assert_eq!(
        locked,
        std::collections::BTreeSet::from([pinned.clone()]),
        "Cargo.lock must lock every ContextDB crate at the manifest revision"
    );
    let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");
    assert!(
        readme.contains(&pinned),
        "README.md must name the ContextDB revision the build compiles ({pinned})"
    );
    assert!(
        !readme.contains("checkout beside this one"),
        "README.md must not tell builders to place a ContextDB checkout the build never uses"
    );
}

#[test]
fn readme_states_the_rust_version_rust_toolchain_toml_pins() {
    let root = workspace_root();
    let toolchain = std::fs::read_to_string(root.join("rust-toolchain.toml"))
        .expect("read rust-toolchain.toml");
    let channel = toolchain
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("channel = \"")
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .expect("rust-toolchain.toml names a channel");
    assert_eq!(channel, "1.94.1", "the pinned toolchain");
    let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");
    assert!(
        readme.contains(&format!("Rust **{channel}**")),
        "README.md must state the Rust version rust-toolchain.toml pins ({channel})"
    );
}

#[test]
fn ci_workflow_uses_swatinem_rust_cache_v2_replacing_hand_rolled_actions_cache_v4_parsed_yaml() {
    let yaml = read_workflow_yaml();
    let steps = workflow_steps(&yaml);
    assert!(workflow_has_enforcing_audit_run_with_setup(&yaml, |run| {
        run_invokes_negative_symbol_check(run)
            || run_invokes_negative_context_graph_dep_tree_check(run)
    }));
    for step in steps
        .iter()
        .filter(|step| step_uses(step) == Some("actions/cache@v4"))
    {
        let cached_path = step_with_field(step, "path");
        assert!(
            !cached_path.is_some_and(|path| {
                yaml_field_contains(path, "target") || yaml_field_contains(path, ".cargo")
            }),
            "actions/cache@v4 must not cache Rust build artifacts"
        );
    }
}

#[test]
fn ci_workflow_model_cache_key_hashes_pinned_revision_constant_source_file_not_cargo_lock_parsed_yaml()
 {
    // CI uses an explicit root; fallback pins still name the root the loader reads.
    assert_eq!(
        cg_catalog::DEFAULT_CACHE_ROOT_LINUX_MACOS,
        "~/.cache/cg/models"
    );
    assert_eq!(
        cg_catalog::DEFAULT_CACHE_ROOT_WINDOWS,
        "%LOCALAPPDATA%\\cg\\models"
    );

    let yaml = read_workflow_yaml();
    let model_cache_steps = workflow_steps(&yaml)
        .into_iter()
        .filter(|step| {
            step_with_field(step, "path")
                .is_some_and(|path| yaml_field_contains(path, MODEL_CACHE_ROOT_REFERENCE))
        })
        .collect::<Vec<_>>();
    assert!(
        model_cache_steps.len() == 2,
        "default and feature jobs each restore their declared production model cache"
    );
    for step in model_cache_steps {
        let key = step_with_field(step, "key").expect("model cache key");
        assert!(
            yaml_field_contains(key, "crates/context-graph/src/embedder/nomic.rs"),
            "model cache key should hash the pinned revision source file"
        );
        assert!(
            !yaml_field_contains(key, "Cargo.lock"),
            "model cache key should not churn on dependency lockfile changes"
        );
    }
}

/// Only superseded PR-number runs are cancelled; manual qualification keeps its own run identity.
#[test]
fn ci_workflow_uses_concurrency_cancel_in_progress_at_workflow_level_parsed_yaml() {
    let yaml = read_workflow_yaml();
    let concurrency = value_mapping_get(&yaml, "concurrency").and_then(Value::as_mapping);
    assert!(
        concurrency.is_some(),
        "workflow should define workflow-level concurrency"
    );
    let concurrency = concurrency.unwrap_or_else(|| unreachable!());
    let group = mapping_get(concurrency, "group").and_then(yaml_string);
    assert!(
        group.is_some(),
        "workflow concurrency should name a stable group"
    );
    assert_eq!(
        group,
        Some(
            "ci-${{ github.event_name == 'pull_request' && github.event.pull_request.number || github.run_id }}"
        )
    );
    assert_eq!(
        mapping_get(concurrency, "cancel-in-progress").and_then(yaml_string),
        Some("${{ github.event_name == 'pull_request' }}")
    );
}

#[test]
fn deny_toml_exists_at_workspace_root_with_bans_licenses_advisories_sources_sections_covering_full_ban_list()
 {
    assert!(run_invokes_deny_against_workspace_config(
        "cargo deny --config deny.toml check"
    ));
    assert!(run_invokes_deny_against_workspace_config(
        "cargo deny -c deny.toml check"
    ));
    for rejected in [
        "cargo deny check --config deny.toml",
        "cargo deny check -c deny.toml",
        "cargo deny check",
        "cargo deny --config other.toml check",
        "cargo deny --config deny.toml check || true",
        "cargo deny --config deny.toml check --exclude banned",
    ] {
        assert!(
            !run_invokes_deny_against_workspace_config(rejected),
            "the dependency audit must reject {rejected}"
        );
    }

    assert_no_cargo_audit_aliases();
    let deny = fs::read_to_string(workspace_root().join("deny.toml")).expect("deny.toml readable");
    let parsed: toml::Value = toml::from_str(&deny).expect("deny.toml parses");
    let bans = parsed
        .get("bans")
        .and_then(toml::Value::as_table)
        .expect("[bans] table");
    if let Some(graph) = parsed.get("graph").and_then(toml::Value::as_table) {
        assert!(
            graph.is_empty(),
            "[graph] must stay absent or empty; target, feature, and exclude pruning can hide banned transitive crates from cargo-deny's audit graph"
        );
    }
    for skip_key in ["skip", "skip-tree"] {
        if let Some(skipped) = bans.get(skip_key).and_then(toml::Value::as_array) {
            assert!(
                skipped.is_empty(),
                "[bans].{skip_key} must stay empty; skip-tree can prune banned transitive crates through non-banned roots, so all cargo-deny graph pruning belongs in explicit scoped wrapper rules only"
            );
        }
    }
    let deny_entries = bans
        .get("deny")
        .and_then(toml::Value::as_array)
        .expect("[bans].deny array");
    assert!(
        !deny_entries.is_empty(),
        "[bans].deny must contain explicit crate entries"
    );
    let denied_names = deny_entries
        .iter()
        .map(|entry| {
            let entry = entry.as_table().expect("banned crate entry table");
            assert!(
                !entry.contains_key("crate"),
                "[bans].deny entries must use cargo-deny's name field, not a crate alias"
            );
            let raw_name = entry
                .get("name")
                .and_then(toml::Value::as_str)
                .expect("banned crate name");
            assert!(
                BANNED_CRATES.contains(&raw_name),
                "[bans].deny entries must use plain banned crate names with no source, path, or version qualifiers; got {raw_name}"
            );
            assert_eq!(
                entry.get("version").and_then(toml::Value::as_str),
                Some("*"),
                "banned crate entries should explicitly apply to all versions"
            );
            if raw_name == "candle-embeddinggemma" {
                assert_eq!(
                    raw_name, "candle-embeddinggemma",
                    "candle-embeddinggemma ban must use the plain crate name, not a source-qualified or version-qualified package id"
                );
                let wrappers = entry
                    .get("wrappers")
                    .and_then(toml::Value::as_array)
                    .expect("candle-embeddinggemma ban should have scoped wrappers");
                let wrappers = wrappers
                    .iter()
                    .map(|wrapper| wrapper.as_str().expect("wrapper crate name"))
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(
                    wrappers,
                    ["cg-catalog"]
                        .into_iter()
                        .collect::<std::collections::BTreeSet<_>>(),
                    "candle-embeddinggemma may be allowed only as a direct cg-catalog dependency"
                );
            } else if raw_name == "reqwest" {
                // The dial-by-key sync transport (iroh, via contextdb-server)
                // uses reqwest internally for relay/net plumbing. cg's own
                // code stays banned from it; the wrapper scope admits it ONLY
                // as the transport crates' dependency.
                let wrappers = entry
                    .get("wrappers")
                    .and_then(toml::Value::as_array)
                    .expect("reqwest ban should have transport-scoped wrappers");
                let wrappers = wrappers
                    .iter()
                    .map(|wrapper| wrapper.as_str().expect("wrapper crate name"))
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(
                    wrappers,
                    ["iroh", "iroh-relay"]
                        .into_iter()
                        .collect::<std::collections::BTreeSet<_>>(),
                    "reqwest may be allowed only as the sync transport's internal dependency"
                );
            } else {
                assert!(
                    !entry.contains_key("wrappers"),
                    "{raw_name} must not use cargo-deny wrapper exceptions; only candle-embeddinggemma (cg-catalog) and reqwest (the sync transport) may be scoped"
                );
            }
            raw_name.to_string()
        })
        .collect::<std::collections::BTreeSet<_>>();
    for banned in BANNED_CRATES {
        assert!(
            denied_names.contains(*banned),
            "missing banned crate {banned}"
        );
    }
    let licenses = parsed
        .get("licenses")
        .and_then(toml::Value::as_table)
        .expect("[licenses] table");
    let allowed_licenses = licenses
        .get("allow")
        .and_then(toml::Value::as_array)
        .expect("[licenses].allow array");
    let allowed_licenses = allowed_licenses
        .iter()
        .map(|license| license.as_str().expect("license is a string"))
        .collect::<std::collections::BTreeSet<_>>();
    let expected_licenses = [
        "Apache-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CDLA-Permissive-2.0",
        "ISC",
        "MIT",
        "Unicode-3.0",
        // Public-domain-equivalent; reachable via the dial-by-key sync
        // transport's wasm-target dependencies; never compiled for shipped targets.
        "Unlicense",
        "Zlib",
    ]
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        allowed_licenses, expected_licenses,
        "[licenses].allow should be the exact non-copyleft dependency-graph floor"
    );
    assert!(
        !licenses.contains_key("exceptions"),
        "[licenses].exceptions must not weaken the license floor"
    );
    let advisory_ignores = parsed
        .get("advisories")
        .and_then(toml::Value::as_table)
        .expect("[advisories] table")
        .get("ignore")
        .and_then(toml::Value::as_array)
        .expect("[advisories].ignore array")
        .iter()
        .map(|advisory| advisory.as_str().expect("advisory ignore is a string"))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        advisory_ignores,
        ALLOWED_ADVISORY_IGNORES
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        "[advisories].ignore should contain only the current informational/unmaintained transitives"
    );
    let sources = parsed
        .get("sources")
        .and_then(toml::Value::as_table)
        .expect("[sources] table");
    assert_eq!(
        sources
            .get("unknown-registry")
            .and_then(toml::Value::as_str),
        Some("deny")
    );
    assert_eq!(
        sources.get("unknown-git").and_then(toml::Value::as_str),
        Some("deny")
    );
    let yaml = read_workflow_yaml();
    assert!(
        workflow_step_contexts(&yaml)
            .iter()
            .any(|(job, step)| step_installs_tools(
                &yaml,
                job,
                step,
                &["cargo-deny", "cargo-expand", "cargo-nextest"]
            )),
        "workflow should install cargo-deny, cargo-expand and cargo-nextest through taiki-e/install-action@v2 in an unconditional failure-loud step"
    );
    let steps = workflow_steps(&yaml);
    assert!(
        steps
            .iter()
            .filter_map(|step| step_run(step))
            .flat_map(str::lines)
            .filter_map(active_shell_line)
            .all(|line| !line_invokes_cargo_install(line)),
        "workflow should not compile audit tools or path-installed cargo subcommands via cargo install"
    );
    assert!(
        !yaml_field_contains(&yaml, "--exclude"),
        "workflow must not prune cargo-deny's graph with --exclude in CI run lines, env, or action fields"
    );
    assert!(
        workflow_step_contexts(&yaml).iter().any(|(job, step)| {
            step_enforcing_run(&yaml, job, step)
                .is_some_and(run_invokes_deny_against_workspace_config)
        }),
        "ci must run cargo deny check against deny.toml in an unconditional failure-loud step with an explicit config argument"
    );
}

#[test]
fn bespoke_runtime_audit_shell_script_deleted_no_longer_referenced_by_ci_workflow() {
    // Runtime audit coverage lives in Rust tests and CI steps.
    let root = workspace_root();
    assert!(
        !root
            .join(".github/scripts/context-graph-runtime-audit.sh")
            .exists()
    );
    let workflow = fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("workflow");
    assert!(!workflow.contains("context-graph-runtime-audit.sh"));
}

#[test]
fn cargo_deny_check_runs_clean_against_current_main_crate_dep_graph_and_fails_loud_when_binary_absent()
 {
    assert_no_cargo_audit_aliases();
    assert!(
        which::which("cargo-deny").is_ok(),
        "install cargo-deny to run this guard"
    );
    let current_syntax = Command::new("cargo")
        .args(["deny", "--config", "deny.toml", "check"])
        .current_dir(workspace_root())
        .output()
        .expect("cargo deny starts");
    let output = if current_syntax.status.success()
        || !String::from_utf8_lossy(&current_syntax.stderr)
            .contains("unexpected argument '--config'")
    {
        current_syntax
    } else {
        Command::new("cargo")
            .args(["deny", "check", "--config", "deny.toml"])
            .current_dir(workspace_root())
            .output()
            .expect("legacy cargo deny starts")
    };
    assert!(
        output.status.success(),
        "cargo deny check with deny.toml failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The job step carrying the vetted dependency-audit run, together with the job it lives in.
fn vetted_audit_step_context(workflow: &Value) -> (&Mapping, &Mapping) {
    workflow_step_contexts(workflow)
        .into_iter()
        .find(|(_, step)| step_run(step).is_some_and(run_is_vetted_audit_run))
        .expect("the workflow must carry the vetted dependency-audit run step")
}

/// A pull request must not be able to fake a green format, lint, or workspace-test run by changing
/// what the shell resolves as `cargo`. The rule that makes that true is that the command-lookup
/// audit inspects the WHOLE job: a `GITHUB_PATH` mutation dropped anywhere in the job — including
/// after the dependency audit and immediately before `cargo test --workspace` — must blank the
/// audit's guards and turn the suite red.
///
/// This drives the rule the only way it can be driven: it takes the real checked-in workflow,
/// confirms the audit reads it as clean, injects exactly that mutation at the workspace-test
/// position, and requires the audit to report the guarded audit step as maskable afterwards. An
/// audit that stops scanning at the step it is guarding never sees the injected mutation and
/// reports clean, which is the regression this proof exists to catch.
#[test]
fn ci_command_lookup_audit_covers_a_mutation_inserted_after_the_audit_step_and_before_workspace_tests()
 {
    assert_exact_python_blocks(&read_workflow_yaml());
    let mut workflow = read_workflow_yaml();

    // Panics unless the workflow carries the vetted dependency-audit run this proof injects around.
    vetted_audit_step_context(&workflow);
    assert!(
        !workflow_command_lookup_can_mask_failures(&workflow),
        "the checked-in workflow must start with no command-lookup mutation reported, so the \
         injected one below is the only thing that can change the verdict"
    );

    let mutation_run = "printf '%s\\n' \"$RUNNER_TEMP/shadow-bin\" >> \"$GITHUB_PATH\"";
    assert!(
        shell_line_mutates_command_lookup(mutation_run),
        "the audit's line classifier must recognize the injected GITHUB_PATH mutation"
    );

    let mut injected = false;
    {
        let jobs = workflow
            .as_mapping_mut()
            .and_then(|workflow| workflow.get_mut(Value::String("jobs".to_string())))
            .and_then(Value::as_mapping_mut)
            .expect("workflow jobs map");
        for job_value in jobs.values_mut() {
            let Some(steps) = job_value
                .as_mapping_mut()
                .and_then(|job| job.get_mut(Value::String("steps".to_string())))
                .and_then(Value::as_sequence_mut)
            else {
                continue;
            };
            let audit_index = steps.iter().position(|step| {
                step.as_mapping()
                    .and_then(step_run)
                    .is_some_and(run_is_vetted_audit_run)
            });
            let lint_index = steps.iter().position(|step| {
                step.as_mapping().and_then(step_run).is_some_and(|run| {
                    run.trim()
                        == "rustup run 1.94.1 cargo clippy --workspace --all-targets -- -D warnings"
                })
            });
            let workspace_test_index = lint_index.map(|index| index + 1);
            let (Some(audit_index), Some(lint_index), Some(workspace_test_index)) =
                (audit_index, lint_index, workspace_test_index)
            else {
                continue;
            };
            assert!(
                audit_index < lint_index && lint_index < workspace_test_index,
                "the real workflow must place the dependency audit before clippy, otherwise this proof is injecting the mutation somewhere it would already \
                 be scanned"
            );

            let mut mutation_step = Mapping::new();
            mutation_step.insert(
                Value::String("run".to_string()),
                Value::String(mutation_run.to_string()),
            );
            steps.insert(workspace_test_index, Value::Mapping(mutation_step));
            assert_eq!(
                steps[workspace_test_index].as_mapping().and_then(step_run),
                Some(mutation_run),
                "the mutation must sit immediately after clippy"
            );
            injected = true;
            break;
        }
    }
    assert!(
        injected,
        "the real workflow must expose the lint job containing the dependency audit and clippy"
    );

    assert!(
        workflow_command_lookup_can_mask_failures(&workflow),
        "the command-lookup audit must inspect the whole job and reject a GITHUB_PATH mutation \
         inserted after the dependency audit and clippy; an \
         audit that stops at the step it guards leaves format, lint and the workspace tests \
         resolvable through an attacker-supplied directory"
    );
}

/// A commit without a parent audits every documentation page and detects source crates without guides.
#[test]
fn doc_claim_check_audits_the_whole_tree_on_a_commit_with_no_parent() {
    let repo = tempfile::tempdir().expect("temporary repository");
    let pages = [
        "README.md",
        "crates/example/README.md",
        "docs/usage.md",
        "skills/example/SKILL.md",
    ];
    for page in pages {
        let path = repo.path().join(page);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path,
            "# Usage\n\nThe command always returns stored conversation passages without modifying any captured source events.\n",
        )
        .unwrap();
    }
    let source = repo.path().join("crates/unguided/src/lib.rs");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(source, "pub fn value() -> u32 { 1 }\n").unwrap();
    let scripts = repo.path().join("scripts");
    std::fs::create_dir_all(&scripts).unwrap();
    std::fs::copy(
        workspace_root().join("scripts/validate-doc-coverage.py"),
        scripts.join("validate-doc-coverage.py"),
    )
    .unwrap();

    for args in [
        vec!["init", "--quiet"],
        vec!["add", "--", "."],
        vec![
            "-c",
            "user.name=Test Author",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "docs: describe conversation retrieval",
        ],
    ] {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(repo.path())
            .output()
            .expect("prepare the parentless repository");
        assert!(output.status.success(), "git setup failed: {output:?}");
    }
    let parents = std::process::Command::new("git")
        .args(["rev-list", "--parents", "-n", "1", "HEAD"])
        .current_dir(repo.path())
        .output()
        .unwrap();
    assert!(parents.status.success());
    assert_eq!(
        String::from_utf8_lossy(&parents.stdout)
            .split_whitespace()
            .count(),
        1
    );

    let root = workspace_root();
    let workflow = read_workflow_yaml();
    let ci_step = workflow["jobs"]["docs"]["steps"]
        .as_sequence()
        .expect("docs steps")
        .iter()
        .find(|step| {
            step["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("Documented behavior claims"))
        })
        .and_then(|step| step["run"].as_str())
        .expect("effective CI documentation commands")
        .to_string();
    let gate = std::fs::read_to_string(root.join("scripts/gate.sh")).unwrap();
    let gate_step = gate
        .split("docs() {\n")
        .nth(1)
        .expect("gate documentation stage")
        .split("\n}\n")
        .next()
        .unwrap();

    for (entry, commands) in [("CI", ci_step.as_str()), ("gate", gate_step)] {
        for base in [
            "",
            "0000000000000000000000000000000000000000",
            "unknown-base",
        ] {
            let output = std::process::Command::new("bash")
                .args(["-e", "-c", commands])
                .current_dir(repo.path())
                .env("DOCUMENTATION_BASE_SHA", base)
                .env("GATE_BASE", base)
                .output()
                .expect("run the documentation entry point on a parentless repository");
            let said = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                output.status.code() == Some(1)
                    && pages.iter().all(|page| said.contains(&format!("{page}:3")))
                    && said.contains("4 unbound claim(s)")
                    && said.contains("carry no AGENTS.md: unguided")
                    && said.contains("initial state, no comparison"),
                "{entry} with base {base:?} must audit all four pages at file:line and disclose initial state; status {}:\n{said}",
                output.status
            );
        }
    }
}

// This proof executes the core Rust arm built by the workspace alongside this acceptance suite.
fn compiled_test_estate_binary() -> std::path::PathBuf {
    let deps = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    fs::read_dir(deps)
        .expect("workspace test binaries")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("test_estate_audit-") && !name.contains('.'))
        })
        .max_by_key(|path| fs::metadata(path).unwrap().modified().unwrap())
        .expect("build the workspace test estate binary before running the CI acceptance suite")
}

/// Suite duration overruns report the binary and seconds without failing either gate check.
#[test]
fn gate_wall_clock_overrun_is_reported_not_failed() {
    let root = workspace_root();
    let junit = root.join("crates/context-graph/tests/fixtures/gate_wall_clock_overrun.xml");
    let baseline: toml::Value = toml::from_str(
        &fs::read_to_string(
            root.join("crates/context-graph/tests/fixtures/test_estate_baseline.toml"),
        )
        .unwrap(),
    )
    .unwrap();
    let binary = "context-graph::test_estate_audit";
    let measured: f64 = fs::read_to_string(&junit)
        .unwrap()
        .lines()
        .filter(|line| line.trim_start().starts_with("<testcase "))
        .map(|line| {
            line.split("time=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap()
                .parse::<f64>()
                .unwrap()
        })
        .sum();
    let number = |value: &toml::Value| {
        value
            .as_float()
            .unwrap_or_else(|| value.as_integer().unwrap() as f64)
    };
    let ceiling = number(&baseline["gate_wall_seconds"][binary])
        * number(&baseline["gate_wall_seconds_policy"]["tolerance_factor"])
        + number(&baseline["gate_wall_seconds_policy"]["slack_seconds"]);
    assert!(measured > ceiling, "the fixture must overrun its ceiling");

    let rust_arm = compiled_test_estate_binary();
    let rust = Command::new(&rust_arm)
        .args([
            "gate_wall_seconds_never_exceed_the_frozen_baseline",
            "--ignored",
            "--exact",
            "--nocapture",
        ])
        .env("CG_GATE_JUNIT", &junit)
        .output()
        .expect("run the gate's Rust wall-clock arm");
    let python = std::process::Command::new("python3")
        .arg(root.join("scripts/check-gate-wall-clock.py"))
        .arg("--junit")
        .arg(&junit)
        .arg("--baseline")
        .arg(root.join("crates/context-graph/tests/fixtures/test_estate_baseline.toml"))
        .output()
        .expect("run the gate's Python wall-clock check");
    let results = [("Rust", rust), ("Python", python)];
    let failures: Vec<String> = results
        .iter()
        .filter_map(|(surface, output)| {
            let said = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let reports_seconds = said.lines().any(|line| {
                line.contains(binary)
                    && line.contains(&format!("{measured:.3}s"))
                    && line.contains(&format!("{ceiling:.3}s"))
            });
            (!output.status.success() || !reports_seconds).then(|| {
                format!("{surface}: status {}; expected success and {binary} with {measured:.3}s measured and {ceiling:.3}s ceiling:\n{said}", output.status)
            })
        })
        .collect();
    assert!(
        failures.is_empty(),
        "duration overruns must be reported without failing the gate:\n{}",
        failures.join("\n")
    );

    let missing =
        root.join("crates/context-graph/tests/fixtures/gate_wall_clock_missing_limit.xml");
    let unbounded = "context-graph::without_recorded_limit";
    assert!(
        !baseline["gate_wall_seconds"]
            .as_table()
            .unwrap()
            .contains_key(unbounded)
    );
    let rust = Command::new(&rust_arm)
        .args([
            "gate_wall_seconds_never_exceed_the_frozen_baseline",
            "--ignored",
            "--exact",
            "--nocapture",
        ])
        .env("CG_GATE_JUNIT", &missing)
        .output()
        .expect("run the Rust wall-clock arm with a missing limit");
    let python = std::process::Command::new("python3")
        .arg(root.join("scripts/check-gate-wall-clock.py"))
        .arg("--junit")
        .arg(&missing)
        .output()
        .expect("run the Python wall-clock checker with a missing limit");
    for (surface, output) in [("Rust", rust), ("Python", python)] {
        let said = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !output.status.success() && said.contains(unbounded),
            "{surface} must fail and name the binary with no recorded limit; status {}:\n{said}",
            output.status
        );
    }
}
