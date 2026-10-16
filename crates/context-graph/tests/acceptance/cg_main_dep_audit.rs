#[path = "../common/ci_stages.rs"]
mod ci_stages;
use ci_stages::{
    REQUIRED_PR_JOB_AGGREGATE_RUN, VETTED_DISK_RECLAMATION_LINES, VETTED_JOB_RECEIPT_RUN,
};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use context_graph::embedder::mock::MockEmbedder;
use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::{
    CG_SCHEMA_VERSION, CgError, Embedder, EmbedderConfig, EmbeddingInput, EvidenceKind, Store,
    StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;
use serde_yaml_ng::{Mapping, Value as YamlValue};

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
const AUDIT_TOOL_INSTALL_SPECS: &[&str] = &[
    "cargo-deny@0.20.2",
    "cargo-expand@1.0.124",
    "cargo-nextest@0.9.140",
];
const EXACT_CONTEXTDB_SHA_INPUT_VALIDATION_LINES: &[&str] = &[
    "if [[ ! \"$CONTEXTDB_SHA\" =~ ^[0-9a-f]{40}$ ]]; then",
    "echo \"contextdb_sha must be an exact 40-character lowercase hexadecimal commit SHA\" >&2",
    "exit 1",
    "fi",
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

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    crate_root()
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn read_workspace_file(path: &str) -> String {
    std::fs::read_to_string(workspace_root().join(path)).expect("workspace file readable")
}

fn read_workflow_yaml() -> YamlValue {
    {
        let mut yaml = serde_yaml_ng::from_str(&read_workspace_file(".github/workflows/ci.yml"))
            .expect("workflow yaml");
        yaml = ci_stages::platform_coverage(
            yaml,
            serde_yaml_ng::from_str(
                &std::fs::read_to_string(
                    workspace_root().join(".github/workflows/macos-advisory.yml"),
                )
                .expect("advisory readable"),
            )
            .expect("advisory yaml"),
        );
        ci_stages::effective(&mut yaml, &read_workspace_file("scripts/gate.sh"));
        yaml
    }
}

fn default_build_dependency_names(manifest_path: &str) -> Vec<String> {
    let contents = read_workspace_file(manifest_path);
    let mut dependencies = Vec::new();
    let mut active_default_build_section = false;
    let mut multiline = None;
    for raw_line in contents.lines() {
        let line = strip_unquoted_toml_comment(raw_line, &mut multiline);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            let section = toml_table_name(line);
            active_default_build_section = matches!(
                section.as_deref(),
                Some("dependencies" | "build-dependencies")
            );
            continue;
        }
        if active_default_build_section
            && let Some((raw_key, _)) = line.split_once('=')
            && let Some(name) = toml_key_parts(raw_key).first()
        {
            dependencies.push(name.clone());
        }
    }
    dependencies
}

pub(crate) fn default_build_tree(package: &str) -> String {
    let output = Command::new("cargo")
        .args(["tree", "-p", package, "--edges", "normal,build"])
        .current_dir(workspace_root())
        .output()
        .expect("run cargo tree for default build dependency audit");
    assert!(
        output.status.success(),
        "cargo tree for {package} default build failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo tree output utf8")
}

/// Every optional dependency the bundled local text embedder pulls in. With the default features
/// off, none of these may appear in the core library's build tree — `ureq` is the one that can
/// open a socket, and the rest are the model runtime that only exists to serve it.
const BUNDLED_EMBEDDER_DEPENDENCIES: &[&str] = &[
    "candle-core",
    "candle-nn",
    "candle-transformers",
    "safetensors",
    "tokenizers",
    "ureq",
];

/// The build tree of `package` with its default features turned off.
fn build_tree_without_default_features(package: &str) -> String {
    let output = Command::new("cargo")
        .args([
            "tree",
            "-p",
            package,
            "--no-default-features",
            "--edges",
            "normal,build",
        ])
        .current_dir(workspace_root())
        .output()
        .expect("run cargo tree for the default-features-off dependency audit");
    assert!(
        output.status.success(),
        "cargo tree for {package} with default features off failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo tree output utf8")
}

/// A build directory of its own for the nested default-features-off compile, so it neither waits
/// on nor invalidates the artifacts the surrounding test run is using. It sits under whatever
/// build directory is already in effect, so it is disposable with the rest of them and stays warm
/// between runs.
fn nested_check_target_dir() -> PathBuf {
    let ambient = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("target"));
    ambient.join("no-default-features-check")
}

/// The published promise is that a consumer who turns the default features off gets a core that
/// fetches nothing. This requires both a successful build in that shape and a dependency tree that
/// omits network fetching. The shape must
/// drop the dependency that can open a socket. So this compiles it and reads its dependency tree,
/// with the default tree as a positive control so a tree reader that silently found nothing cannot
/// pass.
#[test]
fn the_core_library_builds_with_default_features_off_and_drops_every_bundled_embedder_dependency() {
    let default_tree = default_build_tree("context-graph");
    for dependency in BUNDLED_EMBEDDER_DEPENDENCIES {
        assert!(
            default_tree.contains(&format!("{dependency} v")),
            "positive control: the DEFAULT build tree must contain {dependency}, otherwise the \
             default-features-off assertions below prove nothing about what turning the features \
             off removed"
        );
    }

    let bare_tree = build_tree_without_default_features("context-graph");
    for dependency in BUNDLED_EMBEDDER_DEPENDENCIES {
        assert!(
            !bare_tree.contains(&format!("{dependency} v")),
            "with default features off the core library must pull in no {dependency}: a core that \
             fetches nothing cannot carry the dependency that fetches, nor the model runtime that \
             exists only to be fed by it\n{bare_tree}"
        );
    }

    let target_dir = nested_check_target_dir();
    let output = Command::new("cargo")
        .args(["check", "-p", "context-graph", "--no-default-features"])
        .current_dir(workspace_root())
        .env("CARGO_TARGET_DIR", &target_dir)
        .output()
        .expect("run cargo check with default features off");
    assert!(
        output.status.success(),
        "the core library must compile with its default features off — a consumer told to turn \
         them off for a core that fetches nothing must get a core that builds: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
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
            let Ok(contents) = std::fs::read_to_string(&path) else {
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

fn value_mapping_get<'a>(value: &'a YamlValue, key: &str) -> Option<&'a YamlValue> {
    let key = YamlValue::String(key.to_string());
    value.as_mapping()?.get(&key)
}

fn mapping_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a YamlValue> {
    let key = YamlValue::String(key.to_string());
    mapping.get(&key)
}

fn workflow_steps(yaml: &YamlValue) -> Vec<&Mapping> {
    let jobs = value_mapping_get(yaml, "jobs")
        .and_then(YamlValue::as_mapping)
        .expect("workflow jobs map");
    jobs.values()
        .flat_map(|job| {
            value_mapping_get(job, "steps")
                .and_then(YamlValue::as_sequence)
                .into_iter()
                .flatten()
        })
        .filter_map(YamlValue::as_mapping)
        .collect()
}

fn workflow_step_contexts(yaml: &YamlValue) -> Vec<(&Mapping, &Mapping)> {
    let jobs = value_mapping_get(yaml, "jobs")
        .and_then(YamlValue::as_mapping)
        .expect("workflow jobs map");
    let mut contexts = Vec::new();
    for job_value in jobs.values() {
        let Some(job) = job_value.as_mapping() else {
            continue;
        };
        if let Some(steps) = value_mapping_get(job_value, "steps").and_then(YamlValue::as_sequence)
        {
            for step_value in steps {
                if let Some(step) = step_value.as_mapping() {
                    contexts.push((job, step));
                }
            }
        }
    }
    contexts
}

fn workflow_job_steps(yaml: &YamlValue) -> Vec<(&Mapping, Vec<&Mapping>)> {
    let jobs = value_mapping_get(yaml, "jobs")
        .and_then(YamlValue::as_mapping)
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
            .and_then(YamlValue::as_sequence)
            .into_iter()
            .flatten()
            .filter_map(YamlValue::as_mapping)
            .collect();
        job_steps.push((job, steps));
    }
    job_steps
}

fn step_run(step: &Mapping) -> Option<&str> {
    mapping_get(step, "run").and_then(YamlValue::as_str)
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
                && mapping_get(with, "ref").and_then(YamlValue::as_str)
                    == Some(
                        "${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}",
                    )
                && mapping_get(with, "path").and_then(YamlValue::as_str)
                    == Some(CONTEXT_GRAPH_CHECKOUT_PATH)
                && mapping_get(with, "persist-credentials").is_some_and(yaml_explicit_false)
                && mapping_get(with, "fetch-depth").is_some_and(yaml_explicit_zero)
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

fn cache_path_is_model_cache_only(value: &YamlValue) -> bool {
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
    // The catalog delegates these fallback roots to the bundled loader.
    [
        "${{ env.CG_MODEL_CACHE_ROOT }}",
        cg_catalog::DEFAULT_CACHE_ROOT_LINUX_MACOS,
        "~/AppData/Local/cg/models",
        "$HOME/.cache/cg/models",
        "${HOME}/.cache/cg/models",
        "$HOME/AppData/Local/cg/models",
        "${HOME}/AppData/Local/cg/models",
        "$LOCALAPPDATA/cg/models",
        "${LOCALAPPDATA}/cg/models",
        "%LOCALAPPDATA%/cg/models",
        "$XDG_CACHE_HOME/cg/models",
        "${XDG_CACHE_HOME}/cg/models",
    ]
    .iter()
    .any(|prefix| path == *prefix || path.starts_with(&format!("{prefix}/")))
}

fn collect_yaml_string_lines<'a>(value: &'a YamlValue, lines: &mut Vec<&'a str>) {
    match value {
        YamlValue::String(value) => {
            lines.extend(value.lines().map(str::trim).filter(|line| !line.is_empty()));
        }
        YamlValue::Sequence(values) => values
            .iter()
            .for_each(|value| collect_yaml_string_lines(value, lines)),
        YamlValue::Mapping(values) => values
            .values()
            .for_each(|value| collect_yaml_string_lines(value, lines)),
        _ => {}
    }
}

fn field_mentions_command_lookup_target(value: &YamlValue) -> bool {
    AUDIT_TOOL_PATH_MARKERS
        .iter()
        .any(|marker| yaml_field_contains(value, marker))
        || COMMAND_LOOKUP_ENV_KEYS
            .iter()
            .any(|key| yaml_field_contains(value, key))
}

fn yaml_explicit_false(value: &YamlValue) -> bool {
    match value {
        YamlValue::Bool(value) => !value,
        YamlValue::String(value) => value.trim().eq_ignore_ascii_case("false"),
        _ => false,
    }
}

fn yaml_explicit_zero(value: &YamlValue) -> bool {
    match value {
        YamlValue::Number(value) => value.as_i64() == Some(0),
        YamlValue::String(value) => value.trim() == "0",
        _ => false,
    }
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

fn value_default_run_shell(value: &YamlValue) -> Option<&str> {
    value_mapping_get(value, "defaults")
        .and_then(YamlValue::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(YamlValue::as_mapping)
        .and_then(|run| mapping_get(run, "shell"))
        .and_then(YamlValue::as_str)
}

fn mapping_default_run_shell(mapping: &Mapping) -> Option<&str> {
    mapping_get(mapping, "defaults")
        .and_then(YamlValue::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(YamlValue::as_mapping)
        .and_then(|run| mapping_get(run, "shell"))
        .and_then(YamlValue::as_str)
}

fn value_default_run_working_directory(value: &YamlValue) -> Option<&YamlValue> {
    value_mapping_get(value, "defaults")
        .and_then(YamlValue::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(YamlValue::as_mapping)
        .and_then(|run| mapping_get(run, "working-directory"))
}

fn mapping_default_run_working_directory(mapping: &Mapping) -> Option<&YamlValue> {
    mapping_get(mapping, "defaults")
        .and_then(YamlValue::as_mapping)
        .and_then(|defaults| mapping_get(defaults, "run"))
        .and_then(YamlValue::as_mapping)
        .and_then(|run| mapping_get(run, "working-directory"))
}

fn env_overrides_command_lookup(value: &YamlValue) -> bool {
    value.as_mapping().is_some_and(|env| {
        env.keys().filter_map(YamlValue::as_str).any(|key| {
            COMMAND_LOOKUP_ENV_KEYS
                .iter()
                .any(|lookup_key| key.eq_ignore_ascii_case(lookup_key))
        })
    })
}

fn step_shell_can_mask_failures(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    let Some(shell) = mapping_get(step, "shell")
        .and_then(YamlValue::as_str)
        .or_else(|| mapping_default_run_shell(job))
        .or_else(|| value_default_run_shell(workflow))
    else {
        return true;
    };
    shell.trim() != "bash"
}

fn step_working_directory_can_shadow_workspace(
    workflow: &YamlValue,
    job: &Mapping,
    step: &Mapping,
) -> bool {
    mapping_get(step, "working-directory").is_some()
        || mapping_default_run_working_directory(job).is_some()
        || value_default_run_working_directory(workflow).and_then(YamlValue::as_str)
            != Some(CONTEXT_GRAPH_WORKING_DIRECTORY)
}

fn workflow_command_lookup_can_mask_failures(workflow: &YamlValue) -> bool {
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
                            && all_run_shell_lines(run).into_iter().any(|line| {
                                line_has_unpinned_rustup_wrapper(line)
                                    || shell_line_mutates_command_lookup(line)
                            })
                    })
            })
    })
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
        YamlValue::String(runner) => {
            let runner = runner.trim();
            is_safe_github_hosted_runner(runner)
                || (runner.replace(' ', "") == "${{matrix.os}}"
                    && job_matrix_os_values_are_known_github_hosted(job))
        }
        YamlValue::Sequence(runners) => {
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
        .and_then(YamlValue::as_mapping)
        .and_then(|strategy| mapping_get(strategy, "matrix"))
        .and_then(YamlValue::as_mapping)
}

fn job_matrix_values(job: &Mapping, key: &str) -> Option<Vec<String>> {
    let values = job_matrix(job)
        .and_then(|matrix| mapping_get(matrix, key))
        .and_then(YamlValue::as_sequence)?;
    let values = values
        .iter()
        .map(YamlValue::as_str)
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| values.into_iter().map(ToOwned::to_owned).collect())
}

fn is_safe_github_hosted_runner(runner: &str) -> bool {
    SAFE_GITHUB_HOSTED_RUNNERS.contains(&runner.trim())
}

fn step_can_enforce_audit_failure(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    !step_continue_on_error_can_mask_failures(step)
        && !job_continue_on_error_can_mask_failures(job)
        && !step_if_field_can_disable_guard(step)
        && !job_if_field_can_disable_guard(job)
        && !step_shell_can_mask_failures(workflow, job, step)
        && !step_working_directory_can_shadow_workspace(workflow, job, step)
        && !workflow_command_lookup_can_mask_failures(workflow)
}

fn step_enforcing_run<'a>(
    workflow: &YamlValue,
    job: &Mapping,
    step: &'a Mapping,
) -> Option<&'a str> {
    step_can_enforce_audit_failure(workflow, job, step).then(|| step_run(step))?
}

fn step_uses(step: &Mapping) -> Option<&str> {
    let uses = mapping_get(step, "uses").and_then(YamlValue::as_str)?;
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

fn step_with_fields(step: &Mapping) -> Option<&Mapping> {
    mapping_get(step, "with").and_then(YamlValue::as_mapping)
}

fn step_with_field<'a>(step: &'a Mapping, field: &str) -> Option<&'a YamlValue> {
    step_with_fields(step).and_then(|with| mapping_get(with, field))
}

fn action_step_can_enforce_failure(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    !step_continue_on_error_can_mask_failures(step)
        && !job_continue_on_error_can_mask_failures(job)
        && !step_if_field_can_disable_guard(step)
        && !job_if_field_can_disable_guard(job)
        && !workflow_command_lookup_can_mask_failures(workflow)
}

fn step_installs_tools(
    workflow: &YamlValue,
    job: &Mapping,
    step: &Mapping,
    required_tools: &[&str],
) -> bool {
    step_uses(step) == Some("taiki-e/install-action@v2")
        && step_tool_set_equals(step, required_tools)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_is_exact_contextdb_sha_input_validation(step: &Mapping) -> bool {
    mapping_get(step, "if").and_then(YamlValue::as_str)
        == Some("github.event_name == 'workflow_dispatch'")
        && mapping_get(step, "working-directory").and_then(YamlValue::as_str) == Some(".")
        && !step_continue_on_error_can_mask_failures(step)
        && mapping_get(step, "env")
            .and_then(YamlValue::as_mapping)
            .and_then(|env| mapping_get(env, "CONTEXTDB_SHA"))
            .and_then(YamlValue::as_str)
            == Some("${{ inputs.contextdb_sha }}")
        && step_run(step)
            .is_some_and(|run| active_run_lines(run) == EXACT_CONTEXTDB_SHA_INPUT_VALIDATION_LINES)
}

fn job_verifies_selected_checkout(job: &Mapping) -> bool {
    let steps = mapping_get(job, "steps")
        .and_then(YamlValue::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(YamlValue::as_mapping)
        .collect::<Vec<_>>();
    let checkout = steps
        .iter()
        .position(|step| step_is_credential_free_context_graph_checkout(step));
    let verification = steps
        .iter()
        .position(|step| step_is_exact_revision_verification(step));
    matches!((checkout, verification), (Some(c), Some(v)) if c < v)
}

fn assert_selected_checkout_verification(workflow: &YamlValue) {
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
                step["env"]["TESTED_SHA"] = YamlValue::String("${{ github.sha }}".to_string());
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
            .and_then(YamlValue::as_mapping)
            .and_then(|env| mapping_get(env, "REQUESTED_CONTEXTDB_SHA"))
            .and_then(YamlValue::as_str)
            == Some("${{ inputs.contextdb_sha }}")
        && mapping_get(step, "env")
            .and_then(YamlValue::as_mapping)
            .and_then(|env| mapping_get(env, "TESTED_SHA"))
            .and_then(YamlValue::as_str)
            == Some("${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}")
        && step_run(step).is_some_and(run_is_exact_revision_verifier)
}

fn step_is_context_graph_checkout(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    step_is_credential_free_context_graph_checkout(step)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_pins_stable_rust_toolchain(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    step_uses(step) == Some("dtolnay/rust-toolchain@stable")
        && step_with_field(step, "toolchain").and_then(YamlValue::as_str) == Some("1.94.1")
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_uses_safe_rust_cache(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    step_uses(step) == Some("Swatinem/rust-cache@v2")
        && !rust_cache_inputs_can_shadow_command_lookup(step)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn step_uses_safe_model_cache(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    step_uses(step) == Some("actions/cache@v4")
        && !cache_inputs_can_shadow_workspace_or_command_lookup(step)
        && action_step_can_enforce_failure(workflow, job, step)
}

fn prior_steps_prepare_audit_tools(
    workflow: &YamlValue,
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

fn prior_step_can_precede_audit(workflow: &YamlValue, job: &Mapping, step: &Mapping) -> bool {
    (step_run(step).is_some_and(|run| active_run_lines(run) == VETTED_DISK_RECLAMATION_LINES)
        && mapping_get(step, "if").and_then(YamlValue::as_str) == Some("runner.os == 'Linux'"))
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
    workflow: &YamlValue,
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
    let Some(tool_field) = step_with_field(step, "tool").and_then(YamlValue::as_str) else {
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

fn yaml_field_contains(value: &YamlValue, needle: &str) -> bool {
    match value {
        YamlValue::String(value) => value.contains(needle),
        YamlValue::Sequence(values) => values
            .iter()
            .any(|value| yaml_field_contains(value, needle)),
        YamlValue::Mapping(values) => values
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

fn run_has_failure_neutering(run: &str) -> bool {
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

fn run_prunes_deny_graph(run: &str) -> bool {
    active_run_lines(run)
        .into_iter()
        .any(|line| line.contains("--exclude"))
}

fn run_is_vetted_audit_run(run: &str) -> bool {
    active_run_lines(run) == VETTED_AUDIT_RUN_LINES
}

fn run_is_exact_revision_verifier(run: &str) -> bool {
    active_run_lines(run) == EXACT_REVISION_VERIFIER_LINES
}

const VETTED_GATE_WALL_CLOCK_MEASUREMENT_LINES: &[&str] = &[
    "export CG_GATE_JUNIT=\"$repository_root/target/nextest/gate/junit.xml\"",
    "default_baseline",
    "rustup run 1.94.1 cargo test -p context-graph --test test_estate_audit gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact",
    "python3 scripts/check-gate-wall-clock.py --junit \"$CG_GATE_JUNIT\"",
];

fn line_has_unpinned_rustup_wrapper(line: &str) -> bool {
    let words = line.split_whitespace().collect::<Vec<_>>();
    words.first() == Some(&"rustup") && !words.starts_with(&["rustup", "run", "1.94.1", "cargo"])
}

fn assert_pinned_rustup_wrappers(workflow: &YamlValue) {
    for lines in [
        VETTED_COLD_MISS_MODEL_SEED_LINES,
        VETTED_OFFLINE_REAL_MODEL_PROOF_LINES,
        VETTED_GATE_WALL_CLOCK_MEASUREMENT_LINES,
    ] {
        let pinned = lines.join("\n");
        assert!(run_is_vetted_command_lookup_run(&pinned));
        for toolchain in ["stable", "nightly", "1.95.0", "$TOOLCHAIN"] {
            let unknown = pinned.replace("rustup run 1.94.1", &format!("rustup run {toolchain}"));
            assert!(
                !run_is_vetted_command_lookup_run(&unknown),
                "unknown wrappers are never vetted"
            );
            assert!(
                all_run_shell_lines(&unknown)
                    .into_iter()
                    .any(line_has_unpinned_rustup_wrapper)
            );
        }
    }
    for toolchain in ["stable", "nightly", "1.95.0", "$TOOLCHAIN"] {
        let mut unknown = workflow.clone();
        let steps = unknown["jobs"]["test"]["steps"]
            .as_sequence_mut()
            .expect("test steps");
        let mut changed = false;
        for step in steps {
            if step.as_mapping().and_then(step_run).is_some_and(|run| {
                run == ci_stages::body(include_str!("../../../../scripts/gate.sh"), "model_seed")
            }) {
                let run = step["run"]
                    .as_str()
                    .expect("seed run")
                    .replace("rustup run 1.94.1", &format!("rustup run {toolchain}"));
                step["run"] = YamlValue::String(run);
                changed = true;
            }
        }
        assert!(changed, "the control must mutate the real seed step");
        assert!(
            workflow_command_lookup_can_mask_failures(&unknown),
            "{toolchain} must fail the workflow scan"
        );
    }
}

fn assert_exact_python_blocks(workflow: &YamlValue) {
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
                        step["run"] = YamlValue::String(altered.clone());
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
        YamlValue::String("run".to_string()),
        YamlValue::String("python3 -c 'print(123)'".to_string()),
    );
    unknown["jobs"]["lint"]["steps"]
        .as_sequence_mut()
        .expect("lint steps")
        .push(YamlValue::Mapping(interpreter_step));
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

fn run_is_vetted_gate_wall_clock_baseline_direction_run(run: &str) -> bool {
    active_run_lines(run) == VETTED_GATE_WALL_CLOCK_BASELINE_DIRECTION_LINES
}

fn run_invokes_deny_against_workspace_config(run: &str) -> bool {
    if run_is_vetted_audit_run(run) {
        return true;
    }
    if run_has_failure_neutering(run) || run_prunes_deny_graph(run) {
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
    if run_has_failure_neutering(run) {
        return false;
    }
    [
        format!(
            "require_file_without_match {PATTERN_SINGLE} \"$RUNNER_TEMP/context-graph-expanded.rs\""
        ),
        format!(
            "require_file_without_match {PATTERN_DOUBLE} \"$RUNNER_TEMP/context-graph-expanded.rs\""
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

fn run_contains_consecutive_lines(run: &str, expected: &[&str]) -> bool {
    active_run_lines(run)
        .windows(expected.len())
        .any(|window| window == expected)
}

/// CI audits the dependency graph and verifies the selected checkout in every required job.
#[test]
fn ci_runs_cargo_deny_and_cargo_expand_symbol_check() {
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

    let workflow: YamlValue =
        serde_yaml_ng::from_str(include_str!("../../../../.github/workflows/ci.yml"))
            .expect("workflow YAML");
    let workflow = ci_stages::platform_coverage(
        workflow,
        serde_yaml_ng::from_str(include_str!(
            "../../../../.github/workflows/macos-advisory.yml"
        ))
        .expect("advisory yaml"),
    );
    assert!(ci_stages::controls_are_required(&workflow));
    for mutation in [
        "platform",
        "setup",
        "full command",
        "optional",
        "source",
        "admission",
    ] {
        let mut changed = workflow.clone();
        match mutation {
            "platform" => changed["jobs"]["controls"]["strategy"]["matrix"]["os"] = YamlValue::Null,
            "setup" => {
                changed["jobs"]["controls"]["steps"][3]["run"] = YamlValue::String("true".into())
            }
            "full command" => {
                changed["jobs"]["controls"]["steps"][4]["run"] =
                    YamlValue::String(format!("{} StageEvidence", ci_stages::CONTROLS_RUN))
            }
            "optional" => changed["jobs"]["controls"]["continue-on-error"] = YamlValue::Bool(true),
            "source" => {
                changed["jobs"]["controls"]["steps"][1]["with"]["ref"] =
                    YamlValue::String("${{ github.sha }}".into())
            }
            _ => changed["jobs"]["required"]["needs"] = YamlValue::Null,
        }
        assert!(
            !ci_stages::controls_are_required(&changed),
            "{mutation} must fail the controls contract"
        );
    }
    for run in [ci_stages::CONTROLS_SETUP, ci_stages::CONTROLS_RUN] {
        assert!(ci_stages::exact_controls_command(run));
        assert!(!ci_stages::exact_controls_command(&format!("{run}; true")));
        assert!(!ci_stages::exact_controls_command(&format!(
            "{run} StageEvidence"
        )));
    }
    assert_selected_checkout_verification(&read_workflow_yaml());
    assert_pinned_rustup_wrappers(&read_workflow_yaml());
    assert!(allowed_model_cache_path(
        cg_catalog::DEFAULT_CACHE_ROOT_LINUX_MACOS
    ));
    assert!(allowed_model_cache_path(
        &cg_catalog::DEFAULT_CACHE_ROOT_WINDOWS.replace('\\', "/")
    ));
    assert!(
        !allowed_model_cache_path("~/.cache/cg/embedders"),
        "the unsupported cache root is not read by the loader"
    );

    assert_no_cargo_audit_aliases();
    let yaml = read_workflow_yaml();
    let steps = workflow_steps(&yaml);
    let run_steps = steps
        .iter()
        .filter_map(|step| step_run(step))
        .collect::<Vec<_>>();
    let enforcing_run_steps = workflow_step_contexts(&yaml)
        .iter()
        .filter_map(|(job, step)| step_enforcing_run(&yaml, job, step))
        .collect::<Vec<_>>();
    let combined_audit_step = run_steps.iter().find(|run| {
        run_invokes_deny_against_workspace_config(run) && run_invokes_negative_symbol_check(run)
    });
    assert!(
        combined_audit_step.is_some(),
        "CI should run cargo deny check against deny.toml and cargo expand symbol inspection inside the same run block"
    );
    assert!(
        enforcing_run_steps.iter().any(|run| {
            run_invokes_deny_against_workspace_config(run) && run_invokes_negative_symbol_check(run)
        }),
        "CI audit guard must be an unconditional failure-loud workflow step"
    );
    assert!(
        workflow_has_enforcing_audit_run_with_setup(&yaml, |run| {
            run_invokes_deny_against_workspace_config(run) && run_invokes_negative_symbol_check(run)
        }),
        "CI audit guard must run in the same job after default checkout, stable Rust, safe rust-cache, and taiki cargo-deny/cargo-expand setup"
    );
    let expand_check = run_steps
        .iter()
        .find(|run| run_invokes_negative_symbol_check(run));
    assert!(
        expand_check.is_some(),
        "CI should run cargo expand directly and fail when embeddinggemma symbols appear"
    );
    assert!(
        enforcing_run_steps
            .iter()
            .any(|run| run_invokes_negative_symbol_check(run)),
        "CI expanded-source guard must not be hidden behind if, continue-on-error, or a failure-soft shell"
    );
    assert!(
        run_steps
            .iter()
            .all(|run| !run.contains("context-graph-runtime-audit.sh"))
    );
    assert!(
        workflow_has_enforcing_audit_run_with_setup(&yaml, run_invokes_negative_symbol_check),
        "CI should install cargo-deny and cargo-expand through taiki-e/install-action@v2 before the enforcing audit step in the same job"
    );
    assert!(
        run_steps
            .iter()
            .flat_map(|run| run.lines())
            .filter_map(active_shell_line)
            .all(|line| !line_invokes_cargo_install(line)),
        "CI should not hide audit-tool or path-installed cargo subcommand installation behind cargo install"
    );
    assert!(
        !yaml_field_contains(&yaml, "--exclude"),
        "CI should not prune cargo-deny's graph with --exclude in run lines, env, or action fields"
    );
    for step in &steps {
        let _ = step_with_fields(step);
    }
}

fn ci_job_steps(job: &Mapping) -> Vec<&Mapping> {
    mapping_get(job, "steps")
        .and_then(YamlValue::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(YamlValue::as_mapping)
        .collect()
}

fn split_ci_prerequisites_are_enforced(workflow: &YamlValue) -> bool {
    let Some(jobs) = value_mapping_get(workflow, "jobs").and_then(YamlValue::as_mapping) else {
        return false;
    };
    let (Some(lint), Some(tests), Some(required)) = (
        mapping_get(jobs, "lint").and_then(YamlValue::as_mapping),
        mapping_get(jobs, "test").and_then(YamlValue::as_mapping),
        mapping_get(jobs, "required").and_then(YamlValue::as_mapping),
    ) else {
        return false;
    };
    let lint_steps = ci_job_steps(lint);
    let test_steps = ci_job_steps(tests);
    let audit_index = lint_steps.iter().position(|step| {
        step_enforcing_run(workflow, lint, step).is_some_and(run_is_vetted_audit_run)
    });
    let lint_index = lint_steps.iter().position(|step| {
        step_enforcing_run(workflow, lint, step).is_some_and(|run| {
            run.trim() == "rustup run 1.94.1 cargo clippy --workspace --all-targets -- -D warnings"
        })
    });
    let test_index = test_steps.iter().position(|step| {
        step_enforcing_run(workflow, tests, step)
            .is_some_and(|run| run.trim() == "rustup run 1.94.1 cargo test --workspace --no-run")
    });
    let baseline_index = test_steps.iter().position(|step| {
        step_enforcing_run(workflow, tests, step)
            .is_some_and(run_is_vetted_gate_wall_clock_baseline_direction_run)
    });
    if !matches!((audit_index, lint_index), (Some(a), Some(l)) if a < l)
        || !matches!((test_index, baseline_index), (Some(t), Some(b)) if t < b)
    {
        return false;
    }
    for job_steps in [&lint_steps, &test_steps] {
        if !job_steps.iter().any(|step| {
            step_is_credential_free_context_graph_checkout(step)
                && step_with_field(step, "ref").and_then(YamlValue::as_str)
                    == Some(
                        "${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}",
                    )
        }) || !job_steps
            .iter()
            .any(|step| step_is_exact_revision_verification(step))
        {
            return false;
        }
    }
    let needs = mapping_get(required, "needs")
        .and_then(YamlValue::as_sequence)
        .and_then(|needs| {
            needs
                .iter()
                .map(YamlValue::as_str)
                .collect::<Option<Vec<_>>>()
        });
    let Some(mut needs) = needs else {
        return false;
    };
    needs.sort_unstable();
    if needs
        != [
            "controls",
            "docs",
            "feature-qualification-core",
            "feature-qualification-sync",
            "features",
            "lint",
            "source",
            "test",
        ]
        || mapping_get(required, "if").and_then(YamlValue::as_str) != Some("always()")
        || job_continue_on_error_can_mask_failures(required)
    {
        return false;
    }
    ci_job_steps(required).iter().any(|step| {
        step_run(step).is_some_and(|run| run.trim() == REQUIRED_PR_JOB_AGGREGATE_RUN.trim())
            && !step_if_field_can_disable_guard(step)
            && !step_continue_on_error_can_mask_failures(step)
            && mapping_get(step, "env")
                .and_then(YamlValue::as_mapping)
                .and_then(|env| mapping_get(env, "NEEDS_JSON"))
                .and_then(YamlValue::as_str)
                == Some("${{ toJSON(needs) }}")
    })
}

fn remove_ci_steps(workflow: &mut YamlValue, job_name: &str, predicate: impl Fn(&str) -> bool) {
    let steps = workflow["jobs"][job_name]["steps"]
        .as_sequence_mut()
        .expect("job steps");
    let before = steps.len();
    steps.retain(|step| !step.as_mapping().and_then(step_run).is_some_and(&predicate));
    assert!(steps.len() < before, "the control must remove a real step");
}

/// Audit, lint and workspace tests share the selected revision and remain required across jobs.
#[test]
fn ci_command_lookup_audit_covers_mutations_after_dependency_audit_and_before_workspace_tests() {
    assert_exact_python_blocks(&read_workflow_yaml());
    let yaml = read_workflow_yaml();
    assert!(
        !workflow_command_lookup_can_mask_failures(&yaml),
        "the checked-in workflow must start without a command-lookup mutation"
    );
    assert!(
        split_ci_prerequisites_are_enforced(&yaml),
        "audit precedes lint, tests precede baseline inspection, both verify the same selected checkout, and the aggregate requires both"
    );

    for (job, command) in [
        ("lint", "cargo deny --config deny.toml check"),
        (
            "lint",
            "cargo clippy --workspace --all-targets -- -D warnings",
        ),
        ("test", "cargo test --workspace"),
    ] {
        let mut missing = yaml.clone();
        remove_ci_steps(&mut missing, job, |run| {
            if job == "test" {
                run.trim() == "rustup run 1.94.1 cargo test --workspace --no-run"
            } else {
                run.contains(command)
            }
        });
        assert!(
            !split_ci_prerequisites_are_enforced(&missing),
            "removing {command} from {job} must fail the prerequisites"
        );
    }
    for job in ["lint", "test", "controls"] {
        let mut missing_need = yaml.clone();
        missing_need["jobs"]["required"]["needs"]
            .as_sequence_mut()
            .expect("aggregate needs")
            .retain(|need| need.as_str() != Some(job));
        assert!(!split_ci_prerequisites_are_enforced(&missing_need));
    }
    let mut wrong_checkout = yaml.clone();
    for step in wrong_checkout["jobs"]["test"]["steps"]
        .as_sequence_mut()
        .expect("test steps")
    {
        if step
            .as_mapping()
            .is_some_and(step_is_credential_free_context_graph_checkout)
        {
            step["with"]["ref"] = YamlValue::String("${{ github.sha }}".to_string());
        }
    }
    assert!(!split_ci_prerequisites_are_enforced(&wrong_checkout));

    let aggregate_code = REQUIRED_PR_JOB_AGGREGATE_RUN
        .strip_prefix("python3 - <<'PY_CHECK'\n")
        .and_then(|run| run.strip_suffix("PY_CHECK\n"))
        .expect("exact aggregate Python block");
    let success = serde_json::json!({
        "source": {"result": "success"}, "lint": {"result": "success"},
        "docs": {"result": "success"}, "test": {"result": "success"},
        "features": {"result": "success"}, "controls": {"result": "success"},
        "feature-qualification-core": {"result": "success"}, "feature-qualification-sync": {"result": "success"}
    });
    let evaluate = |needs: &serde_json::Value| {
        Command::new("python3")
            .args(["-I", "-c", aggregate_code])
            .env("NEEDS_JSON", needs.to_string())
            .output()
            .expect("evaluate the shipped aggregate")
            .status
            .success()
    };
    assert!(evaluate(&success));
    for job in ["lint", "test", "controls"] {
        let mut missing = success.clone();
        missing.as_object_mut().expect("needs map").remove(job);
        assert!(!evaluate(&missing), "missing {job} must fail the aggregate");
        for result in ["failure", "cancelled", "skipped"] {
            let mut failed = success.clone();
            failed[job]["result"] = serde_json::Value::String(result.to_string());
            assert!(!evaluate(&failed), "{job} {result} must fail the aggregate");
        }
    }
    let mut unexpected = success;
    unexpected["extra"] = serde_json::json!({"result": "success"});
    assert!(!evaluate(&unexpected));

    let mutation_run = "printf '%s\\n' \"$RUNNER_TEMP/shadow-bin\" >> \"$GITHUB_PATH\"";
    assert!(shell_line_mutates_command_lookup(mutation_run));
    for (job, command, after) in [
        (
            "lint",
            "rustup run 1.94.1 cargo clippy --workspace --all-targets -- -D warnings",
            true,
        ),
        (
            "test",
            "rustup run 1.94.1 cargo test --workspace --no-run",
            false,
        ),
    ] {
        let mut mutated = yaml.clone();
        let steps = mutated["jobs"][job]["steps"]
            .as_sequence_mut()
            .expect("job steps");
        let index = steps
            .iter()
            .position(|step| {
                step.as_mapping()
                    .and_then(step_run)
                    .is_some_and(|run| run.trim() == command)
            })
            .expect("the guarded command must run");
        let index = index + usize::from(after);
        let mut mutation_step = Mapping::new();
        mutation_step.insert(
            YamlValue::String("run".to_string()),
            YamlValue::String(mutation_run.to_string()),
        );
        steps.insert(index, YamlValue::Mapping(mutation_step));
        assert_eq!(
            steps[index].as_mapping().and_then(step_run),
            Some(mutation_run)
        );
        assert!(
            workflow_command_lookup_can_mask_failures(&mutated),
            "command lookup mutation in {job} must fail"
        );
    }
}

#[test]
fn default_build_dep_tree_for_context_graph_and_cg_catalog_has_no_tokio_runtime() {
    let root_manifests = [
        "crates/context-graph/Cargo.toml",
        "crates/cg-catalog/Cargo.toml",
    ];
    let direct_default_build_deps = root_manifests
        .iter()
        .flat_map(|manifest| default_build_dependency_names(manifest))
        .collect::<Vec<_>>();
    assert!(
        !direct_default_build_deps.iter().any(|name| name == "tokio"),
        "context-graph + cg-catalog default-build manifests must not directly depend on tokio: {direct_default_build_deps:?}"
    );
    for package in ["context-graph", "cg-catalog"] {
        let tree = default_build_tree(package);
        assert!(
            !tree.lines().any(|line| line.contains("tokio v")),
            "{package} default-build dependency tree must not pull tokio:\n{tree}"
        );
    }
}

#[test]
fn embedder_trait_surface_remains_byte_identical_with_catalog_workspace_member() {
    let embedder: Box<dyn Embedder> = Box::new(MockEmbedder::new("acceptance_mock_text", 8));
    let output = embedder
        .embed(EmbeddingInput::Text("dependency audit probe".to_string()))
        .expect("mock embedder output");
    assert_eq!("acceptance_mock_text", output.embedding_space_id);
    assert_eq!(8, output.dimension);
    assert_eq!(8, output.vector.len());
    assert_eq!(vec![EvidenceKind::Text], embedder.supported_inputs());
    assert_eq!(8, embedder.embedding_space().dimension);

    let disabled_nomic = NomicTextEmbedder::new(EmbedderConfig::disabled());
    let error = disabled_nomic
        .embed(EmbeddingInput::Text("must not download".to_string()))
        .expect_err("disabled Nomic embedder should not load a model");
    assert!(matches!(error, CgError::TextEmbedderDisabled));
}

#[test]
fn store_open_with_default_config_against_freshly_created_path_writes_cg_schema_version_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("fresh-main-crate-store.db");
    let mut config = StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    };
    config.default_text_embedder = Some(EmbedderConfig::disabled());

    let store = Store::open(config).expect("fresh store opens");
    assert_eq!(
        "embedding_loader=disabled",
        store.last_query_trace().expect("loader trace")
    );
    assert!(
        !store
            .list_embedding_spaces()
            .expect("embedding spaces")
            .is_empty()
    );
    drop(store);

    let db = Database::open(&db_path).expect("reopen contextdb file");
    let rows = db
        .execute("SELECT version FROM cg_schema_version", &HashMap::new())
        .expect("schema version query");
    assert_eq!(1, rows.rows.len(), "schema version row count");
    assert_eq!(
        Some(&Value::Int64(CG_SCHEMA_VERSION as i64)),
        rows.rows.first().and_then(|row| row.first())
    );
}

#[test]
fn nomic_loads_existing_acceptance_tests_remain_registered_with_catalog_workspace_member() {
    let main_rs = std::fs::read_to_string(crate_root().join("tests/acceptance/main.rs"))
        .expect("acceptance main readable");
    assert!(
        main_rs.contains("mod nomic_loads;"),
        "nomic_loads acceptance module should remain wired into the acceptance binary"
    );
    let tests = std::fs::read_to_string(crate_root().join("tests/acceptance/nomic_loads.rs"))
        .expect("nomic_loads source readable");
    for required in [
        "fn nomic_embed_text_adapter_implements_embedder_trait_with_declared_dimension_768_and_l2_normalized_output",
        "fn store_open_with_default_config_against_freshly_created_path_writes_cg_schema_version_row_matching_current_version_and_emits_no_legacy_or_migration_observable",
    ] {
        assert!(
            tests.contains(required),
            "missing acceptance test {required}"
        );
    }
    for retired in [
        "cg_main_crate_has_no_embeddinggemma_runtime_code_grep_audit_against_cargo_expanded_source_plus_no_embeddinggemma_transitive_dep_in_cargo_tree",
        "cargo_tree_for_context_graph_default_dep_graph_carries_no_native_dep_sys_crates_no_python_runtime_no_api_key_dep",
    ] {
        assert!(
            !tests.contains(retired),
            "unsupported audit remains registered"
        );
    }
}
