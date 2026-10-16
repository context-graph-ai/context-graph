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
