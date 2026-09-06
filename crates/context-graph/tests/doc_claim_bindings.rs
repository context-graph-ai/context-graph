//! The gate fails on a dead documentation claim or an unbound one.
//!
//! ## The promise
//!
//! `README.md`, a crate `README.md`, or a skill guide can say a behavior holds. If nothing
//! machine-checks that, the sentence rots the moment the behavior changes and nobody notices
//! until a user does. This test is that machine check, reimplemented here in plain Rust (no
//! subprocess, no network, no filesystem read outside this repository tree) so it can run inside
//! this crate's own guarded test set: `scripts/validate-doc-coverage.py` remains the tool a human
//! or CI runs directly with `--base`/`--changed-files` for the diff-scoped unbound-claim check
//! across the whole repository; this test enforces the subset of that same promise that holds at
//! every commit regardless of what it touched — dead references everywhere, and zero unbound
//! claims in `crates/context-graph/README.md` specifically, this crate's own front page.
//!
//! A documented claim is bound to a test with an HTML comment placed at (or immediately after) the
//! end of the claim's paragraph:
//!
//! ```text
//! ... prepared value carrying the exact final token count.<!-- enforced by: `some_test_name` -->
//! ```
//!
//! ## What is checked here
//!
//! - **Dead references**: every `<!-- enforced by: \`name\` -->` tag in `README.md`,
//!   `crates/*/README.md`, `docs/*.md` (if present) and `skills/**/SKILL.md` (if present) must
//!   name a live default-tier `#[test]`/`#[tokio::test]` function found under
//!   `crates/*/tests/**/*.rs` or `crates/*/src/**/*.rs`, including parameterized Tokio tests.
//!   A test-file stem is not a proof alias: it can survive after the test it once held is gone.
//! - **Unbound claims** (`crates/context-graph/README.md` only): a paragraph of at least 12
//!   whitespace-separated words, that is not a heading/table-row/bare-comment block, and carries
//!   no `enforced by` tag anywhere within it, is unbound and fails the gate — mirroring the
//!   Python validator's heuristic exactly (see its module doc for the same rules stated there).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// 1-based line spans (first, last) of gated inline `mod` blocks.
type GatedLineSpans = Vec<(usize, usize)>;
/// Excluded files and the gated block line spans within each compiled file.
type ModuleExclusions = (HashSet<PathBuf>, HashMap<PathBuf, GatedLineSpans>);

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels below the repository root")
        .to_path_buf()
}

/// Recursively collect every `.rs` file under `dir`, skipping `target` build directories (which
/// can contain generated `.rs` files under `OUT_DIR` that are not part of this repository's own
/// source tree).
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("target") {
                continue;
            }
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn crate_dirs(repo: &Path) -> Vec<PathBuf> {
    let crates_dir = repo.join("crates");
    let Ok(entries) = std::fs::read_dir(&crates_dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// Any attribute line at all, so `#[test]` followed by `#[ignore]`/`#[should_panic]` before the
/// `fn` still binds — mirrors the Python validator's `ATTR_LINE_RE`.
fn is_attr_line(line: &str) -> bool {
    line.trim_start().starts_with("#[")
}

/// Extract the function name from a line shaped like `fn some_name(` / `pub fn some_name<...`,
/// mirroring the Python validator's `FN_RE`. Returns `None` if the line is not a function
/// signature at all.
fn fn_name_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let trimmed = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
    let trimmed = trimmed.strip_prefix("async ").unwrap_or(trimmed);
    let rest = trimmed.strip_prefix("fn ")?;
    let end = rest.find(['(', '<'])?;
    let name = rest[..end].trim();
    if name.is_empty()
        || !name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
    {
        return None;
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return None;
    }
    Some(name.to_string())
}

/// Live default-tier test-function names mapped to the repo-relative files that define them, plus
/// the subset defined directly under a `tests/` tree (as opposed to `src/`).
struct TestNames {
    files_by_name: HashMap<String, Vec<String>>,
    test_tree_names: HashSet<String>,
    default_tier_function_names: HashSet<String>,
}

/// Strip a leading `pub` / `pub(...)` from an already-trimmed line, requiring the mandatory
/// whitespace the grammar demands between it and what follows — mirrors the Python validator's
/// `MOD_DECL_RE` optional-`pub` group exactly, including its `\s+` requirement. Returns `None`
/// when a `pub`/`pub(...)` prefix is present but not followed by whitespace (malformed, no match)
/// and `Some(trimmed)` unchanged when there is no `pub` prefix at all.
fn strip_pub_prefix(trimmed: &str) -> Option<&str> {
    let Some(rest) = trimmed.strip_prefix("pub") else {
        return Some(trimmed);
    };
    let rest = if let Some(inner) = rest.strip_prefix('(') {
        let close = inner.find(')')?;
        &inner[close + 1..]
    } else {
        rest
    };
    if rest.starts_with(char::is_whitespace) {
        Some(rest.trim_start())
    } else {
        None
    }
}

/// Parse a `mod name;` / `mod name { ... }` declaration (optionally `pub`/`pub(...)`-prefixed)
/// from an already-trimmed line, mirroring the Python validator's `MOD_DECL_RE`. Returns the
/// module name and the terminator (`;` for a file declaration, `{` for an inline block).
fn parse_mod_decl(trimmed: &str) -> Option<(String, char)> {
    let after_pub = strip_pub_prefix(trimmed)?;
    let rest = after_pub.strip_prefix("mod")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    let name = &rest[..end];
    let first = name.chars().next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    match rest[end..].trim_start().chars().next() {
        Some(term @ (';' | '{')) => Some((name.to_string(), term)),
        _ => None,
    }
}

/// Result of scanning one module file reached from a `tests/<target>/main.rs` root: the child
/// module files/dirs it declares (with whether each inherits or newly acquires the
/// `#[cfg(context_graph_internal_test_support)]` gate), and the inclusive 1-based line spans of
/// any gated inline `mod { ... }` blocks written directly inside this file.
struct ModuleScan {
    children: Vec<(PathBuf, PathBuf, bool)>,
    gated_ranges: GatedLineSpans,
}

/// Scan one Rust module file for `mod` structure, tracking brace depth to resolve inline module
/// blocks and following `#[cfg(context_graph_internal_test_support)]` gates onto both file-backed
/// (`mod x;`) and inline (`mod x { ... }`) declarations, at any nesting depth — mirrors the Python
/// validator's `_scan_module_file` exactly.
fn scan_module_file(path: &Path, module_dir: &Path, inherited_gate: bool) -> ModuleScan {
    let mut children = Vec::new();
    let mut gated_ranges = Vec::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return ModuleScan {
            children,
            gated_ranges,
        };
    };
    let lines: Vec<&str> = text.lines().collect();
    // (depth outside the block, module dir for its children, gated, first line)
    let mut inline_stack: Vec<(i64, PathBuf, bool, usize)> = Vec::new();
    let mut depth: i64 = 0;
    let mut pending_gate = false;

    for (index, line) in lines.iter().enumerate() {
        let lineno = index + 1;
        let trimmed = line.trim();
        let current_dir = inline_stack
            .last()
            .map(|(_, dir, _, _)| dir.clone())
            .unwrap_or_else(|| module_dir.to_path_buf());
        let inside_gate = inherited_gate || inline_stack.last().is_some_and(|(_, _, g, _)| *g);

        if trimmed == "#[cfg(context_graph_internal_test_support)]" {
            pending_gate = true;
            continue;
        }

        if let Some((name, term)) = parse_mod_decl(trimmed) {
            let gate = inside_gate || pending_gate;
            pending_gate = false;
            if term == ';' {
                for (candidate, child_dir) in [
                    (
                        current_dir.join(format!("{name}.rs")),
                        current_dir.join(&name),
                    ),
                    (
                        current_dir.join(&name).join("mod.rs"),
                        current_dir.join(&name),
                    ),
                ] {
                    if candidate.is_file() {
                        children.push((candidate, child_dir, gate));
                        break;
                    }
                }
            } else {
                inline_stack.push((depth, current_dir.join(&name), gate, lineno));
            }
        } else if !trimmed.is_empty() && !trimmed.starts_with("#[") {
            pending_gate = false;
        }

        depth += line.matches('{').count() as i64 - line.matches('}').count() as i64;
        while let Some(&(open_depth, _, gated, first_line)) = inline_stack.last() {
            if depth > open_depth {
                break;
            }
            inline_stack.pop();
            if gated {
                gated_ranges.push((first_line, lineno));
            }
        }
    }
    for (_, _, gated, first_line) in inline_stack {
        if gated {
            gated_ranges.push((first_line, lines.len()));
        }
    }
    ModuleScan {
        children,
        gated_ranges,
    }
}

/// `(files the Rust resolver never compiles into the default gate, gated inline-`mod`-block line
/// spans inside a compiled file)` for every multi-file test harness (`crates/*/tests/*/main.rs`)
/// under `repo` — walking the full `mod` chain from each harness root, at any depth, so a test
/// hidden behind a gated file declaration OR a gated inline `mod { ... }` block anywhere along
/// that chain is excluded. Mirrors the Python validator's `module_exclusions` exactly.
fn module_exclusions(repo: &Path) -> ModuleExclusions {
    let mut excluded_files: HashSet<PathBuf> = HashSet::new();
    let mut gated_blocks: HashMap<PathBuf, GatedLineSpans> = HashMap::new();

    for crate_dir in crate_dirs(repo) {
        let tests_dir = crate_dir.join("tests");
        let Ok(entries) = std::fs::read_dir(&tests_dir) else {
            continue;
        };
        let mut harness_dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        harness_dirs.sort();
        for harness_dir in harness_dirs {
            let main_rs = harness_dir.join("main.rs");
            if !main_rs.is_file() {
                continue;
            }
            let mut pending: Vec<(PathBuf, PathBuf, bool)> =
                vec![(main_rs, harness_dir.clone(), false)];
            let mut seen: HashSet<PathBuf> = HashSet::new();
            while let Some((path, module_dir, gate)) = pending.pop() {
                if !seen.insert(path.clone()) {
                    continue;
                }
                if gate {
                    excluded_files.insert(path.clone());
                }
                let scan = scan_module_file(&path, &module_dir, gate);
                if !scan.gated_ranges.is_empty() && !gate {
                    gated_blocks
                        .entry(path)
                        .or_default()
                        .extend(scan.gated_ranges);
                }
                pending.extend(scan.children);
            }
        }
    }
    (excluded_files, gated_blocks)
}

fn extract_test_names(repo: &Path) -> TestNames {
    let mut files_by_name: HashMap<String, Vec<String>> = HashMap::new();
    let mut test_tree_names: HashSet<String> = HashSet::new();
    let mut default_tier_function_names: HashSet<String> = HashSet::new();
    let (excluded_files, gated_blocks) = module_exclusions(repo);

    for crate_dir in crate_dirs(repo) {
        for sub in ["tests", "src"] {
            let root = crate_dir.join(sub);
            let mut files = Vec::new();
            collect_rs_files(&root, &mut files);
            for file in &files {
                if excluded_files.contains(file) {
                    continue;
                }
                let gated_spans = gated_blocks.get(file);
                let rel = file
                    .strip_prefix(repo)
                    .unwrap_or(file)
                    .to_string_lossy()
                    .replace('\\', "/");
                let is_test_tree = rel.contains("/tests/") || rel.starts_with("tests/");
                let Ok(text) = std::fs::read_to_string(file) else {
                    continue;
                };
                let mut attributes = Vec::new();
                for (index, line) in text.lines().enumerate() {
                    let lineno = index + 1;
                    if gated_spans.is_some_and(|spans| {
                        spans
                            .iter()
                            .any(|&(first, last)| (first..=last).contains(&lineno))
                    }) {
                        attributes.clear();
                        continue;
                    }
                    if is_attr_line(line) {
                        attributes.push(line.trim().to_string());
                        continue;
                    }
                    if !attributes.is_empty() {
                        if let Some(name) = fn_name_from_line(line) {
                            let is_test = attributes.iter().any(|attribute| {
                                attribute == "#[test]"
                                    || attribute == "#[tokio::test]"
                                    || attribute.starts_with("#[tokio::test(")
                            });
                            let excluded_from_default_tier = attributes.iter().any(|attribute| {
                                attribute.starts_with("#[ignore")
                                    || attribute == "#[cfg(context_graph_internal_test_support)]"
                            });
                            if is_test && !excluded_from_default_tier {
                                files_by_name
                                    .entry(name.clone())
                                    .or_default()
                                    .push(rel.clone());
                                if is_test_tree {
                                    test_tree_names.insert(name.clone());
                                }
                                default_tier_function_names.insert(name);
                            }
                            attributes.clear();
                        } else if !line.trim().is_empty() {
                            attributes.clear();
                        }
                    }
                }
            }
        }
    }

    TestNames {
        files_by_name,
        test_tree_names,
        default_tier_function_names,
    }
}

fn all_doc_paths(repo: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let root_readme = repo.join("README.md");
    if root_readme.is_file() {
        paths.push(root_readme);
    }
    for crate_dir in crate_dirs(repo) {
        let readme = crate_dir.join("README.md");
        if readme.is_file() {
            paths.push(readme);
        }
    }
    let docs_dir = repo.join("docs");
    if let Ok(entries) = std::fs::read_dir(&docs_dir) {
        let mut docs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
            .collect();
        docs.sort();
        paths.extend(docs);
    }
    let skills_dir = repo.join("skills");
    if skills_dir.is_dir() {
        let mut skill_files = Vec::new();
        collect_named_files(&skills_dir, "SKILL.md", &mut skill_files);
        skill_files.sort();
        paths.extend(skill_files);
    }
    paths
}

fn collect_named_files(dir: &Path, name: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_named_files(&path, name, out);
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            out.push(path);
        }
    }
}

/// Extract the inner text of every `<!-- enforced by: ... -->` tag on `line`, without a regex
/// crate dependency: a small manual scan for the literal marker.
fn enforced_by_tag_bodies(line: &str) -> Vec<String> {
    const MARKER: &str = "enforced by:";
    let mut bodies = Vec::new();
    let mut search_from = 0usize;
    while let Some(open_rel) = line[search_from..].find("<!--") {
        let open = search_from + open_rel;
        let Some(close_rel) = line[open..].find("-->") else {
            break;
        };
        let close = open + close_rel;
        let inner = &line[open + 4..close];
        if let Some(marker_pos) = inner.find(MARKER) {
            bodies.push(inner[marker_pos + MARKER.len()..].trim().to_string());
        }
        search_from = close + 3;
    }
    bodies
}

/// Extract every `name`-shaped token (optionally backtick-wrapped) from an `enforced by:` tag
/// body — a body can name more than one test.
fn names_in_tag_body(body: &str) -> Vec<String> {
    let cleaned = body.trim().trim_matches('`');
    cleaned
        .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'))
        .filter(|token| {
            token.len() > 1
                && token
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        })
        .map(str::to_string)
        .collect()
}

/// `(name, 1-based line number)` for every enforced-by reference in `text`.
fn doc_refs(text: &str) -> Vec<(String, usize)> {
    let mut refs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        for body in enforced_by_tag_bodies(line) {
            for name in names_in_tag_body(&body) {
                refs.push((name, index + 1));
            }
        }
    }
    refs
}

/// Every `enforced by` reference in `repo`'s documentation that names no test function the
/// repository actually carries, as `(name, doc file, line)`. A dead reference is a published
/// promise that some test protects a sentence when nothing does.
fn dead_doc_references(repo: &Path, test_names: &TestNames) -> Vec<(String, String, usize)> {
    let mut dead: Vec<(String, String, usize)> = Vec::new();
    for doc_path in all_doc_paths(repo) {
        let rel = doc_path
            .strip_prefix(repo)
            .unwrap_or(&doc_path)
            .to_string_lossy()
            .to_string();
        let Ok(text) = std::fs::read_to_string(&doc_path) else {
            continue;
        };
        for (name, lineno) in doc_refs(&text) {
            if !test_names.files_by_name.contains_key(&name) {
                dead.push((name, rel.clone(), lineno));
            }
        }
    }
    dead
}

#[test]
fn doc_claims_have_no_dead_reference() {
    let repo = repository_root();
    let test_names = extract_test_names(&repo);
    assert!(
        !test_names.test_tree_names.is_empty(),
        "test discovery must find at least one test-tree test; an empty result means the \
         repository-relative walk found nothing, not that there truly are zero tests"
    );
    let dead = dead_doc_references(&repo, &test_names);

    assert!(
        dead.is_empty(),
        "the following documented `enforced by` references name no real test function \
         (dead reference): {}",
        dead.iter()
            .map(|(name, file, line)| format!("{name} ({file}:{line})"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
fn documented_claim_requires_a_live_default_tier_test_function() {
    let repo = repository_root();
    let test_names = extract_test_names(&repo);
    let mut non_live = Vec::new();

    for doc_path in all_doc_paths(&repo) {
        let rel = doc_path
            .strip_prefix(&repo)
            .unwrap_or(&doc_path)
            .to_string_lossy()
            .to_string();
        let text = std::fs::read_to_string(&doc_path)
            .unwrap_or_else(|err| panic!("reading {}: {err}", doc_path.display()));
        for (name, line) in doc_refs(&text) {
            if !test_names.default_tier_function_names.contains(&name) {
                non_live.push(format!("{name} ({rel}:{line})"));
            }
        }
    }

    assert!(
        non_live.is_empty(),
        "a documented claim must name a live default-tier Rust test function, not a test-file \
         stem, an ignored/internal-only test, or a test whose enclosing module never compiles \
         into the default gate: {}",
        non_live.join(", ")
    );
}

#[test]
fn implemented_event_search_errors_are_not_documented_as_skeletons() {
    let repo = repository_root();
    let error_path = repo.join("crates/context-graph/src/error.rs");
    let error_source = std::fs::read_to_string(&error_path)
        .unwrap_or_else(|err| panic!("reading {}: {err}", error_path.display()));
    let implementation_root = repo.join("crates/context-graph/src");
    let mut implementation_files = Vec::new();
    collect_rs_files(&implementation_root, &mut implementation_files);

    for variant in [
        "EventHistoryEmbedderConstructionRefused",
        "EventHistoryPlainEmbedRequiresExplicitRole",
        "EventSearchPreparedValueInvalid",
        "EventHistoryProfileIdentityUnverified",
        "PreparedInputExceedsModelLimit",
    ] {
        let implementation_uses = implementation_files
            .iter()
            .filter(|path| path.as_path() != error_path.as_path())
            .filter_map(|path| std::fs::read_to_string(path).ok())
            .map(|source| source.matches(&format!("CgError::{variant}")).count())
            .sum::<usize>();
        assert!(
            implementation_uses > 0,
            "{variant} must have a live context-graph implementation use before this proof \
             classifies its public documentation"
        );

        let variant_offset = error_source
            .find(variant)
            .unwrap_or_else(|| panic!("{variant} must be declared in {}", error_path.display()));
        let mut documentation_start = variant_offset.saturating_sub(1_200);
        while !error_source.is_char_boundary(documentation_start) {
            documentation_start -= 1;
        }
        let mut documentation_end = (variant_offset + variant.len() + 320).min(error_source.len());
        while !error_source.is_char_boundary(documentation_end) {
            documentation_end -= 1;
        }
        let documentation = &error_source[documentation_start..documentation_end];
        assert!(
            !documentation.contains("Skeleton variant"),
            "{variant} has live implementation use but its public error documentation still \
             calls it a Skeleton variant"
        );
    }
}

fn known_failing_test_names(repo: &Path) -> HashSet<String> {
    let manifest = repo.join("crates/context-graph/tests/fixtures/known_failing_tests.txt");
    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|err| {
        panic!(
            "reading known-failing manifest {}: {err}",
            manifest.display()
        )
    });
    text.lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .filter_map(|line| line.rsplit("::").next())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn documented_claim_cannot_name_a_known_failing_test() {
    let repo = repository_root();
    let known_failing = known_failing_test_names(&repo);
    let mut published_failures = Vec::new();

    for doc_path in all_doc_paths(&repo) {
        let rel = doc_path
            .strip_prefix(&repo)
            .unwrap_or(&doc_path)
            .to_string_lossy()
            .to_string();
        let text = std::fs::read_to_string(&doc_path)
            .unwrap_or_else(|err| panic!("reading {}: {err}", doc_path.display()));
        for (name, line) in doc_refs(&text) {
            if known_failing.contains(&name) {
                published_failures.push(format!("{name} ({rel}:{line})"));
            }
        }
    }

    assert!(
        published_failures.is_empty(),
        "a published proof tag cannot name a test in the known-failing manifest: {}",
        published_failures.join(", ")
    );
}

/// Yield `(1-based start line, block text)` for each blank-line-delimited block of `text`, never
/// looking inside a fenced (```` ``` ````) code block — mirrors the Python validator's
/// `claim_paragraphs`.
fn claim_blocks(text: &str) -> Vec<(usize, String)> {
    let mut blocks = Vec::new();
    let mut block_lines: Vec<&str> = Vec::new();
    let mut block_start = 0usize;
    let mut in_fence = false;

    let flush =
        |block_lines: &mut Vec<&str>, block_start: usize, blocks: &mut Vec<(usize, String)>| {
            if !block_lines.is_empty() {
                blocks.push((block_start, block_lines.join("\n")));
                block_lines.clear();
            }
        };

    for (index, line) in text.lines().enumerate() {
        let lineno = index + 1;
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            flush(&mut block_lines, block_start, &mut blocks);
            continue;
        }
        if in_fence {
            continue;
        }
        if trimmed.is_empty() {
            flush(&mut block_lines, block_start, &mut blocks);
            continue;
        }
        if block_lines.is_empty() {
            block_start = lineno;
        }
        block_lines.push(line);
    }
    flush(&mut block_lines, block_start, &mut blocks);
    blocks
}

/// Strip every `<!-- enforced by: ... -->` tag out of `block`, mirroring the Python validator's
/// `ENFORCED_RE.sub("", block_text)` used by its word-count heuristic.
fn strip_enforced_by_tags(block: &str) -> String {
    let mut result = String::with_capacity(block.len());
    let mut search_from = 0usize;
    loop {
        let Some(open_rel) = block[search_from..].find("<!--") else {
            result.push_str(&block[search_from..]);
            break;
        };
        let open = search_from + open_rel;
        result.push_str(&block[search_from..open]);
        let Some(close_rel) = block[open..].find("-->") else {
            result.push_str(&block[open..]);
            break;
        };
        let close = open + close_rel + 3;
        search_from = close;
    }
    result
}

fn is_claim_block(block: &str) -> bool {
    let first = block.trim_start();
    if first.starts_with('#') || first.starts_with('|') || first.starts_with("<!--") {
        return false;
    }
    let stripped = strip_enforced_by_tags(block);
    stripped.split_whitespace().count() >= 12
}

fn is_bound_block(block: &str) -> bool {
    block.contains("enforced by:")
}

/// Every behaviour paragraph in `text` — twelve or more words, not a heading, table row or
/// comment — that names no test binding at all, as `(line, preview)`. An unbound claim is a
/// published behaviour sentence nothing fails on when it stops being true.
fn unbound_claim_blocks(text: &str) -> Vec<(usize, String)> {
    claim_blocks(text)
        .into_iter()
        .filter(|(_, block)| is_claim_block(block) && !is_bound_block(block))
        .map(|(lineno, block)| {
            let preview: String = block.split_whitespace().collect::<Vec<_>>().join(" ");
            let preview = preview.chars().take(60).collect::<String>();
            (lineno, preview)
        })
        .collect()
}

#[test]
fn context_graph_readme_carries_no_unbound_behavior_claim() {
    let repo = repository_root();
    let readme_path = repo.join("crates/context-graph/README.md");
    let text = std::fs::read_to_string(&readme_path).unwrap_or_else(|err| {
        panic!(
            "expected the context-graph crate README at {}: {err}",
            readme_path.display()
        )
    });

    let unbound = unbound_claim_blocks(&text);

    assert!(
        unbound.is_empty(),
        "crates/context-graph/README.md carries {} unbound behavior claim(s) — a paragraph of \
         12+ words with no `enforced by` test binding anywhere in it — starting at: {}",
        unbound.len(),
        unbound
            .iter()
            .map(|(lineno, preview)| format!("{lineno}: {preview}..."))
            .collect::<Vec<_>>()
            .join(" | ")
    );
}

/// A fixture-driven proof of the same rule
/// [`documented_claim_requires_a_live_default_tier_test_function`] now enforces on the real repo:
/// an ordinary `#[test]` function hidden by an enclosing `#[cfg(context_graph_internal_test_support)]
/// mod` declaration in its harness's `main.rs` must not count as default-tier, while a sibling test
/// reachable through an ungated `mod` declaration must. Built under a synthetic `crates/<name>/`
/// tree so it exercises `extract_test_names`/`module_exclusions` exactly the way
/// they walk the real repository, without depending on any specific test file this crate happens
/// to carry today.
#[test]
fn a_test_hidden_by_an_enclosing_module_cfg_is_not_counted_as_default_tier() {
    let fixture_repo = tempfile::tempdir().expect("fixture repo tempdir");
    let crate_dir = fixture_repo.path().join("crates").join("fixture_crate");
    let harness_dir = crate_dir.join("tests").join("integration");
    std::fs::create_dir_all(&harness_dir).expect("create fixture harness dir");

    std::fs::write(
        harness_dir.join("main.rs"),
        "#[cfg(context_graph_internal_test_support)]\nmod hidden;\nmod visible;\n",
    )
    .expect("write fixture main.rs");
    std::fs::write(
        harness_dir.join("hidden.rs"),
        "#[test]\nfn ordinary_test_inside_a_cfg_excluded_module() {}\n",
    )
    .expect("write fixture hidden.rs");
    std::fs::write(
        harness_dir.join("visible.rs"),
        "#[test]\nfn ordinary_test_inside_a_reachable_module() {}\n",
    )
    .expect("write fixture visible.rs");

    let test_names = extract_test_names(fixture_repo.path());

    assert!(
        test_names
            .default_tier_function_names
            .contains("ordinary_test_inside_a_reachable_module"),
        "a test inside a module reachable through an ungated `mod` declaration must be counted \
         as default-tier; default-tier names observed: {:?}",
        test_names.default_tier_function_names
    );

    // RED: the Rust resolver never compiles a test inside a module declared under an enclosing
    // `#[cfg(context_graph_internal_test_support)] mod` declaration, so it must never be counted
    // as default-tier — but `extract_test_names` decides purely from the test function's own
    // attributes today, and this function's own attributes carry no exclusion marker at all, so it
    // is counted anyway.
    assert!(
        !test_names
            .default_tier_function_names
            .contains("ordinary_test_inside_a_cfg_excluded_module"),
        "a test hidden behind an enclosing #[cfg(context_graph_internal_test_support)] mod \
         declaration in its harness's main.rs must never be counted as default-tier, even though \
         the test function's own attributes carry no exclusion marker; extract_test_names counted \
         it anyway: {:?}",
        test_names.default_tier_function_names
    );
}

/// Both documentation checks are stated as "the gate fails on X" — a dead `enforced by` reference,
/// and a behaviour paragraph in a touched doc with no test named. Nothing anywhere had ever fed
/// either detector an X and watched it fail: both claims rested on the real tree happening to
/// contain neither. A detector that silently reported nothing would satisfy every one of those
/// runs, and the published promise that a stale documentation claim stops the change would be
/// false with no test failing.
///
/// So this builds a repository that carries exactly one of each defect and requires both detectors
/// to name them, then removes the defects and requires the same detectors to report nothing. The
/// fixture is a synthetic `crates/<name>/` tree, so the detectors walk it the same way they walk
/// the real repository.
#[test]
fn the_documentation_binding_checks_report_a_dead_reference_and_an_unbound_claim() {
    let fixture_repo = tempfile::tempdir().expect("fixture repo tempdir");
    let crate_dir = fixture_repo.path().join("crates").join("fixture_crate");
    let tests_dir = crate_dir.join("tests");
    std::fs::create_dir_all(&tests_dir).expect("create fixture tests dir");
    std::fs::write(
        tests_dir.join("behaviour.rs"),
        "#[test]\nfn the_store_refuses_a_second_writer() {}\n",
    )
    .expect("write fixture test file");

    let test_names = extract_test_names(fixture_repo.path());
    assert!(
        test_names
            .files_by_name
            .contains_key("the_store_refuses_a_second_writer"),
        "the fixture's one real test must be discovered, otherwise every reference below would \
         look dead for the wrong reason; discovered: {:?}",
        test_names.files_by_name.keys().collect::<Vec<_>>()
    );

    let readme_with_defects = "\
# Fixture crate

Opening the store a second time while another writer holds it is refused with a typed error
naming the process that holds it.
<!-- enforced by: `the_store_refuses_a_second_writer` -->

Closing the store flushes every pending write before the handle is released, so a reader that
opens afterwards never observes a partial commit.

Reopening the store after a crash replays the write-ahead log before answering any read.
<!-- enforced by: `the_store_replays_the_write_ahead_log_after_a_crash` -->
";
    std::fs::write(crate_dir.join("README.md"), readme_with_defects)
        .expect("write fixture README with defects");

    let dead = dead_doc_references(fixture_repo.path(), &test_names);
    assert_eq!(
        dead.len(),
        1,
        "a documentation reference naming a test the repository does not carry must be reported \
         exactly once, got: {dead:?}"
    );
    assert_eq!(
        dead[0].0, "the_store_replays_the_write_ahead_log_after_a_crash",
        "the reported dead reference must be the one naming no real test, got: {dead:?}"
    );
    assert!(
        dead[0].1.ends_with("README.md"),
        "the dead reference must be reported against the documentation file that carries it, \
         got: {dead:?}"
    );

    let unbound = unbound_claim_blocks(readme_with_defects);
    assert_eq!(
        unbound.len(),
        1,
        "a behaviour paragraph with no test binding must be reported exactly once, and the two \
         bound paragraphs must not be, got: {unbound:?}"
    );
    assert!(
        unbound[0].1.starts_with("Closing the store flushes"),
        "the reported unbound claim must be the paragraph carrying no binding, got: {unbound:?}"
    );

    let readme_repaired = "\
# Fixture crate

Opening the store a second time while another writer holds it is refused with a typed error
naming the process that holds it.
<!-- enforced by: `the_store_refuses_a_second_writer` -->

Closing the store flushes every pending write before the handle is released, so a reader that
opens afterwards never observes a partial commit.
<!-- enforced by: `the_store_refuses_a_second_writer` -->
";
    std::fs::write(crate_dir.join("README.md"), readme_repaired)
        .expect("write repaired fixture README");

    assert!(
        dead_doc_references(fixture_repo.path(), &test_names).is_empty(),
        "documentation whose every reference names a real test must report no dead reference"
    );
    assert!(
        unbound_claim_blocks(readme_repaired).is_empty(),
        "documentation whose every behaviour paragraph carries a binding must report no unbound \
         claim"
    );
}

// ---------------------------------------------------------------------------------------------
// A changed crate carries a tested agent guide
// ---------------------------------------------------------------------------------------------

/// The crate name a changed-file path like `crates/cg-catalog/src/lib.rs` belongs to, or `None`
/// for a path outside `crates/*/`.
fn crate_name_of_changed_path(path: &str) -> Option<&str> {
    path.strip_prefix("crates/")?.split('/').next()
}

/// Pure reimplementation of the rule `scripts/validate-doc-coverage.py` will enforce on a
/// changed-file list: any crate whose `src/` a change touches must carry `crates/<crate>/AGENTS.md`.
/// Returns the sorted set of crate names that are missing their guide. A crate named more than
/// once in `changed_files` is reported once.
fn crates_missing_an_agents_guide(
    changed_files: &[&str],
    existing_agents_md_crates: &std::collections::HashSet<&str>,
) -> Vec<String> {
    let mut missing: HashSet<String> = HashSet::new();
    for path in changed_files {
        let Some(crate_name) = crate_name_of_changed_path(path) else {
            continue;
        };
        if !path[("crates/".len() + crate_name.len())..].starts_with("/src/") {
            continue;
        }
        if !existing_agents_md_crates.contains(crate_name) {
            missing.insert(crate_name.to_string());
        }
    }
    let mut missing: Vec<String> = missing.into_iter().collect();
    missing.sort();
    missing
}

/// Crate names whose `src/` a change touched while leaving `crates/<crate>/AGENTS.md` out of the
/// same change, sorted, minus the crates an owner declared exempt.
///
/// The Rust mirror of `crates_with_a_stale_agents_guide` in `scripts/validate-doc-coverage.py`;
/// the two implement the same rule and are updated together.
fn crates_with_a_stale_agents_guide(
    changed_files: &[&str],
    existing_agents_md_crates: &std::collections::HashSet<&str>,
    declared_exceptions: &std::collections::HashSet<&str>,
) -> Vec<String> {
    let mut stale: HashSet<String> = HashSet::new();
    for path in changed_files {
        let Some(crate_name) = crate_name_of_changed_path(path) else {
            continue;
        };
        if !path[("crates/".len() + crate_name.len())..].starts_with("/src/") {
            continue;
        }
        if declared_exceptions.contains(crate_name) {
            continue;
        }
        // A crate with no guide at all is the missing-guide rule's finding, not this one's.
        if !existing_agents_md_crates.contains(crate_name) {
            continue;
        }
        let guide = format!("crates/{crate_name}/AGENTS.md");
        if !changed_files.contains(&guide.as_str()) {
            stale.insert(crate_name.to_string());
        }
    }
    let mut stale: Vec<String> = stale.into_iter().collect();
    stale.sort();
    stale
}

/// A crate guide states that crate's invariants and names the tests behind them, so a change that
/// rewrites the crate's source and leaves the guide untouched leaves an authoritative-looking
/// description of behavior that is gone. Existence alone never catches that: the guide is still
/// there, and the gate says PASS. This drives the staleness rule over a changed-file list carrying
/// each case — source moved without its guide, source moved with its guide, a file outside `src/`,
/// a crate with no guide at all, and a declared exception.
#[test]
fn a_crate_whose_source_changed_without_its_agents_md_is_reported_stale() {
    let guided: HashSet<&str> = ["cg-catalog", "context-graph", "cg-cli"]
        .into_iter()
        .collect();
    let no_exceptions: HashSet<&str> = HashSet::new();

    let changed_files = [
        "crates/cg-catalog/src/lib.rs",
        "crates/context-graph/src/store/mod.rs",
        "crates/context-graph/AGENTS.md",
        "crates/cg-cli/tests/cli_acceptance.rs",
    ];
    let stale = crates_with_a_stale_agents_guide(&changed_files, &guided, &no_exceptions);
    assert_eq!(
        stale,
        vec!["cg-catalog".to_string()],
        "only the crate whose src/ changed with its guide untouched may be reported:          context-graph changed both, cg-cli changed nothing under src/; got {stale:?}"
    );

    let repaired = [
        "crates/cg-catalog/src/lib.rs",
        "crates/cg-catalog/AGENTS.md",
        "crates/context-graph/src/store/mod.rs",
        "crates/context-graph/AGENTS.md",
    ];
    assert!(
        crates_with_a_stale_agents_guide(&repaired, &guided, &no_exceptions).is_empty(),
        "a change that moves each touched crate's guide with its source must report nothing"
    );

    let unguided: HashSet<&str> = ["context-graph"].into_iter().collect();
    assert!(
        crates_with_a_stale_agents_guide(&changed_files, &unguided, &no_exceptions).is_empty(),
        "a crate with no guide at all is the missing-guide rule's finding; reporting it here too          would name the same crate twice for two different repairs"
    );

    let excepted: HashSet<&str> = ["cg-catalog"].into_iter().collect();
    assert!(
        crates_with_a_stale_agents_guide(&changed_files, &guided, &excepted).is_empty(),
        "a crate declared in the exceptions fixture must not be reported"
    );
}

/// The exceptions fixture the staleness rule reads is a tracked file whose every line names a real
/// crate, so an exemption cannot silently outlive the crate it was written for.
#[test]
fn every_declared_guide_staleness_exception_names_a_crate_in_the_tree() {
    let root = repository_root();
    let fixture =
        root.join("crates/context-graph/tests/fixtures/agents_guide_staleness_exceptions.txt");
    let text = std::fs::read_to_string(&fixture)
        .unwrap_or_else(|error| panic!("reading {}: {error}", fixture.display()));
    for line in text.lines() {
        let name = line.split('#').next().unwrap_or("").trim();
        if name.is_empty() {
            continue;
        }
        assert!(
            root.join("crates").join(name).join("Cargo.toml").exists(),
            "{} names {name}, which is not a crate in this tree; an exemption for a crate that no              longer exists is a rule nobody is following any more",
            fixture.display()
        );
    }
}

#[test]
fn a_changed_crate_with_no_agents_md_is_reported_missing_and_a_guided_one_is_not() {
    let changed_files = [
        "crates/cg-catalog/src/lib.rs",
        "crates/cg-catalog/src/model_connection.rs",
        "crates/context-graph/src/store/mod.rs",
        "crates/cg-cli/tests/cli_acceptance.rs",
    ];

    // context-graph already has a guide; cg-cli's only changed file is under `tests/`, which is
    // not this rule's concern; cg-catalog's src/ changed and has no guide, so it alone is
    // reported.
    let context_graph_guided: HashSet<&str> = ["context-graph"].into_iter().collect();
    let missing = crates_missing_an_agents_guide(&changed_files, &context_graph_guided);
    assert_eq!(
        missing,
        vec!["cg-catalog".to_string()],
        "a crate whose src/ changed with no AGENTS.md must be reported exactly once, a crate \
         whose only changed file is outside src/ must not be reported, and a crate that already \
         has a guide must not be reported: got {missing:?}"
    );

    // With the guide present, the same changed-file list passes clean.
    let all_guided: HashSet<&str> = ["cg-catalog", "context-graph"].into_iter().collect();
    let missing_once_guided = crates_missing_an_agents_guide(&changed_files, &all_guided);
    assert!(
        missing_once_guided.is_empty(),
        "a crate whose AGENTS.md exists must never be reported missing: got \
         {missing_once_guided:?}"
    );

    // A crate's src/ file appearing more than once in the changed-file list is reported once, not
    // once per file.
    let repeated_changed_files = [
        "crates/cg-catalog/src/lib.rs",
        "crates/cg-catalog/src/model_connection.rs",
    ];
    let no_guides: HashSet<&str> = HashSet::new();
    let missing_repeated = crates_missing_an_agents_guide(&repeated_changed_files, &no_guides);
    assert_eq!(
        missing_repeated,
        vec!["cg-catalog".to_string()],
        "cg-catalog must be reported exactly once even though two of its src/ files changed: \
         got {missing_repeated:?}"
    );
}
