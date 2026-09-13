//! Every file and directory the root `AGENTS.md` names must actually be there.
//!
//! ## The promise
//!
//! `AGENTS.md` is what a contributor — a person, or a model small enough that reading the whole
//! tree is not an option — is handed to find the owning crate and the rule it must not break. Its
//! whole value is that following a pointer lands somewhere. A rename or a module split that leaves
//! the pointers behind does not announce itself: the file still reads as authoritative while
//! sending its reader to a path that is gone, and the reader's next move is to invent a place for
//! the change instead. So the pointers are checked by machine, in the same test gate as the code
//! they point at.
//!
//! ## What is checked
//!
//! Existence only. Every backticked token in `AGENTS.md` that is shaped like a repository path — it
//! contains a separator and either ends in a directory slash or carries a source-file extension —
//! must resolve under the repository root. Nothing here reads what is inside those files: what they
//! SAY is the job of the tests `AGENTS.md` cites, and a content check here would only be a second,
//! staler copy of them.

use std::path::{Path, PathBuf};

/// Extensions that make a token with a separator in it a file this repository tracks.
const SOURCE_EXTENSIONS: [&str; 7] = [".rs", ".toml", ".md", ".yml", ".yaml", ".sh", ".py"];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels below the repository root")
        .to_path_buf()
}

/// `crates/cg-cli/src/main.rs:178-181` is a citation of the file at `crates/cg-cli/src/main.rs`.
/// Line numbers move on their own and are not what this test is about.
fn without_line_citation(token: &str) -> &str {
    match token.rsplit_once(':') {
        Some((path, tail))
            if !tail.is_empty()
                && tail
                    .chars()
                    .all(|character| character.is_ascii_digit() || character == '-') =>
        {
            path
        }
        _ => token,
    }
}

fn looks_like_a_repository_path(token: &str) -> bool {
    if !token.contains('/') || token.contains("://") {
        return false;
    }
    // A placeholder (`<data_root>/cg/fabric.toml`), a glob, or anything with whitespace in it is
    // prose about a shape, not a pointer at one file.
    if token
        .chars()
        .any(|character| matches!(character, '<' | '>' | '*' | ' ' | '\t'))
    {
        return false;
    }
    token.ends_with('/')
        || SOURCE_EXTENSIONS
            .iter()
            .any(|extension| token.ends_with(extension))
}

/// Every backticked span in `text`. `AGENTS.md` names each path inside backticks, which is also
/// what keeps an ordinary English sentence from being mistaken for a pointer.
fn backticked_spans(text: &str) -> Vec<&str> {
    text.split('`')
        .enumerate()
        .filter_map(|(index, span)| (index % 2 == 1).then_some(span))
        .collect()
}

#[test]
fn every_path_the_root_agents_file_names_is_in_the_tree() {
    let root = repository_root();
    let agents_path = root.join("AGENTS.md");
    let agents = std::fs::read_to_string(&agents_path).unwrap_or_else(|error| {
        panic!(
            "the repository's contributor routing file must be tracked at {}: {error}",
            agents_path.display()
        )
    });

    let mut checked = Vec::new();
    for span in backticked_spans(&agents) {
        let candidate = without_line_citation(span);
        if !looks_like_a_repository_path(candidate) {
            continue;
        }
        let trimmed = candidate.trim_end_matches('/');
        assert!(
            root.join(trimmed).exists(),
            "AGENTS.md sends a contributor to `{candidate}`, which is not in the tree. A pointer \
             that no longer resolves is worse than no pointer: the reader invents a place for the \
             change instead. Repoint it or drop it."
        );
        checked.push(candidate.to_string());
    }

    assert!(
        checked.len() >= 20,
        "AGENTS.md earns its keep by ROUTING — naming the crate directories and the files that own \
         each rule. Only {} path(s) were found to check ({checked:?}), which means the routing was \
         removed or rewritten into prose that points nowhere.",
        checked.len()
    );
}

/// The crate table routes a contributor to the crate that owns a change, so a crate missing from
/// it is unreachable through the one document a fresh reader is handed — and a row naming a
/// directory that is gone sends them nowhere. Neither shows up in the existence walk above: that
/// one reads the guide and checks the tree, and nothing reads the tree and checks the guide.
#[test]
fn the_root_crate_table_names_every_crate_in_the_tree_and_no_directory_that_is_not_one() {
    let root = repository_root();
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).expect("root AGENTS.md readable");

    let mut named: Vec<String> = Vec::new();
    for span in backticked_spans(&agents) {
        let candidate = without_line_citation(span);
        let Some(rest) = candidate.strip_prefix("crates/") else {
            continue;
        };
        let Some(name) = rest.strip_suffix('/') else {
            continue;
        };
        if name.is_empty() || name.contains('/') || named.iter().any(|seen| seen == name) {
            continue;
        }
        named.push(name.to_string());
    }
    named.sort();

    let mut in_tree: Vec<String> = std::fs::read_dir(root.join("crates"))
        .expect("crates directory readable")
        .filter_map(|entry| {
            let entry = entry.expect("crates directory entry readable");
            let path = entry.path();
            (path.join("Cargo.toml").exists())
                .then(|| entry.file_name().to_string_lossy().into_owned())
        })
        .collect();
    in_tree.sort();

    let unlisted: Vec<&String> = in_tree
        .iter()
        .filter(|name| !named.contains(name))
        .collect();
    assert!(
        unlisted.is_empty(),
        "the crate table in AGENTS.md names no row for {unlisted:?}. A contributor routed by that          table cannot reach a crate it does not mention, so a new crate joins the table in the          change that adds it. Named: {named:?}"
    );

    let phantom: Vec<&String> = named
        .iter()
        .filter(|name| !in_tree.contains(name))
        .collect();
    assert!(
        phantom.is_empty(),
        "the crate table in AGENTS.md names {phantom:?}, which is not a crate in `crates/`. A row          for a crate that is gone routes a reader to nothing. In the tree: {in_tree:?}"
    );
}

/// The three crates a contributor is routed to by name for the work that arrives most often —
/// store and ontology behavior, a command or its output, a model registration or connection —
/// each carry their own guide. A contributor sent to one of them finds the invariants of the crate
/// they are about to edit in that crate, not only in the root guide, because a rule kept one level
/// away from the code it governs is a rule a reader has to already know exists.
#[test]
fn the_crates_a_contributor_is_routed_to_by_name_carry_their_own_agent_guide() {
    let root = repository_root();
    for crate_name in ["context-graph", "cg-catalog", "cg-cli"] {
        let path = root.join("crates").join(crate_name).join("AGENTS.md");
        assert!(
            path.exists(),
            "crates/{crate_name}/AGENTS.md must exist: a contributor to {crate_name} deserves the \
             same routing and invariant guide a contributor to context-graph already has at \
             crates/context-graph/AGENTS.md"
        );
    }
}

/// The pointer-existence rule this file enforces for the root `AGENTS.md` must hold for every
/// crate guide, not only the root one — a crate guide's own stale pointer is exactly as costly to
/// its reader.
#[test]
fn every_path_every_crate_agents_file_names_is_in_the_tree() {
    let root = repository_root();
    let crates_dir = root.join("crates");
    let mut total_checked = 0usize;
    for entry in std::fs::read_dir(&crates_dir)
        .expect("crates/ directory must exist")
        .flatten()
    {
        let crate_dir = entry.path();
        if !crate_dir.is_dir() {
            continue;
        }
        let agents_path = crate_dir.join("AGENTS.md");
        if !agents_path.exists() {
            continue;
        }
        let contents = std::fs::read_to_string(&agents_path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", agents_path.display()));
        for span in backticked_spans(&contents) {
            let candidate = without_line_citation(span);
            if !looks_like_a_repository_path(candidate) {
                continue;
            }
            let trimmed = candidate.trim_end_matches('/');
            // A crate guide names its own paths crate-relative (`src/embedder.rs`) or
            // repository-relative (`crates/context-graph/src/embedder.rs`); either resolves.
            let resolves = crate_dir.join(trimmed).exists() || root.join(trimmed).exists();
            assert!(
                resolves,
                "{} sends a contributor to `{candidate}`, which is not in the tree (checked \
                 relative to the crate and to the repository root). Repoint it or drop it.",
                agents_path.display()
            );
            total_checked += 1;
        }
    }
    assert!(
        total_checked > 0,
        "at least one crate AGENTS.md must exist and name at least one repository path to check \
         (found {total_checked}) — every crate this change touches gets a guide that routes to \
         real files"
    );
}
