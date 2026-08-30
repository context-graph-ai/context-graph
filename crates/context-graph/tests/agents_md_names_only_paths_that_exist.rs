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
