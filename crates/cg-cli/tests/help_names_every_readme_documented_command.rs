//! `cg --help` must name every operator command that `crates/context-graph/README.md` documents,
//! or a reader who trusts the README to find the command, then checks help to learn how to run it,
//! finds nothing there.
//!
//! ## The promise
//!
//! AGENTS.md's "Docs ship with the behavior" section names the public operator surface as
//! `crates/context-graph/README.md`, and requires the help text in `crates/cg-cli/src/main.rs` to
//! agree with it. Agreement runs from the README outward: whatever the README documents as a
//! command a reader can type, `cg --help` must list. It is not the reverse — `cg` dispatches
//! plenty of internal or not-yet-surfaced verbs the README deliberately keeps quiet about, and
//! whether to promote one of those to the public help screen is the product owner's call, not
//! something a test should force.
//!
//! ## What this file pins
//!
//! This test does not hand-list the commands it expects README readers to find — a hand list goes
//! stale the moment the README documents a new one and nobody remembers to update this file too.
//! Instead it parses the README's own backtick-quoted `` `cg <command>` `` mentions at test time to
//! derive the documented set, then asserts every name in that derived set appears somewhere in
//! `cg --help`. Any future command the README documents but help omits fails this test on the same
//! assertion, with no test edit required.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Everything `cg --help` printed, stdout and stderr combined.
fn cg_help() -> String {
    let home = tempfile::tempdir().expect("a throwaway home for a help-only invocation");
    let output = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_cg")))
        .arg("--help")
        .env("HOME", home.path())
        .env("CG_DATA_ROOT", home.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("run the compiled cg binary");
    assert_eq!(
        output.status.code(),
        Some(0),
        "asking for help is not a failure: cg --help exited {:?}",
        output.status.code()
    );
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// The operator commands `crates/context-graph/README.md` documents, parsed out of the README's
/// own backtick-quoted `` `cg <command>` `` mentions instead of hand-listed, so this set tracks the
/// README as it grows. A mention counts only inside backticks (`` `cg status` ``, `` `cg sync` ``)
/// — plain prose that happens to contain the letters "cg" followed by a word (`cg embedder cache`,
/// `whether cg is connected`) is not the README documenting a command, so it is deliberately not
/// picked up.
fn readme_documented_commands() -> Vec<String> {
    let readme = include_str!("../../context-graph/README.md");
    let mut commands = Vec::new();
    let mut rest = readme;
    while let Some(idx) = rest.find("`cg ") {
        rest = &rest[idx + "`cg ".len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_lowercase() || c == '-'))
            .unwrap_or(rest.len());
        let word = &rest[..end];
        if !word.is_empty() {
            commands.push(word.to_string());
        }
    }
    commands.sort();
    commands.dedup();
    commands
}

#[test]
fn every_readme_documented_command_is_named_in_cg_help() {
    let commands = readme_documented_commands();
    assert!(
        !commands.is_empty(),
        "the README parse found no backtick-quoted `cg <command>` mentions at all — the parse \
         likely drifted from the README's shape, since it documents `cg status` at minimum"
    );

    let help = cg_help();
    let mut missing: Vec<&str> = commands
        .iter()
        .map(String::as_str)
        .filter(|command| {
            !help
                .lines()
                .any(|line| line.trim_start().starts_with(command))
        })
        .collect();
    missing.sort_unstable();
    missing.dedup();

    assert!(
        missing.is_empty(),
        "crates/context-graph/README.md documents these commands but `cg --help` never lists \
         them, so a reader who found the command in the README learns nothing about it from help: \
         {missing:?}\n\n{help}"
    );
}
