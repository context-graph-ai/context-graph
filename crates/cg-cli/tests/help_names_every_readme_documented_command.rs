//! Every operator-reference command is discoverable in root help or its retained command help.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Everything `cg --help` printed, stdout and stderr combined.
fn cg_help(command: Option<&str>) -> String {
    let home = tempfile::tempdir().expect("a throwaway home for a help-only invocation");
    let output = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_cg")))
        .args(command)
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

/// The operator commands `docs/cli.md` documents, parsed out of the reference's
/// own backtick-quoted `` `cg <command>` `` mentions instead of hand-listed, so this set tracks the
/// reference as it grows. A mention counts only inside backticks (`` `cg status` ``, `` `cg sync` ``)
/// — plain prose that happens to contain the letters "cg" followed by a word (`cg embedder cache`,
/// `whether cg is connected`) is not the reference documenting a command, so it is deliberately not
/// picked up.
fn readme_documented_commands() -> Vec<String> {
    let readme = include_str!("../../../docs/cli.md");
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

/// Every documented command is named by root help or its command-specific help.
#[test]
fn every_readme_documented_command_is_named_in_cg_help() {
    let commands = readme_documented_commands();
    assert!(
        !commands.is_empty(),
        "the operator-reference parse found no backtick-quoted `cg <command>` mentions at all — the parse \
         likely drifted from the reference's shape, since it documents `cg status` at minimum"
    );

    let help = cg_help(None);
    let mut missing: Vec<&str> = commands
        .iter()
        .map(String::as_str)
        .filter(|command| {
            !help
                .lines()
                .any(|line| line.trim_start().starts_with(command))
                && !cg_help(Some(command)).contains(&format!("cg {command}"))
        })
        .collect();
    missing.sort_unstable();
    missing.dedup();

    assert!(
        missing.is_empty(),
        "docs/cli.md documents these commands but neither root nor command help names \
         them, so a reader who found the command in the reference learns nothing about it from help: \
         {missing:?}\n\n{help}"
    );
}
