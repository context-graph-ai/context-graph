//! Top-level help lists exactly the public commands while hidden commands remain callable.

use std::collections::BTreeSet;
use std::process::{Command, Output, Stdio};

fn help(args: &[&str], home: &std::path::Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cg"))
        .args(args)
        .env("HOME", home)
        .env("CG_DATA_ROOT", home)
        .stdin(Stdio::null())
        .output()
        .expect("run cg help")
}

/// Public help contains only the declared commands, and a hidden command still answers its help.
#[test]
fn help_presents_exactly_the_public_command_list() {
    let home = tempfile::tempdir().unwrap();
    let hidden = help(&["sync", "--help"], home.path());
    let hidden_said = format!(
        "{}{}",
        String::from_utf8_lossy(&hidden.stdout),
        String::from_utf8_lossy(&hidden.stderr)
    );
    assert!(
        hidden.status.success() && hidden_said.contains("Usage: cg sync"),
        "the hidden sync command must remain callable: {hidden_said}"
    );

    let output = help(&["--help"], home.path());
    assert!(output.status.success(), "cg --help failed: {output:?}");
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let listed: BTreeSet<String> = said
        .lines()
        .take_while(|line| *line != "Examples:")
        .filter_map(|line| {
            let entry = line.strip_prefix("  ")?;
            if entry.starts_with(char::is_whitespace) {
                return None;
            }
            let mut words = entry.split_whitespace();
            let command = words.next()?;
            if !command.chars().all(|c| c.is_ascii_lowercase() || c == '-') {
                return None;
            }
            let mut name = command.to_string();
            if command == "model" {
                let subcommand = words.next()?;
                name.push(' ');
                name.push_str(subcommand);
                if subcommand == "auth" {
                    name.push(' ');
                    name.push_str(words.next()?);
                }
            }
            Some(name)
        })
        .collect();
    let expected: BTreeSet<String> = [
        "import",
        "find",
        "show",
        "status",
        "forget",
        "model connect",
        "model fetch",
        "model list",
        "model show",
        "model status",
        "model auth login",
        "doctor",
        "selftest",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    assert_eq!(
        listed, expected,
        "cg --help must list exactly the public command set:\n{said}"
    );
}
