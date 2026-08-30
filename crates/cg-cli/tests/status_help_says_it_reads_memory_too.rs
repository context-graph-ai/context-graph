//! `cg status` answers about a model AND about the memory store, so its help must say both.
//!
//! ## The promise
//!
//! Help is what a developer and an agent read to decide what a command is for. `cg status` opens
//! this machine's Context Graph memory store on every run and reports what is in it beside the
//! model connection. Help that describes it as a model-connection command alone tells someone who
//! wants to know whether their memory is there to go and look for a command that does not exist,
//! and leaves whoever wires `cg status` into a supervisor unaware that it touches the store at all
//! — so the day another process is holding that store, the non-zero exit arrives as a surprise
//! they never planned a branch for.
//!
//! ## What this file pins
//!
//! Only the DISCLOSURE, on every help surface a caller reaches it from — `cg --help`,
//! `cg model --help`, and `cg status --help`, the last of which `cg model status` shares because
//! both verbs dispatch into the same handler: that `status` names the model connection, names the
//! memory store, and says what a store it could not read is rendered as.
//! The behavior itself is proved by the store-contention tests beside this one
//! (`a_busy_store_is_told_to_the_operator.rs`,
//! `cg_status_never_reports_a_refused_store_as_empty.rs`); nothing here re-proves them, and
//! nothing here asserts an exit code for a reader-held store, which is not a settled contract.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Everything a `cg` invocation put in front of the caller.
fn everything_said(args: &[&str]) -> String {
    let home = tempfile::tempdir().expect("a throwaway home for a help-only invocation");
    let output = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_cg")))
        .args(args)
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
        "asking for help is not a failure: cg {} exited {:?}",
        args.join(" "),
        output.status.code()
    );
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// The `cg --help` entry for one command: its own line plus every continuation line indented under
/// it. A command entry opens at exactly two spaces; a continuation is indented further.
fn help_entry_for(help: &str, command: &str) -> String {
    let mut lines = help.lines();
    let head = lines
        .find(|line| {
            line.strip_prefix("  ")
                .is_some_and(|rest| rest.starts_with(command))
        })
        .unwrap_or_else(|| panic!("`cg --help` lists no `{command}` command:\n{help}"));
    let mut entry = head.to_string();
    for line in lines {
        if !line.starts_with("    ") {
            break;
        }
        entry.push('\n');
        entry.push_str(line);
    }
    entry
}

fn assert_names_both_subjects(surface: &str, said: &str) {
    let lowered = said.to_lowercase();
    assert!(
        lowered.contains("model"),
        "{surface} must still name the model connection it reports on: {said}"
    );
    assert!(
        lowered.contains("memory"),
        "{surface} must say that status also reads this machine's memory store — a caller who is \
         not told goes looking for a command that does not exist: {said}"
    );
    assert!(
        lowered.contains("store"),
        "{surface} must name the STORE, which is what an operator acts on when status refuses: \
         {said}"
    );
}

#[test]
fn the_top_level_help_entry_for_status_names_the_memory_store_beside_the_model() {
    let help = everything_said(&["--help"]);
    let entry = help_entry_for(&help, "status");
    assert_names_both_subjects("the `cg --help` entry for `status`", &entry);
}

#[test]
fn the_model_help_entry_for_status_names_the_memory_store_beside_the_model() {
    let help = everything_said(&["model", "--help"]);
    let entry = help_entry_for(&help, "status");
    assert_names_both_subjects("the `cg model --help` entry for `status`", &entry);
}

#[test]
fn cg_status_help_says_it_reads_memory_and_what_a_refused_read_looks_like() {
    let said = everything_said(&["status", "--help"]);
    assert_names_both_subjects("`cg status --help`", &said);

    let lowered = said.to_lowercase();
    assert!(
        lowered.contains("busy"),
        "`cg status --help` must disclose the settled answer for a store another process is \
         writing, or a caller meets it first in production: {said}"
    );
    assert!(
        lowered.contains("empty"),
        "`cg status --help` must say that a store it could not read is never rendered as an empty \
         store — that is the whole difference between `your memory is gone` and `your memory could \
         not be reached`: {said}"
    );
}
