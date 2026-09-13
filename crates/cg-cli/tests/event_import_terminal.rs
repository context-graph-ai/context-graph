//! The one-time Context question is asked on the controlling terminal, never on stdout, and a
//! basename match never links a project without confirmation.

mod event_support;

use event_support::{Machine, str_at, u64_at, write_session};
use serde_json::Value;

#[test]
fn first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes()
 {
    let machine = Machine::without_model();
    let project = machine.project("orchard");
    write_session(
        &machine.history_root(),
        &project,
        "sess-orchard",
        &[("orchard question", "orchard answer")],
    );

    let unattended = machine.run(&project, &["import", "."]);
    assert_eq!(unattended.code, 2, "{}", unattended.describe());
    assert_eq!(
        str_at(&unattended.json(), "/state"),
        "context_binding_required"
    );
    assert!(!machine.data_root.exists(), "no terminal, no writes");

    let (imported, transcript) = machine.run_interactive(
        &project,
        &["import", ".", "--search", "defer"],
        &[("Context name [orchard]: ", "\r")],
    );
    assert_eq!(
        imported.code,
        0,
        "{} transcript={transcript:?}",
        imported.describe()
    );
    assert!(
        transcript.contains("No context is linked to this project."),
        "the question is written to the terminal: {transcript:?}"
    );
    assert!(
        !imported.stdout.contains("Context name") && !imported.stdout.contains("No context"),
        "the prompt never enters JSON stdout: {}",
        imported.stdout
    );
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/context/name"), "orchard");
    assert_eq!(str_at(&receipt, "/selection/source"), "project_binding");
    assert_eq!(
        receipt.pointer("/selection/binding_created"),
        Some(&Value::Bool(true))
    );
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);

    let deep = project.join("nested").join("folder");
    std::fs::create_dir_all(&deep).expect("descendant directory");
    for args in [vec!["status"], vec!["import", "../..", "--search", "defer"]] {
        let output = machine.run(&deep, &args);
        assert_eq!(output.code, 0, "{args:?}: {}", output.describe());
        let value = output.json();
        assert_eq!(str_at(&value, "/selection/source"), "project_binding");
        assert_eq!(str_at(&value, "/context/name"), "orchard");
    }
}

#[test]
fn an_existing_same_name_context_requires_confirmation_and_enter_means_no() {
    let machine = Machine::without_model();
    let elsewhere = machine.project("elsewhere");
    let file = write_session(
        &machine.history_root(),
        &elsewhere,
        "sess-elsewhere",
        &[("elsewhere question", "elsewhere answer")],
    );
    let seeded = machine.run(
        &machine.root,
        &[
            "import",
            &file.display().to_string(),
            "--context",
            "shared",
            "--search",
            "defer",
        ],
    );
    assert_eq!(seeded.code, 0, "{}", seeded.describe());

    let project = machine.project("shared");
    write_session(
        &machine.history_root(),
        &project,
        "sess-shared",
        &[("shared question", "shared answer")],
    );
    let (declined, transcript) = machine.run_interactive(
        &project,
        &["import", ".", "--search", "defer"],
        &[
            ("Context name [shared]: ", "\r"),
            (
                "Context \"shared\" already exists. Link this project to it? [y/N] ",
                "\r",
            ),
        ],
    );
    assert_eq!(
        declined.code,
        2,
        "{} transcript={transcript:?}",
        declined.describe()
    );
    assert_eq!(
        str_at(&declined.json(), "/state"),
        "context_binding_required"
    );
    let status = machine.run(&project, &["status"]);
    assert_ne!(
        status
            .json()
            .pointer("/selection/source")
            .and_then(Value::as_str),
        Some("project_binding"),
        "a basename match alone never links: {}",
        status.describe()
    );

    let (linked, transcript) = machine.run_interactive(
        &project,
        &["import", ".", "--search", "defer"],
        &[
            ("Context name [shared]: ", "\r"),
            (
                "Context \"shared\" already exists. Link this project to it? [y/N] ",
                "y\r",
            ),
        ],
    );
    assert_eq!(
        linked.code,
        0,
        "{} transcript={transcript:?}",
        linked.describe()
    );
    let linked = linked.json();
    assert_eq!(str_at(&linked, "/context/name"), "shared");
    assert_eq!(str_at(&linked, "/selection/source"), "project_binding");

    let named = machine.project("plum");
    write_session(
        &machine.history_root(),
        &named,
        "sess-plum",
        &[("plum question", "plum answer")],
    );
    let (typed, transcript) = machine.run_interactive(
        &named,
        &["import", ".", "--search", "defer"],
        &[("Context name [plum]: ", "fruit\r")],
    );
    assert_eq!(
        typed.code,
        0,
        "{} transcript={transcript:?}",
        typed.describe()
    );
    assert_eq!(str_at(&typed.json(), "/context/name"), "fruit");
}
