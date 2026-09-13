//! `cg forget` confirms on the controlling terminal, where Enter means no, and without a terminal
//! it requires `--yes`.

mod event_support;

use event_support::{Machine, str_at, u64_at, write_session};

#[test]
fn cg_forget_confirms_on_the_terminal_and_enter_changes_nothing() {
    let machine = Machine::without_model();
    let project = machine.project("pantry");
    write_session(
        &machine.history_root(),
        &project,
        "sess-pantry-one",
        &[("pantry question", "pantry answer")],
    );
    write_session(
        &machine.history_root(),
        &project,
        "sess-pantry-two",
        &[("second question", "second answer")],
    );
    let imported = machine.run(
        &project,
        &["import", ".", "--context", "pantry", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    let unattended = machine.run(&project, &["forget", "--context", "pantry"]);
    assert_eq!(unattended.code, 2, "{}", unattended.describe());
    let unattended = unattended.json();
    assert_eq!(str_at(&unattended, "/state"), "confirmation_required");
    assert_eq!(
        str_at(&unattended, "/retry"),
        "cg forget --context pantry --yes"
    );

    let (declined, transcript) = machine.run_interactive(
        &project,
        &["forget", "--session", "sess-pantry-one"],
        &[("Forget 1 sessions (2 events) from \"pantry\"? [y/N] ", "\r")],
    );
    assert_eq!(
        declined.code,
        2,
        "{} transcript={transcript:?}",
        declined.describe()
    );
    assert!(!declined.stdout.contains("Forget 1 sessions"));
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        4,
        "Enter means no"
    );

    let (session, transcript) = machine.run_interactive(
        &project,
        &["forget", "--session", "sess-pantry-one"],
        &[(
            "Forget 1 sessions (2 events) from \"pantry\"? [y/N] ",
            "y\r",
        )],
    );
    assert_eq!(
        session.code,
        0,
        "{} transcript={transcript:?}",
        session.describe()
    );
    let session = session.json();
    assert_eq!(str_at(&session, "/state"), "forgotten");
    assert_eq!(u64_at(&session, "/counts/events"), 2);
    assert_eq!(
        str_at(&session, "/search/status"),
        "blocked",
        "without the local model the rebuild of the remaining history is blocked: {session}"
    );
    assert!(str_at(&session, "/search/next").contains("cg model fetch --defaults"));
    assert!(
        !session.to_string().contains("cg import"),
        "the user is never told to re-import: {session}"
    );
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 2);

    let (whole, transcript) = machine.run_interactive(
        &project,
        &["forget", "--context", "pantry", "--format", "text"],
        &[(
            "Forget 1 sessions (2 events) from \"pantry\"? [y/N] ",
            "y\r",
        )],
    );
    assert_eq!(
        whole.code,
        0,
        "{} transcript={transcript:?}",
        whole.describe()
    );
    assert!(whole.stdout.contains("pantry"), "{}", whole.describe());
    assert!(!whole.stdout.contains("cg import"), "{}", whole.describe());
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 0);
    assert_eq!(
        str_at(&status, "/context/name"),
        "pantry",
        "the Context remains"
    );
}
