//! What one write is allowed to hold on this machine, who set that number, and what happens when a
//! write does not fit it.
//!
//! The user sets nothing to get a working answer: cg reads the machine. Anyone who wants a
//! different number says so with a switch for one command or with `[write] memory_limit_bytes` in
//! `cg/fabric.toml` for every command — there is no environment variable, and `cg status` prints
//! both the number in force and which of the three set it, so an operator whose import was refused
//! reads the number they have to raise.

mod event_support;

use std::fs;
use std::path::Path;

use serde_json::json;

use event_support::{Machine, str_at, u64_at, write_session};

fn write_budget_config(machine: &Machine, contents: &str) {
    let directory = machine.data_root.join("cg");
    fs::create_dir_all(&directory).expect("create isolated cg configuration directory");
    fs::write(directory.join("fabric.toml"), contents).expect("write isolated write policy");
}

fn imported_project(machine: &Machine, name: &str) -> std::path::PathBuf {
    let project = machine.project(name);
    write_session(
        &machine.history_root(),
        &project,
        &format!("session-{name}"),
        &[("why bound the writer", "so a refusal replaces a kill")],
    );
    let imported = machine.run(
        &project,
        &["import", ".", "--context", name, "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    project
}

fn write_many_turns(path: &Path, project: &Path) {
    let mut lines = String::new();
    for index in 0..350 {
        let hour = 9 + index / 60;
        let minute = index % 60;
        let timestamp = format!("2026-09-01T{hour:02}:{minute:02}:00Z");
        let user = json!({
            "type": "user", "uuid": uuid::Uuid::new_v4(),
            "sessionId": "mid-budget-session", "cwd": project, "timestamp": timestamp,
            "message": {"role": "user", "content": format!("question {index} {}", "reasoning ".repeat(40))},
        });
        let assistant = json!({
            "type": "assistant", "uuid": uuid::Uuid::new_v4(),
            "sessionId": "mid-budget-session", "cwd": project, "timestamp": timestamp,
            "message": {"id": format!("mid-budget-m{index}"), "role": "assistant",
                "content": [{"type": "text", "text": format!("answer {index} {}", "detail ".repeat(40))}]},
        });
        lines.push_str(&user.to_string());
        lines.push('\n');
        lines.push_str(&assistant.to_string());
        lines.push('\n');
    }
    fs::write(path, lines).expect("write multi-batch session");
}

#[test]
fn status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it() {
    let machine = Machine::without_model();
    let project = imported_project(&machine, "write-budget");

    let derived = machine.run(&project, &["status", "--context", "write-budget"]);
    assert_eq!(derived.code, 0, "{}", derived.describe());
    let budget = u64_at(&derived.json(), "/store/write_memory_limit_bytes");
    assert!(budget > 0, "{}", derived.describe());
    assert!(
        str_at(&derived.json(), "/store/write_memory_limit_source")
            .starts_with("derived from this machine"),
        "with nothing declared the user set nothing and cg read the machine: {}",
        derived.describe()
    );

    write_budget_config(&machine, "[write]\nmemory_limit_bytes = 268435456\n");
    let configured = machine.run(&project, &["status", "--context", "write-budget"]);
    assert_eq!(configured.code, 0, "{}", configured.describe());
    assert_eq!(
        268_435_456,
        u64_at(&configured.json(), "/store/write_memory_limit_bytes")
    );
    assert!(
        str_at(&configured.json(), "/store/write_memory_limit_source")
            .contains("[write] memory_limit_bytes"),
        "{}",
        configured.describe()
    );

    let switched = machine.run(
        &project,
        &[
            "status",
            "--context",
            "write-budget",
            "--write-memory-limit-bytes",
            "536870912",
        ],
    );
    assert_eq!(switched.code, 0, "{}", switched.describe());
    assert_eq!(
        536_870_912,
        u64_at(&switched.json(), "/store/write_memory_limit_bytes"),
        "a number named on the command beats this machine's config file: {}",
        switched.describe()
    );
    assert!(
        str_at(&switched.json(), "/store/write_memory_limit_source")
            .contains("--write-memory-limit-bytes"),
        "{}",
        switched.describe()
    );
}

#[test]
fn a_small_writer_setting_does_not_block_help_status_or_show() {
    let machine = Machine::without_model();
    let project = imported_project(&machine, "small-writer-read");
    let limit = ["--write-memory-limit-bytes", "1"];

    let help = machine.run(&project, &["--help", limit[0], limit[1]]);
    assert_eq!(help.code, 0, "{}", help.describe());
    assert!(help.stdout.contains("Usage: cg <command>"));

    let status = machine.run(
        &project,
        &[
            "status",
            "--context",
            "small-writer-read",
            limit[0],
            limit[1],
        ],
    );
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(1, u64_at(&status.json(), "/store/write_memory_limit_bytes"));

    let show = machine.run(
        &project,
        &[
            "show",
            "00000000-0000-0000-0000-000000000000",
            "--context",
            "small-writer-read",
            limit[0],
            limit[1],
        ],
    );
    assert_eq!(show.code, 1, "{}", show.describe());
    assert_eq!("passage_not_found", str_at(&show.json(), "/state"));
}

#[test]
fn empty_history_can_be_forgotten_with_a_small_container_writer_share() {
    let machine = Machine::without_model();
    let project = imported_project(&machine, "empty-small-container");
    let container_share = context_graph::write_budget::WriterMemoryLimit::derived_from(
        context_graph::write_budget::MachineMemory {
            total_bytes: Some(16 * 1024 * 1024 * 1024),
            cgroup_limit_bytes: Some(512 * 1024 * 1024),
        },
    );
    assert_eq!(container_share.bytes.get(), 128 * 1024 * 1024);
    assert!(matches!(
        container_share.source,
        context_graph::write_budget::WriterMemoryLimitSource::DerivedFromMachine { .. }
    ));
    assert!(container_share.too_small_refusal().is_some());

    let refused = machine.run(
        &project,
        &[
            "forget",
            "--context",
            "empty-small-container",
            "--yes",
            "--write-memory-limit-bytes",
            "1",
        ],
    );
    assert_eq!(refused.code, 1, "{}", refused.describe());
    let refusal = refused.json();
    assert_eq!(str_at(&refusal, "/state"), "purge_refused");
    assert_eq!(
        str_at(&refusal, "/condition"),
        "WriteDidNotFitTheMemoryLimit"
    );
    assert!(str_at(&refusal, "/message").contains("no history was erased"));
    let unchanged = machine.run(&project, &["status", "--context", "empty-small-container"]);
    assert_eq!(unchanged.code, 0, "{}", unchanged.describe());
    assert_eq!(u64_at(&unchanged.json(), "/history/captured_events"), 2);

    let first = machine.run(
        &project,
        &[
            "forget",
            "--context",
            "empty-small-container",
            "--yes",
            "--write-memory-limit-bytes",
            "134217728",
        ],
    );
    assert_eq!(first.code, 0, "{}", first.describe());
    assert_eq!(str_at(&first.json(), "/state"), "forgotten");
    assert_eq!(u64_at(&first.json(), "/counts/events"), 2);

    // The test cannot set its cgroup ceiling, so the switch supplies the same effective limit
    // for a nonempty erasure. Empty selections run without a switch and never seek writable
    // admission, whatever machine-derived writer limit is in force on the test host.
    for _ in 0..3 {
        let empty = machine.run(&project, &["forget", "--context", "empty-small-container"]);
        assert_eq!(empty.code, 0, "{}", empty.describe());
        assert_eq!(str_at(&empty.json(), "/state"), "nothing_to_forget");
        assert_eq!(u64_at(&empty.json(), "/counts/events"), 0);
    }
}

#[test]
fn the_text_rendering_prints_the_same_budget_and_origin_the_json_carries() {
    let machine = Machine::without_model();
    let project = imported_project(&machine, "write-budget-text");
    write_budget_config(&machine, "[write]\nmemory_limit_bytes = 268435456\n");

    let rendered = machine.run(
        &project,
        &[
            "status",
            "--context",
            "write-budget-text",
            "--format",
            "text",
        ],
    );
    assert_eq!(rendered.code, 0, "{}", rendered.describe());
    assert!(
        rendered.stdout.contains("write memory: 256.0 MiB")
            && rendered.stdout.contains("[write] memory_limit_bytes"),
        "the human rendering owes the operator the same two facts the JSON carries: {}",
        rendered.describe()
    );
}

#[test]
fn a_write_that_does_not_fit_the_declared_budget_is_refused_in_words_that_name_both_knobs() {
    let machine = Machine::without_model();
    let project = imported_project(&machine, "write-budget-refused");
    write_session(
        &machine.history_root(),
        &project,
        "session-write-budget-next",
        &[("why bound the writer", "so a refusal replaces a kill")],
    );

    let refused = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "write-budget-refused",
            "--search",
            "defer",
            "--write-memory-limit-bytes",
            "1",
        ],
    );
    assert_eq!(refused.code, 1, "{}", refused.describe());
    let refusal = refused.json();
    let message = str_at(&refusal, "/message");
    assert!(
        message.contains("nothing was written")
            && message.contains("--write-memory-limit-bytes")
            && message.contains("[write] memory_limit_bytes"),
        "a write that does not fit is told what it asked for, what it is allowed and the two \
         knobs that change the second: {}",
        refused.describe()
    );

    let status = machine.run(&project, &["status", "--context", "write-budget-refused"]);
    assert_eq!(
        status.code,
        0,
        "the existing store remains readable: {}",
        status.describe()
    );
    assert_eq!(
        u64_at(&status.json(), "/history/captured_events"),
        2,
        "the refused import left no event from the new session: {}",
        status.describe()
    );

    let retry = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "write-budget-refused",
            "--search",
            "defer",
        ],
    );
    assert_eq!(
        retry.code,
        0,
        "the store accepts a later import: {}",
        retry.describe()
    );
    assert_eq!(u64_at(&retry.json(), "/capture/events/inserted"), 2);
    let status = machine.run(&project, &["status", "--context", "write-budget-refused"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(u64_at(&status.json(), "/history/captured_events"), 4);

    let repeated = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "write-budget-refused",
            "--search",
            "defer",
        ],
    );
    assert_eq!(repeated.code, 0, "{}", repeated.describe());
    assert_eq!(u64_at(&repeated.json(), "/capture/events/inserted"), 0);
}

/// A machine whose store already holds the first 512 events of `mid-budget.jsonl`, committed under
/// a limit the engine accepts, with the whole 700-event file on disk ready to import.
fn store_holding_the_first_512_events() -> (Machine, std::path::PathBuf, String) {
    const GENEROUS_LIMIT: &str = "268435456";
    let machine = Machine::without_model();
    let project = machine.project("mid-budget");
    let source = project.join("mid-budget.jsonl");
    write_many_turns(&source, &project);
    let whole = fs::read_to_string(&source).expect("read the whole session");
    let first_512 = whole.lines().take(512).collect::<Vec<_>>().join("\n") + "\n";
    fs::write(&source, first_512).expect("write the first 512 events");
    let path = source.to_str().expect("UTF-8 source path").to_owned();
    let committed = machine.run(
        &project,
        &[
            "import",
            &path,
            "--context",
            "mid-budget",
            "--search",
            "defer",
            "--write-memory-limit-bytes",
            GENEROUS_LIMIT,
        ],
    );
    assert_eq!(committed.code, 0, "{}", committed.describe());
    assert_eq!(u64_at(&committed.json(), "/capture/events/inserted"), 512);
    fs::write(&source, whole).expect("write the whole session");
    (machine, project, path)
}

fn import_whole_session(
    machine: &Machine,
    project: &std::path::Path,
    path: &str,
    limit: u64,
) -> event_support::Output {
    machine.run(
        project,
        &[
            "import",
            path,
            "--context",
            "mid-budget",
            "--search",
            "defer",
            "--write-memory-limit-bytes",
            &limit.to_string(),
        ],
    )
}

// What this proves: events committed before a writer-memory refusal survive it, the refusal says
// what limit was in force and what the write needed, and a retry imports exactly the events that
// are missing, inserting nothing twice. The limits come from the engine's own accept/refuse
// answer, not from a literal: the limit starts below what the remaining events need and doubles
// until the engine accepts, and the last refusal before that is the one examined. The engine
// currently refuses at a fixed floor rather than part-way through an import, so the refusal is
// produced by a limit under that floor.
#[test]
fn committed_events_survive_a_writer_memory_refusal_and_a_retry_imports_only_the_rest() {
    let (machine, project, path) = store_holding_the_first_512_events();

    let mut limit: u64 = 1024 * 1024;
    let mut last_refusal = None;
    let retry = loop {
        let attempt = import_whole_session(&machine, &project, &path, limit);
        if attempt.code == 0 {
            break attempt;
        }
        assert_eq!(attempt.code, 1, "{}", attempt.describe());
        // A refused write changes nothing: the committed events are all still there.
        let status = machine.run(&project, &["status", "--context", "mid-budget"]);
        assert_eq!(status.code, 0, "{}", status.describe());
        assert_eq!(u64_at(&status.json(), "/history/captured_events"), 512);
        last_refusal = Some((limit, attempt));
        limit *= 2;
        assert!(
            limit <= 4 * 1024 * 1024 * 1024,
            "the engine accepted no writer limit up to 4 GiB"
        );
    };

    let (refused_limit, limited) =
        last_refusal.expect("the engine refused a limit below the floor");
    let refusal = limited.json();
    assert_eq!(str_at(&refusal, "/state"), "write_memory_limit");
    assert_eq!(
        str_at(&refusal, "/condition"),
        "WriteDidNotFitTheMemoryLimit"
    );
    assert_eq!(u64_at(&refusal, "/limit_bytes"), refused_limit);
    assert!(u64_at(&refusal, "/requested_bytes") > 0);
    assert_eq!(u64_at(&refusal, "/capture/events/inserted"), 0);
    assert_eq!(str_at(&refusal, "/search/status"), "not_run");
    let message = str_at(&refusal, "/message");
    assert!(
        message.contains("cg import stopped")
            && message.contains("MiB")
            && message.contains("Earlier committed events remain")
            && message.contains("--write-memory-limit-bytes")
            && message.contains("[write] memory_limit_bytes")
            && !message.contains("nothing was written"),
        "{}",
        limited.describe()
    );

    // The first limit the engine accepted is the retry: it imports exactly the missing events.
    assert_eq!(u64_at(&retry.json(), "/capture/events/inserted"), 188);
    assert_eq!(u64_at(&retry.json(), "/capture/local_embedding_calls"), 0);
    let status = machine.run(&project, &["status", "--context", "mid-budget"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(u64_at(&status.json(), "/history/captured_events"), 700);

    let repeated = machine.run(
        &project,
        &[
            "import",
            &path,
            "--context",
            "mid-budget",
            "--search",
            "defer",
        ],
    );
    assert_eq!(repeated.code, 0, "{}", repeated.describe());
    assert_eq!(u64_at(&repeated.json(), "/capture/events/inserted"), 0);
    assert_eq!(
        u64_at(&repeated.json(), "/capture/local_embedding_calls"),
        0
    );
}

#[test]
fn a_write_budget_that_is_not_a_byte_count_is_a_config_error_before_any_store_is_opened() {
    let machine = Machine::without_model();
    let project = machine.project("invalid-write-budget");
    for invalid in [
        "[write]\nmemory_limit_bytes = 0\n",
        "[write]\nmemory_limit_bytes = -1\n",
        "[write]\nmemory_limit_bytes = 'many'\n",
    ] {
        write_budget_config(&machine, invalid);
        let output = machine.run(&project, &["status"]);
        assert_eq!(output.code, 1, "{}", output.describe());
        let refusal = output.json();
        assert_eq!(str_at(&refusal, "/state"), "config_error");
        assert!(
            str_at(&refusal, "/message").contains("memory_limit_bytes"),
            "{}",
            output.describe()
        );
    }
}
