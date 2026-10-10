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

/// Turns per request's session; writer memory per batch does not depend on event count, so a
/// small session measures the same ceiling as a large one.
const FIXTURE_TURNS: usize = 32;
const FIXTURE_EVENTS: u64 = 2 * FIXTURE_TURNS as u64;
const MIB: u64 = 1 << 20;

/// One session file of `FIXTURE_TURNS` user/assistant turns.
fn write_many_turns(path: &Path, project: &Path, session: &str) {
    let mut lines = String::new();
    for index in 0..FIXTURE_TURNS {
        let hour = 9 + index / 60;
        let minute = index % 60;
        let timestamp = format!("2026-09-01T{hour:02}:{minute:02}:00Z");
        let user = json!({
            "type": "user", "uuid": uuid::Uuid::new_v4(),
            "sessionId": session, "cwd": project, "timestamp": timestamp,
            "message": {"role": "user", "content": format!("question {index} {}", "reasoning ".repeat(40))},
        });
        let assistant = json!({
            "type": "assistant", "uuid": uuid::Uuid::new_v4(),
            "sessionId": session, "cwd": project, "timestamp": timestamp,
            "message": {"id": format!("{session}-m{index}"), "role": "assistant",
                "content": [{"type": "text", "text": format!("answer {index} {}", "detail ".repeat(40))}]},
        });
        lines.push_str(&user.to_string());
        lines.push('\n');
        lines.push_str(&assistant.to_string());
        lines.push('\n');
    }
    fs::write(path, lines).expect("write multi-batch session");
}

/// Scratch stores for measuring request 2's ceiling. Each holds request 1 committed under the
/// default writer budget; a refused request 2 writes nothing, so the store is reused until a
/// probe admits request 2.
#[derive(Default)]
struct RequestTwoProbe {
    scratch: Option<(Machine, std::path::PathBuf, std::path::PathBuf)>,
}

impl RequestTwoProbe {
    /// Events request 2 commits under `limit`.
    fn committed_under(&mut self, limit: u64) -> u64 {
        let (machine, project, later) = self.scratch.take().unwrap_or_else(|| {
            let machine = Machine::without_model();
            let project = machine.project("mid-budget-probe");
            let first = project.join("mid-budget-probe.jsonl");
            let later = project.join("mid-budget-probe-later.jsonl");
            write_many_turns(&first, &project, "mid-budget-probe-session");
            write_many_turns(&later, &project, "mid-budget-probe-later-session");
            let request_one = machine.run(
                &project,
                &[
                    "import",
                    first.to_str().expect("UTF-8 probe path"),
                    "--context",
                    "mid-budget-probe",
                    "--search",
                    "defer",
                ],
            );
            assert_eq!(request_one.code, 0, "{}", request_one.describe());
            (machine, project, later)
        });
        let probe = machine.run(
            &project,
            &[
                "import",
                later.to_str().expect("UTF-8 probe path"),
                "--context",
                "mid-budget-probe",
                "--search",
                "defer",
                "--write-memory-limit-bytes",
                &limit.to_string(),
            ],
        );
        let committed = match probe.code {
            0 => u64_at(&probe.json(), "/capture/events/inserted"),
            // A refusal before capture began carries no capture section: nothing was written.
            1 if str_at(&probe.json(), "/state") == "write_memory_limit" => probe
                .json()
                .pointer("/capture/events/inserted")
                .map_or(0, |_| u64_at(&probe.json(), "/capture/events/inserted")),
            _ => panic!("scratch import under {limit} bytes: {}", probe.describe()),
        };
        if committed == 0 {
            self.scratch = Some((machine, project, later));
        }
        committed
    }
}

/// The writer limits on either side of `admits`, to one MiB: the largest measured limit it
/// refuses and the smallest it admits.
fn limit_bracket(mut admits: impl FnMut(u64) -> bool) -> (u64, u64) {
    let mut admitted = MIB;
    while !admits(admitted) {
        assert!(
            admitted < 64 << 30,
            "no writer limit up to 64 GiB admits the fixture"
        );
        admitted *= 2;
    }
    let mut refused = if admitted > MIB { admitted / 2 } else { 0 };
    while admitted - refused > MIB {
        let middle = (refused + (admitted - refused) / 2) / MIB * MIB;
        if admits(middle) {
            admitted = middle;
        } else {
            refused = middle;
        }
    }
    (refused, admitted)
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

#[test]
fn an_import_that_exceeds_writer_memory_after_a_batch_reports_committed_progress_and_resumes() {
    // Writer memory per batch does not grow with payload, so work already committed comes from
    // an earlier request: each import carries its own writer ceiling. The ceiling is measured on
    // this build, never a literal: request 2 gets the largest measured limit its first batch does
    // not fit, after request 1 committed its own session under the default writer budget.
    let mut probe = RequestTwoProbe::default();
    let (refused_limit, admitted_limit) =
        limit_bracket(|limit| probe.committed_under(limit) == FIXTURE_EVENTS);
    let limit = refused_limit;
    let bracket = format!(
        "request 2 is refused at {refused_limit} and admitted from {admitted_limit}; limit {limit}"
    );
    assert!(
        limit > 0,
        "empty bracket: request 2 is admitted at every measured writer limit; {bracket}"
    );

    let machine = Machine::without_model();
    let project = machine.project("mid-budget");
    let first = project.join("mid-budget.jsonl");
    let later = project.join("mid-budget-later.jsonl");
    write_many_turns(&first, &project, "mid-budget-session");
    write_many_turns(&later, &project, "mid-budget-later-session");
    let first = first.to_str().expect("UTF-8 source path");
    let source = later.to_str().expect("UTF-8 source path");
    let committed = machine.run(
        &project,
        &[
            "import",
            first,
            "--context",
            "mid-budget",
            "--search",
            "defer",
        ],
    );
    assert_eq!(committed.code, 0, "{}", committed.describe());
    assert_eq!(
        u64_at(&committed.json(), "/capture/events/inserted"),
        FIXTURE_EVENTS
    );
    let before = machine.run(&project, &["status", "--context", "mid-budget"]);
    assert_eq!(before.code, 0, "{}", before.describe());

    let limited = machine.run(
        &project,
        &[
            "import",
            source,
            "--context",
            "mid-budget",
            "--search",
            "defer",
            "--write-memory-limit-bytes",
            &limit.to_string(),
        ],
    );
    assert_eq!(limited.code, 1, "{}; {bracket}", limited.describe());
    let refusal = limited.json();
    assert_eq!(str_at(&refusal, "/state"), "write_memory_limit");
    assert_eq!(
        str_at(&refusal, "/condition"),
        "WriteDidNotFitTheMemoryLimit"
    );
    assert_eq!(u64_at(&refusal, "/limit_bytes"), limit);
    assert!(u64_at(&refusal, "/requested_bytes") > 0);
    assert_eq!(str_at(&refusal, "/capture/status"), "partial");
    assert_eq!(
        u64_at(&refusal, "/capture/events/inserted"),
        0,
        "request 2 wrote nothing; {bracket}"
    );
    assert_eq!(
        u64_at(&refusal, "/capture/last_committed_position/committed_bytes"),
        0
    );
    assert_eq!(str_at(&refusal, "/search/status"), "not_run");
    let message = str_at(&refusal, "/message");
    assert!(
        message.contains("cg import stopped")
            && message.contains(&context_graph::write_budget::human_bytes(limit))
            && message.contains("Earlier committed events remain")
            && message.contains("--write-memory-limit-bytes")
            && message.contains("[write] memory_limit_bytes")
            && !message.contains("nothing was written"),
        "{}; {bracket}",
        limited.describe()
    );

    let status = machine.run(&project, &["status", "--context", "mid-budget"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(
        status.json()["history"],
        before.json()["history"],
        "request 1's committed work is unchanged by the refused request"
    );
    assert_eq!(
        u64_at(&status.json(), "/history/captured_events"),
        FIXTURE_EVENTS
    );

    let retry = machine.run(
        &project,
        &[
            "import",
            source,
            "--context",
            "mid-budget",
            "--search",
            "defer",
            "--write-memory-limit-bytes",
            &admitted_limit.max(268_435_456).to_string(),
        ],
    );
    assert_eq!(retry.code, 0, "{}; {bracket}", retry.describe());
    assert_eq!(
        u64_at(&retry.json(), "/capture/events/inserted"),
        FIXTURE_EVENTS
    );
    assert_eq!(u64_at(&retry.json(), "/capture/local_embedding_calls"), 0);
    let status = machine.run(&project, &["status", "--context", "mid-budget"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(
        u64_at(&status.json(), "/history/captured_events"),
        2 * FIXTURE_EVENTS
    );

    let repeated = machine.run(
        &project,
        &[
            "import",
            first,
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
