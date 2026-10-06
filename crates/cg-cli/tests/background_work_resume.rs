//! A killed background search worker is resumed by the user's next `cg` command.
//!
//! The user-facing contract: after the detached worker is killed mid-embedding, the next `cg`
//! command that opens the data root — an import, or a read such as `find` or `status` — starts the
//! worker again when durable pending work exists and no live worker holds it, with no lease to
//! outwait and no manual cleanup. The resumed worker keeps every committed passage and embeds only
//! what never committed. `cg status` reports the killed work as interrupted, never as running. A
//! caught-up store starts nothing.
//!
//! The worker is killed with SIGKILL on its recorded process group while it is inside an
//! embedding call, after at least one batch has committed. Every rendezvous is a pipe: the worker's
//! mid-embedding marker is a FIFO the test reads, and the end of a worker is observed through a
//! process descriptor. Nothing sleeps and no clock is advanced.

#![cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]

mod event_support;

use std::collections::{BTreeMap, HashMap};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use contextdb_core::Value as DbValue;
use event_support::{Machine, str_at, write_session};

const OBSERVER_ENV: &str = "CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER";
const CONTEXT: &str = "resume-after-kill";
/// A hang guard for a rendezvous that must happen; the test fails rather than waits forever.
const RENDEZVOUS: Duration = Duration::from_secs(600);

/// A barrier directory that records the spawned worker's process group and never pauses it after
/// a completed pass.
fn observer(machine: &Machine, name: &str) -> PathBuf {
    let path = machine.root.join(name);
    std::fs::create_dir_all(&path).expect("create worker observer directory");
    std::fs::write(
        path.join("observe-embedding-worker-batches"),
        b"record the worker process group\n",
    )
    .expect("enable the worker process-group record");
    std::fs::write(path.join("release"), b"do not pause after a pass\n")
        .expect("pre-release the completed-pass barrier");
    path
}

/// Make the worker under this observer stop inside an embedding call once a batch has committed.
/// Its marker is a FIFO, so the test learns of the pause the moment the worker writes it.
fn pause_mid_embed(observer: &Path) -> std::sync::mpsc::Receiver<()> {
    std::fs::write(
        observer.join("pause-mid-embed-after-committed-batch"),
        b"pause inside the next embedding call after a committed batch\n",
    )
    .expect("enable the mid-embedding pause");
    let fifo = observer.join("mid-embed-entered");
    let name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).expect("fifo path");
    // SAFETY: name is a valid NUL-terminated path owned for the duration of the call.
    assert_eq!(
        unsafe { libc::mkfifo(name.as_ptr(), 0o600) },
        0,
        "create the marker FIFO"
    );
    let (entered, entry) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        // Opening a FIFO for reading blocks until the worker opens it to write its marker.
        if std::fs::read(&fifo).is_ok() {
            let _ = entered.send(());
        }
    });
    entry
}

fn worker_group(observer: &Path) -> i32 {
    std::fs::read_to_string(observer.join("embedding-worker-process-group"))
        .expect("the command that started the worker recorded its process group")
        .trim()
        .parse()
        .expect("process group is a pid")
}

/// Block until the process `pid` has exited, through a process descriptor rather than polling.
fn wait_for_exit(pid: i32) {
    // SAFETY: pidfd_open takes a pid and flags and returns a new descriptor or -1.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    if fd < 0 {
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH),
            "open a process descriptor for the worker"
        );
        return; // Already gone.
    }
    let fd = i32::try_from(fd).expect("descriptor fits");
    let mut poll = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let guard_ms = i32::try_from(RENDEZVOUS.as_millis()).expect("guard fits");
    // SAFETY: poll reads one initialized pollfd; the descriptor is closed right after.
    let ready = unsafe { libc::poll(&mut poll, 1, guard_ms) };
    // SAFETY: fd was returned by pidfd_open above and is closed exactly once.
    unsafe { libc::close(fd) };
    assert_eq!(ready, 1, "the worker {pid} exited within the hang guard");
}

/// Every passage row as id -> (rendered text, current).
fn passages(machine: &Machine) -> BTreeMap<uuid::Uuid, (String, bool)> {
    let db = contextdb_engine::Database::open(machine.db_path()).expect("open the store");
    let rows = db
        .execute(
            "SELECT id, rendered_text, is_current FROM event_search_passages",
            &HashMap::new(),
        )
        .expect("read passages")
        .rows;
    db.close().expect("close the store");
    rows.into_iter()
        .map(|row| match row.as_slice() {
            [
                DbValue::Uuid(id),
                DbValue::Text(text),
                DbValue::Bool(current),
            ] => (*id, (text.clone(), *current)),
            other => panic!("unexpected passage row {other:?}"),
        })
        .collect()
}

/// Three sessions, so the worker commits at least one batch before the one it is killed in.
fn three_sessions(machine: &Machine) -> PathBuf {
    let project = machine.project("resume-after-kill");
    for (session, question, answer) in [
        (
            "sess-transport",
            "Why did we pick Iroh instead of NATS for the transport?",
            "Iroh keeps a direct peer-to-peer path; NATS would add a broker we must operate.",
        ),
        (
            "sess-ledger",
            "How should the ledger reconcile foreign currency entries?",
            "Convert at the booking-date rate and keep the original amount beside it.",
        ),
        (
            "sess-garden",
            "When should the tomato seedlings go outside?",
            "After the last frost, once nights stay above ten degrees.",
        ),
    ] {
        write_session(
            &machine.history_root(),
            &project,
            session,
            &[(question, answer)],
        );
    }
    project
}

/// Import in the background, kill the worker inside an embedding call after a committed batch,
/// and return the passages that had committed when it died.
fn kill_worker_mid_embed(
    machine: &Machine,
    project: &Path,
) -> BTreeMap<uuid::Uuid, (String, bool)> {
    let killed = observer(machine, "killed-worker");
    let entered = pause_mid_embed(&killed);
    let imported = machine.run_with_env(
        project,
        &["import", ".", "--context", CONTEXT],
        OBSERVER_ENV,
        &killed,
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    entered
        .recv_timeout(RENDEZVOUS)
        .expect("the background worker paused inside an embedding call");
    let group = worker_group(&killed);
    // SAFETY: the group was recorded by the command this test ran, for the worker it started.
    assert_eq!(
        unsafe { libc::kill(-group, libc::SIGKILL) },
        0,
        "kill the worker group"
    );
    wait_for_exit(group);
    let committed = passages(machine);
    assert!(
        !committed.is_empty(),
        "at least one batch committed before the kill"
    );
    committed
}

/// The resumed store keeps every passage that committed before the kill, unchanged, and covers
/// the rest exactly once.
fn assert_resumed_without_redoing_committed_work(
    machine: &Machine,
    project: &Path,
    committed: &BTreeMap<uuid::Uuid, (String, bool)>,
) {
    let status = machine.run(project, &["status", "--context", CONTEXT]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(
        str_at(&status, "/history/search/status"),
        "ready",
        "{status}"
    );
    assert_eq!(str_at(&status, "/history/search/work"), "idle", "{status}");
    let resumed = passages(machine);
    for (id, passage) in committed {
        assert_eq!(
            resumed.get(id),
            Some(passage),
            "a passage committed before the kill keeps its identity and text"
        );
    }
    assert!(
        resumed.values().all(|(_, current)| *current),
        "nothing was rebuilt: no passage was retired by the resume"
    );
    let mut texts: Vec<&String> = resumed.values().map(|(text, _)| text).collect();
    texts.sort();
    texts.dedup();
    assert_eq!(texts.len(), resumed.len(), "no passage was embedded twice");
}

#[test]
fn a_killed_background_worker_resumes_on_the_next_import() {
    // The next import restarts a killed worker at once and redoes only what never committed.
    let machine = Machine::with_nomic();
    let project = three_sessions(&machine);
    let committed = kill_worker_mid_embed(&machine, &project);

    let resumed = observer(&machine, "resumed-by-import");
    let again = machine.run_with_env(
        &project,
        &["import", ".", "--context", CONTEXT],
        OBSERVER_ENV,
        &resumed,
    );
    assert_eq!(again.code, 0, "{}", again.describe());
    wait_for_exit(worker_group(&resumed));
    assert_resumed_without_redoing_committed_work(&machine, &project, &committed);
}

#[test]
fn a_killed_background_worker_resumes_on_the_next_read_command() {
    // A read command alone restarts the killed worker; its own output and exit code are unchanged.
    for read in [
        vec!["find", "foreign currency"],
        vec!["status", "--context", CONTEXT],
    ] {
        let machine = Machine::with_nomic();
        let project = three_sessions(&machine);
        let committed = kill_worker_mid_embed(&machine, &project);

        let resumed = observer(&machine, "resumed-by-read");
        let answered = machine.run_with_env(&project, &read, OBSERVER_ENV, &resumed);
        assert_eq!(answered.code, 0, "{read:?}: {}", answered.describe());
        let _ = answered.json();
        wait_for_exit(worker_group(&resumed));
        assert_resumed_without_redoing_committed_work(&machine, &project, &committed);
    }
}

#[test]
fn cg_status_reports_interrupted_not_running_after_a_crash() {
    // After the worker is killed, status reports the pending search as interrupted, never running.
    let machine = Machine::with_nomic();
    let project = three_sessions(&machine);
    kill_worker_mid_embed(&machine, &project);

    // Status answers from the store as it finds it, before its resume hook restarts the worker.
    let resumed = observer(&machine, "resumed-by-status");
    let status = machine.run_with_env(
        &project,
        &["status", "--context", CONTEXT],
        OBSERVER_ENV,
        &resumed,
    );
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(
        str_at(&status, "/history/search/status"),
        "interrupted",
        "killed work is reported as interrupted: {status}"
    );
    assert_eq!(
        str_at(&status, "/history/search/work"),
        "interrupted",
        "{status}"
    );
    wait_for_exit(worker_group(&resumed));
}

#[test]
fn a_read_command_on_a_caught_up_store_starts_no_worker() {
    // A store with nothing pending starts no background worker when it is read.
    let machine = Machine::with_nomic();
    let project = three_sessions(&machine);
    let imported = machine.run(
        &project,
        &["import", ".", "--context", CONTEXT, "--search", "wait"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    let watched = observer(&machine, "caught-up-read");
    let status = machine.run_with_env(
        &project,
        &["status", "--context", CONTEXT],
        OBSERVER_ENV,
        &watched,
    );
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(str_at(&status.json(), "/history/search/status"), "ready");
    assert!(
        !watched.join("embedding-worker-process-group").exists(),
        "reading a caught-up store started a background worker"
    );
}
