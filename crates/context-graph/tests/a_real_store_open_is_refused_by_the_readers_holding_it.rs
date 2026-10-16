//! A REAL `Store::open` refused by live direct readers must come back as the busy condition,
//! carrying the readers it was actually refused by.
//!
//! Companion tests cover the typed condition and its operator rendering. This file exercises
//! the conversion through a real `Store::open` refused by live readers.
//!
//! ## The contract this file proves
//!
//! - A context-graph writer that meets live direct readers is refused with
//!   `CgError::StoreHeldByReaders`, produced by the shipped open path — not by a test constructing
//!   the value.
//! - The refusal names the store it was refused on.
//! - The count is an observation of the live readers: two readers park, and the refusal says two.
//! - Each reader in the refusal is one of the processes actually parked on the store — its
//!   operating-system process id, what that process IS, and which run of it — so an operator sent
//!   after "process N" finds the process that is really holding their store.
//! - A free store opens before any reader is parked, so the typed refusal belongs to live reader
//!   contention rather than the harness or a stale registry entry.
//! - A shared writer waits while those readers hold the store, without holding the registry mutex:
//!   its retry observer opens an unrelated store before either reader is released.
//! - Releasing both readers admits that waiting writer, and a later open of the original path
//!   reuses its registered database handle.
//!
//! ## How the condition is produced, and why it takes two processes
//!
//! Direct-reader contention exists only WHILE a committed image is being loaded: a read session
//! that has finished hydrating holds nothing a writer must wait for. So an in-process read session
//! cannot produce it. A second process is parked inside hydration through the shipped
//! hydration checkpoint (`test-seams`), after its breadcrumb is published and its read lock is held.
//! The reader prints the identity it
//! after its breadcrumb is published and its read lock is held, and prints the identity it
//! published. The refusing writer is a third process, so it observes the readers exactly as a
//! separate context-graph client does.
//!
//! Every assertion below is about a value or a pathname. Nothing waits on elapsed time: the parent
//! advances only when a child has printed the marker that says it has arrived.

#![cfg(unix)]

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::channel;
use std::time::Duration;

use context_graph::{CgError, Store, StoreConfig};
use contextdb_engine::ReadSession;
use contextdb_engine::persistence::read_persistence_test_scaffold::arm_read_image_hydration_pause_for_test;

/// Which role a re-invocation of this test binary is playing. Absent in the parent.
const CHILD_ROLE_ENV: &str = "CG_READER_CONTENTION_CHILD_ROLE";
/// The store every child works on.
const CHILD_STORE_ENV: &str = "CG_READER_CONTENTION_STORE_PATH";

/// contextdb's own hydration-checkpoint marker. The parked reader does not print this itself —
/// the shipped checkpoint inside `ReadSession::open` does, after the reader's breadcrumb exists
/// and its read lock is held. That is what makes the wait deterministic rather than timed.
const HYDRATION_HELD: &str = "READ_IMAGE_HYDRATION_HELD=";
/// What the checkpoint waits to read on stdin before letting the reader finish.
const RELEASE: &str = "release";
/// The parked reader finished its open and let the store go.
const READER_RELEASED: &str = "CG_PARKED_READER_RELEASED";
/// What the context-graph writer open came back with.
const WRITER_OPEN: &str = "CG_WRITER_OPEN=";
/// One reader the refusal named.
const WRITER_READER: &str = "CG_WRITER_REFUSED_BY=";

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn decode_hex(encoded: &str) -> Result<Vec<u8>, String> {
    if !encoded.len().is_multiple_of(2) {
        return Err(format!("hex value `{encoded}` has an odd length"));
    }
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let nibble = |byte: u8| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(format!("`{}` is not a hex digit", byte as char)),
            };
            Ok((nibble(pair[0])? << 4) | nibble(pair[1])?)
        })
        .collect()
}

fn child_store_path() -> PathBuf {
    PathBuf::from(std::env::var_os(CHILD_STORE_ENV).expect("a child is given the store to work on"))
}

/// A context-graph store open that asks for no embedder: this file is about who holds the store,
/// and a model download would be a second thing to go wrong.
fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

/// A direct reader parked inside the production `ReadSession::open` of a context-graph store,
/// holding it exactly as a live `cg` read holds it.
#[test]
fn a_parked_direct_reader_child() {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("parked-direct-reader") {
        return;
    }
    arm_read_image_hydration_pause_for_test();
    let session = ReadSession::open(child_store_path())
        .expect("a direct reader opens the committed context-graph store");
    drop(session);
    println!("{READER_RELEASED}");
    std::io::stdout()
        .flush()
        .expect("flush the parked-reader release marker");
}

/// A context-graph writer taking the store through the shipped `Store::open`, reporting what that
/// open came back with. This is the value under test: nothing here builds a `CgError`.
#[test]
fn a_context_graph_writer_open_child() {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("context-graph-writer-open") {
        return;
    }
    // Exercise the typed exhaustion result without holding two reader processes for 60 seconds.
    context_graph::test_support::set_store_open_retry_allowance_for_test(Some(
        std::time::Duration::ZERO,
    ));
    match Store::open(store_config(child_store_path())) {
        Ok(store) => {
            drop(store);
            println!("{WRITER_OPEN}acquired");
        }
        Err(CgError::StoreHeldByReaders {
            observed_direct_readers,
            readers,
            path,
        }) => {
            println!(
                "{WRITER_OPEN}held_by_readers:{observed_direct_readers}:{}",
                encode_hex(path.as_os_str().as_encoded_bytes())
            );
            for reader in readers {
                println!(
                    "{WRITER_READER}{}:{}:{}",
                    reader.process_id,
                    reader.process_start,
                    encode_hex(reader.process_name.as_bytes())
                );
            }
        }
        Err(other) => println!(
            "{WRITER_OPEN}other:{}",
            encode_hex(format!("{other:?}").as_bytes())
        ),
    }
    std::io::stdout()
        .flush()
        .expect("flush the writer-open result");
}

/// What a parked reader published about itself, read back from the shipped checkpoint's marker.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ParkedReader {
    process_id: u32,
    process_name: String,
    process_start: u64,
}

struct ParkedReaderChild {
    child: Child,
    stdout: BufReader<ChildStdout>,
    released: bool,
}

impl ParkedReaderChild {
    fn spawn(store_path: &Path, runtime_directory: &Path) -> Self {
        let mut child =
            Command::new(std::env::current_exe().expect("the current integration-test binary"))
                .arg("--exact")
                .arg("a_parked_direct_reader_child")
                .arg("--nocapture")
                .env(CHILD_ROLE_ENV, "parked-direct-reader")
                .env(CHILD_STORE_ENV, store_path)
                .env("XDG_RUNTIME_DIR", runtime_directory)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("start a parked direct reader");
        let stdout = child.stdout.take().expect("the parked reader's stdout");
        Self {
            child,
            stdout: BufReader::new(stdout),
            released: false,
        }
    }

    fn process_id(&self) -> u32 {
        self.child.id()
    }

    /// Advance only when the reader says it is holding the store, never on elapsed time.
    fn wait_until_parked(&mut self) -> ParkedReader {
        let mut line = String::new();
        while self
            .stdout
            .read_line(&mut line)
            .expect("read the parked reader's markers")
            != 0
        {
            let Some(index) = line.find(HYDRATION_HELD) else {
                line.clear();
                continue;
            };
            let value = line[index + HYDRATION_HELD.len()..].trim().to_owned();
            assert_ne!(
                value, "unverified",
                "a parked reader that published no identity cannot be the one the refusal names"
            );
            let mut fields = value.splitn(4, ':');
            let _breadcrumb_path = fields.next().expect("the marker carries a breadcrumb path");
            let process_id = fields
                .next()
                .expect("the marker carries a process id")
                .parse::<u32>()
                .expect("the marker's process id is a number");
            let process_start = fields
                .next()
                .expect("the marker carries a process start identity")
                .parse::<u64>()
                .expect("the marker's process start identity is a number");
            let process_name = String::from_utf8(
                decode_hex(fields.next().expect("the marker carries a process name"))
                    .expect("the marker's process name is hex"),
            )
            .expect("the marker's process name is text");
            return ParkedReader {
                process_id,
                process_name,
                process_start,
            };
        }
        panic!("the parked reader exited before it reached the store's hydration checkpoint");
    }

    fn release(mut self) {
        let stdin = self
            .child
            .stdin
            .as_mut()
            .expect("the parked reader keeps its release pipe");
        writeln!(stdin, "{RELEASE}").expect("release the parked reader");
        stdin.flush().expect("flush the release");
        let mut line = String::new();
        let mut released = false;
        while self
            .stdout
            .read_line(&mut line)
            .expect("read the parked reader's completion")
            != 0
        {
            if line.contains(READER_RELEASED) {
                released = true;
                break;
            }
            line.clear();
        }
        let status = self.child.wait().expect("wait for the parked reader");
        self.released = true;
        assert!(
            released && status.success(),
            "a released reader finishes its open and exits cleanly: released={released}, \
             status={status}"
        );
    }
}

impl Drop for ParkedReaderChild {
    fn drop(&mut self) {
        if !self.released {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// What the shipped `Store::open` came back with, read back from the writer child.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WriterOpenOutcome {
    Acquired,
    HeldByReaders {
        observed_direct_readers: u64,
        path: PathBuf,
        readers: Vec<ParkedReader>,
    },
    Other(String),
}

fn open_a_context_graph_writer(store_path: &Path, runtime_directory: &Path) -> WriterOpenOutcome {
    let output =
        Command::new(std::env::current_exe().expect("the current integration-test binary"))
            .arg("--exact")
            .arg("a_context_graph_writer_open_child")
            .arg("--nocapture")
            .env(CHILD_ROLE_ENV, "context-graph-writer-open")
            .env(CHILD_STORE_ENV, store_path)
            .env("XDG_RUNTIME_DIR", runtime_directory)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("run the context-graph writer open");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "the writer child must report its outcome rather than fail: {text}"
    );

    let mut readers = Vec::new();
    let mut outcome = None;
    for line in text.lines() {
        if let Some(index) = line.find(WRITER_READER) {
            let mut fields = line[index + WRITER_READER.len()..].trim().splitn(3, ':');
            let process_id = fields
                .next()
                .expect("a refused-by line carries a process id")
                .parse::<u32>()
                .expect("a refused-by process id is a number");
            let process_start = fields
                .next()
                .expect("a refused-by line carries a process start identity")
                .parse::<u64>()
                .expect("a refused-by process start identity is a number");
            let process_name = String::from_utf8(
                decode_hex(
                    fields
                        .next()
                        .expect("a refused-by line carries a process name"),
                )
                .expect("a refused-by process name is hex"),
            )
            .expect("a refused-by process name is text");
            readers.push(ParkedReader {
                process_id,
                process_name,
                process_start,
            });
            continue;
        }
        let Some(index) = line.find(WRITER_OPEN) else {
            continue;
        };
        let value = line[index + WRITER_OPEN.len()..].trim();
        if value == "acquired" {
            outcome = Some(WriterOpenOutcome::Acquired);
        } else if let Some(rest) = value.strip_prefix("held_by_readers:") {
            let mut fields = rest.splitn(2, ':');
            let observed_direct_readers = fields
                .next()
                .expect("the refusal carries an observed reader count")
                .parse::<u64>()
                .expect("the observed reader count is a number");
            let path = PathBuf::from(
                String::from_utf8(
                    decode_hex(fields.next().expect("the refusal carries a store path"))
                        .expect("the refusal's store path is hex"),
                )
                .expect("the refusal's store path is text"),
            );
            outcome = Some(WriterOpenOutcome::HeldByReaders {
                observed_direct_readers,
                path,
                readers: Vec::new(),
            });
        } else if let Some(rest) = value.strip_prefix("other:") {
            outcome = Some(WriterOpenOutcome::Other(
                String::from_utf8(decode_hex(rest).expect("the other-condition text is hex"))
                    .expect("the other-condition text is text"),
            ));
        }
    }
    match outcome.unwrap_or_else(|| panic!("the writer child published no outcome: {text}")) {
        WriterOpenOutcome::HeldByReaders {
            observed_direct_readers,
            path,
            ..
        } => WriterOpenOutcome::HeldByReaders {
            observed_direct_readers,
            path,
            readers,
        },
        settled => settled,
    }
}

/// A runtime root short enough that a local channel pathname below it still fits a Unix socket
/// address, and owner-only so it passes the shipped runtime-root validation.
fn runtime_root() -> tempfile::TempDir {
    let root = tempfile::Builder::new()
        .prefix("cgr")
        .tempdir_in("/tmp")
        .expect("a runtime root short enough for local channel addresses");
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700))
        .expect("make the runtime root owner-only");
    root
}

/// A committed context-graph store for the readers to hold and the writer to be refused on.
fn seeded_store(directory: &Path) -> PathBuf {
    let db_path = directory.join("context-graph.db");
    let store = Store::open(store_config(db_path.clone())).expect("create the context-graph store");
    drop(store);
    db_path
}

#[test]
fn a_real_context_graph_open_is_refused_by_the_live_readers_holding_the_store() {
    let runtime = runtime_root();
    let root = tempfile::tempdir().expect("store root");
    let store_path = seeded_store(root.path());
    let unrelated_directory = root.path().join("unrelated-store");
    fs::create_dir(&unrelated_directory).expect("an unrelated store directory");
    let unrelated = seeded_store(&unrelated_directory);

    drop(
        Store::open(store_config(store_path.clone()))
            .expect("a free store opens, so the busy condition belongs only to a held store"),
    );

    // Two readers, so the reported count cannot be satisfied by a floor a single refusal would
    // carry: only a real observation of the live readers reaches two.
    let mut first = ParkedReaderChild::spawn(&store_path, runtime.path());
    let mut second = ParkedReaderChild::spawn(&store_path, runtime.path());
    let first_parked = first.wait_until_parked();
    let second_parked = second.wait_until_parked();
    assert_eq!(
        first_parked.process_id,
        first.process_id(),
        "the identity the store published is the process this test actually started"
    );
    assert_eq!(
        second_parked.process_id,
        second.process_id(),
        "the identity the store published is the process this test actually started"
    );

    let outcome = open_a_context_graph_writer(&store_path, runtime.path());

    let WriterOpenOutcome::HeldByReaders {
        observed_direct_readers,
        path,
        readers,
    } = &outcome
    else {
        panic!(
            "a context-graph open that meets live direct readers must come back as the busy \
             condition, produced by the open itself — got {outcome:?}"
        );
    };
    assert_eq!(
        path, &store_path,
        "the refusal names the store the writer was refused on, so an operator running several \
         knows which one is busy"
    );
    assert_eq!(
        *observed_direct_readers, 2,
        "two readers are holding the store, so the count is two — it is an observation of the \
         live readers, not a floor"
    );

    let mut named = readers.clone();
    named.sort_by(|left, right| left.process_id.cmp(&right.process_id));
    let mut parked = vec![first_parked.clone(), second_parked.clone()];
    parked.sort_by(|left, right| left.process_id.cmp(&right.process_id));
    assert_eq!(
        named, parked,
        "every field of every reader the refusal names is the field the holding process itself \
         published — the process an operator goes and finds, what it is, and which run of it"
    );

    let (waiting, observed) = channel();
    let (finish, finished) = channel();
    let writer_path = store_path.clone();
    let writer = std::thread::spawn(move || {
        context_graph::test_support::observe_store_open_retries_for_test(move |error| {
            assert!(matches!(error, CgError::StoreHeldByReaders { .. }));
            drop(
                Store::open(store_config(unrelated.clone()))
                    .expect("registry is released while waiting"),
            );
            waiting.send(()).unwrap();
        });
        let store = Store::open(store_config(writer_path)).expect("reader release admits writer");
        finish.send(()).unwrap();
        store
    });
    observed
        .recv_timeout(Duration::from_secs(10))
        .expect("real reader refusal reached the shared Store wait");
    assert!(finished.try_recv().is_err());
    first.release();
    second.release();
    finished
        .recv_timeout(Duration::from_secs(10))
        .expect("writer opens after reader release");
    let store = writer.join().unwrap();
    let shared = Store::open(store_config(store_path)).expect("reuse registered writer");
    assert!(std::sync::Arc::ptr_eq(
        &store.sync_database(),
        &shared.sync_database()
    ));
}
