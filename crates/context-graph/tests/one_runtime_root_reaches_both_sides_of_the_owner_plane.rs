//! One provisioned runtime root has to reach BOTH halves of the owner plane.
//!
//! ## The promise
//!
//! A container, a packaged service, or a Home Assistant add-on has no per-user runtime directory to
//! fall back on, so it provisions one directory and expects everything to meet there. If that root
//! reached only the owner, a packaged deployment would start a `cg` store owner that nobody can
//! ever inspect — and the client would not even be told that: asking in the wrong directory looks
//! exactly like nobody owning the store, so the operator is told "no process is holding the store"
//! about the very process they were pointed at.
//!
//! ## What carries it
//!
//! `CONTEXTDB_OWNER_READ_RUNTIME_DIR` — the substrate's own name, read with the substrate's
//! meaning, naming a DIRECTORY and nothing else. cg invents no name of its own, because the whole
//! point is that ONE value serves both sides: the owner puts its channel where
//! `owner_control::owner_read_config` was told to, and `owner_control::request_owner` asks in
//! exactly that directory through `ReadSession::open_owner_only_in_runtime_dir`. It never says
//! whether owner inspection runs and never what a read may spend.
//!
//! ## How this file proves it rather than assuming it
//!
//! Both halves run as separate PROCESSES with `XDG_RUNTIME_DIR` scrubbed, so nothing in the parent
//! environment can be what made them meet. Three runs together are what make the pin
//! discriminating:
//!
//! - the same root on both sides — the reader gets the owner's own answer back, and the channel is
//!   really inside the root that was provisioned;
//! - DIFFERENT roots — the reader does not reach the owner, so the variable is what carried it and
//!   not some other agreement the two processes could have reached anyway;
//! - the variable unset on both sides, with the platform's own locator in force — the ordinary path
//!   works, and a provisioned root nobody was told about stays empty.
//!
//! Nothing waits on elapsed time: the parent advances when the owner has printed that it is
//! holding the store.

#![cfg(unix)]

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use context_graph::owner_control::request_owner;
use context_graph::{ConsumerDeclaration, ConsumerReader, Store, StoreConfig};

/// Which role a re-invocation of this test binary is playing. Absent in the parent.
const CHILD_ROLE_ENV: &str = "CG_OWNER_PLANE_CHILD_ROLE";
/// The store both halves are pointed at.
const CHILD_STORE_ENV: &str = "CG_OWNER_PLANE_STORE_PATH";
/// The one variable an operator provisions for the whole owner plane.
const RUNTIME_DIR_ENV: &str = "CONTEXTDB_OWNER_READ_RUNTIME_DIR";

/// The administrative frame the reader sends and the answer only a live owner can produce.
const FRAME: &str = "who-holds-this-store";
const ANSWER: &str = "the owner answered from inside the provisioned runtime root";

const OWNER_HOLDING: &str = "CG_OWNER_HOLDING";
const OWNER_RELEASED: &str = "CG_OWNER_RELEASED";
const RELEASE: &str = "release";
const READER_ANSWER: &str = "CG_READER_ANSWER=";
const READER_REFUSED: &str = "CG_READER_REFUSED=";
const CONSUMER_OPENED: &str = "CG_CONSUMER_OPENED";
const CONSUMER_REFUSED: &str = "CG_CONSUMER_REFUSED=";

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn decode_hex(encoded: &str) -> String {
    assert!(
        encoded.len().is_multiple_of(2),
        "hex value `{encoded}` has an odd length"
    );
    let bytes: Vec<u8> = encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let nibble = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                other => panic!("`{}` is not a hex digit", other as char),
            };
            (nibble(pair[0]) << 4) | nibble(pair[1])
        })
        .collect();
    String::from_utf8(bytes).expect("a marker payload is text")
}

fn child_store_path() -> PathBuf {
    PathBuf::from(std::env::var_os(CHILD_STORE_ENV).expect("a child is given the store"))
}

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

/// The store OWNER: it takes the store and answers administrative frames for as long as it holds
/// it, exactly as a running context-graph does.
#[test]
fn a_store_owner_child() {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("store-owner") {
        return;
    }
    let store = Store::open_with_control_handler(
        store_config(child_store_path()),
        std::sync::Arc::new(|frame: String| {
            assert_eq!(frame, FRAME, "the owner is asked the frame the reader sent");
            ANSWER.to_string()
        }),
    )
    .expect("the owner child takes the store and serves administrative frames");
    println!("{OWNER_HOLDING}");
    std::io::stdout()
        .flush()
        .expect("flush the owner-holding marker");
    let mut command = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut command)
        .expect("read the owner release command");
    assert_eq!(command.trim(), RELEASE);
    drop(store);
    println!("{OWNER_RELEASED}");
    std::io::stdout()
        .flush()
        .expect("flush the owner-released marker");
}

/// A separate read-only `cg` asking the owner a question — the owner-only door, which never touches
/// the store's file.
#[test]
fn an_owner_only_reader_child() {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("owner-only-reader") {
        return;
    }
    match request_owner(&child_store_path(), FRAME) {
        Ok(answer) => println!("{READER_ANSWER}{}", encode_hex(answer.as_bytes())),
        Err(refusal) => println!(
            "{READER_REFUSED}{}",
            encode_hex(format!("{refusal:?}").as_bytes())
        ),
    }
    std::io::stdout()
        .flush()
        .expect("flush the reader's outcome");
}

/// A separate consumer opening a stopped store through the ordinary owner-or-file read door.
/// The parent gives this child a valid provisioned root and a deliberately unusable platform
/// locator, so only the provisioned root can make the read session available.
#[test]
fn a_stopped_store_consumer_child() {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("stopped-store-consumer") {
        return;
    }
    match ConsumerReader::open(child_store_path(), ConsumerDeclaration::default()) {
        Ok(_reader) => println!("{CONSUMER_OPENED}"),
        Err(refusal) => println!(
            "{CONSUMER_REFUSED}{}",
            encode_hex(format!("{refusal:?}").as_bytes())
        ),
    }
    std::io::stdout()
        .flush()
        .expect("flush the consumer's outcome");
}

/// How a child is told where the owner plane lives. `XDG_RUNTIME_DIR` is always scrubbed first, so
/// a child that finds the plane can only have found it through what this names.
#[derive(Clone, Copy)]
enum RuntimeLocation<'a> {
    /// The operator provisioned this directory through the one variable.
    Provisioned(&'a Path),
    /// Nothing was provisioned; the platform's own locator is in force.
    PlatformLocator(&'a Path),
}

fn apply(command: &mut Command, location: RuntimeLocation<'_>) {
    command.env_remove("XDG_RUNTIME_DIR");
    command.env_remove(RUNTIME_DIR_ENV);
    match location {
        RuntimeLocation::Provisioned(root) => {
            command.env(RUNTIME_DIR_ENV, root);
        }
        RuntimeLocation::PlatformLocator(root) => {
            command.env("XDG_RUNTIME_DIR", root);
        }
    }
}

struct OwnerChild {
    child: Child,
    stdout: BufReader<ChildStdout>,
    released: bool,
}

impl OwnerChild {
    /// Start the owner and return only once it says it is holding the store.
    fn holding(store_path: &Path, location: RuntimeLocation<'_>) -> Self {
        let mut command =
            Command::new(std::env::current_exe().expect("the current integration-test binary"));
        command
            .arg("--exact")
            .arg("a_store_owner_child")
            .arg("--nocapture")
            .env(CHILD_ROLE_ENV, "store-owner")
            .env(CHILD_STORE_ENV, store_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        apply(&mut command, location);
        let mut child = command.spawn().expect("start the store owner");
        let stdout = child.stdout.take().expect("the owner's stdout");
        let mut owner = Self {
            child,
            stdout: BufReader::new(stdout),
            released: false,
        };
        let mut line = String::new();
        loop {
            let read = owner
                .stdout
                .read_line(&mut line)
                .expect("read the owner's markers");
            assert_ne!(read, 0, "the owner exited before it took the store: {line}");
            if line.contains(OWNER_HOLDING) {
                return owner;
            }
            line.clear();
        }
    }

    fn release(mut self) {
        let stdin = self.child.stdin.as_mut().expect("the owner's release pipe");
        writeln!(stdin, "{RELEASE}").expect("release the owner");
        stdin.flush().expect("flush the release");
        let mut line = String::new();
        let mut released = false;
        while self
            .stdout
            .read_line(&mut line)
            .expect("read the owner's completion")
            != 0
        {
            if line.contains(OWNER_RELEASED) {
                released = true;
                break;
            }
            line.clear();
        }
        let status = self.child.wait().expect("wait for the owner");
        self.released = true;
        assert!(
            released && status.success(),
            "a released owner lets the store go and exits cleanly: released={released}, \
             status={status}"
        );
    }
}

impl Drop for OwnerChild {
    fn drop(&mut self) {
        if !self.released {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// What the separate read-only command came back with.
#[derive(Debug, PartialEq, Eq)]
enum ReaderOutcome {
    Answered(String),
    Refused(String),
}

fn ask_the_owner(store_path: &Path, location: RuntimeLocation<'_>) -> ReaderOutcome {
    let mut command =
        Command::new(std::env::current_exe().expect("the current integration-test binary"));
    command
        .arg("--exact")
        .arg("an_owner_only_reader_child")
        .arg("--nocapture")
        .env(CHILD_ROLE_ENV, "owner-only-reader")
        .env(CHILD_STORE_ENV, store_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply(&mut command, location);
    let output = command.output().expect("run the owner-only reader");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "the reader must report an outcome rather than fail: {text}"
    );
    for line in text.lines() {
        if let Some(index) = line.find(READER_ANSWER) {
            return ReaderOutcome::Answered(decode_hex(line[index + READER_ANSWER.len()..].trim()));
        }
        if let Some(index) = line.find(READER_REFUSED) {
            return ReaderOutcome::Refused(decode_hex(line[index + READER_REFUSED.len()..].trim()));
        }
    }
    panic!("the reader published no outcome: {text}");
}

fn open_the_stopped_store(
    store_path: &Path,
    provisioned_root: &Path,
    unusable_platform_root: &Path,
) -> ReaderOutcome {
    let mut command =
        Command::new(std::env::current_exe().expect("the current integration-test binary"));
    command
        .arg("--exact")
        .arg("a_stopped_store_consumer_child")
        .arg("--nocapture")
        .env(CHILD_ROLE_ENV, "stopped-store-consumer")
        .env(CHILD_STORE_ENV, store_path)
        .env(RUNTIME_DIR_ENV, provisioned_root)
        .env("XDG_RUNTIME_DIR", unusable_platform_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = command.output().expect("run the stopped-store consumer");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "the consumer must report an outcome rather than fail: {text}"
    );
    for line in text.lines() {
        if line.contains(CONSUMER_OPENED) {
            return ReaderOutcome::Answered(CONSUMER_OPENED.to_string());
        }
        if let Some(index) = line.find(CONSUMER_REFUSED) {
            return ReaderOutcome::Refused(decode_hex(
                line[index + CONSUMER_REFUSED.len()..].trim(),
            ));
        }
    }
    panic!("the consumer published no outcome: {text}");
}

/// A root short enough that a local channel pathname below it still fits a Unix socket address, and
/// owner-only so it passes the shipped runtime-root validation.
fn provisioned_root(prefix: &str) -> tempfile::TempDir {
    let root = tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in("/tmp")
        .expect("a runtime root short enough for local channel addresses");
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700))
        .expect("make the runtime root owner-only");
    root
}

fn store_directory() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("cgs")
        .tempdir_in("/tmp")
        .expect("a store directory")
}

/// Every path below `root`, so what the owner plane actually placed there can be named.
fn entries_below(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(listing) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in listing.flatten() {
            let path = entry.path();
            if fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_dir()) {
                pending.push(path.clone());
            }
            found.push(path);
        }
    }
    found
}

fn holds_a_socket(root: &Path) -> bool {
    entries_below(root)
        .iter()
        .any(|path| fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_socket()))
}

#[test]
fn one_provisioned_root_reaches_the_owner_and_a_separate_read_only_command() {
    let runtime = provisioned_root("cgo");
    let stores = store_directory();
    let store_path = stores.path().join("context-graph.db");

    let owner = OwnerChild::holding(&store_path, RuntimeLocation::Provisioned(runtime.path()));

    assert!(
        holds_a_socket(runtime.path()),
        "the owner put its channel in the directory it was provisioned with, or the root reached \
         only half the plane: {:?}",
        entries_below(runtime.path())
    );

    assert_eq!(
        ask_the_owner(&store_path, RuntimeLocation::Provisioned(runtime.path())),
        ReaderOutcome::Answered(ANSWER.to_string()),
        "a separate read-only command given the SAME root reaches the live owner and gets the \
         owner's own answer — which only a running owner can produce"
    );

    owner.release();

    let companion_path = contextdb_core::store_companion_path(&store_path);
    let store_before = fs::read(&store_path).expect("read the stopped store before inspection");
    let companion_before =
        fs::read(&companion_path).expect("read the stopped store's companion before inspection");
    let unusable_platform_root = runtime.path().join("platform-root-does-not-exist");
    assert!(!unusable_platform_root.exists());
    assert_eq!(
        open_the_stopped_store(&store_path, runtime.path(), &unusable_platform_root),
        ReaderOutcome::Answered(CONSUMER_OPENED.to_string()),
        "the same provisioned root must reach an ordinary consumer read after the owner stops; \
         an unusable platform locator cannot make a correctly supplied deployment unreadable"
    );
    assert_eq!(
        fs::read(&store_path).expect("read the stopped store after inspection"),
        store_before,
        "the consumer read must not change the stopped store"
    );
    assert_eq!(
        fs::read(&companion_path).expect("read the stopped store's companion after inspection"),
        companion_before,
        "the consumer read must not change the stopped store's companion"
    );
}

#[test]
fn a_reader_pointed_at_a_different_root_does_not_reach_the_owner() {
    let owners_root = provisioned_root("cgo");
    let other_root = provisioned_root("cgx");
    let stores = store_directory();
    let store_path = stores.path().join("context-graph.db");

    let owner = OwnerChild::holding(
        &store_path,
        RuntimeLocation::Provisioned(owners_root.path()),
    );

    // The discriminator. If the reader reached the owner from a root the owner never used, the
    // root was not what carried it and the run above would prove nothing.
    let outcome = ask_the_owner(&store_path, RuntimeLocation::Provisioned(other_root.path()));
    assert!(
        !matches!(&outcome, ReaderOutcome::Answered(answer) if answer == ANSWER),
        "a reader asking in a directory the owner never used must not reach it — otherwise the \
         provisioned root is not what the two halves are meeting on: {outcome:?}"
    );
    assert!(
        entries_below(other_root.path())
            .iter()
            .all(|path| !fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_socket())),
        "nothing put a channel in a root that only the reader was told about"
    );

    owner.release();
}

#[test]
fn with_nothing_provisioned_the_platform_locator_works_exactly_as_it_did() {
    let platform = provisioned_root("cgp");
    let unused = provisioned_root("cgu");
    let stores = store_directory();
    let store_path = stores.path().join("context-graph.db");

    // Neither side is told about `unused`; both fall back to the platform's own locator, which is
    // the path every existing deployment already takes.
    let owner = OwnerChild::holding(
        &store_path,
        RuntimeLocation::PlatformLocator(platform.path()),
    );

    assert_eq!(
        ask_the_owner(
            &store_path,
            RuntimeLocation::PlatformLocator(platform.path())
        ),
        ReaderOutcome::Answered(ANSWER.to_string()),
        "the ordinary owner plane works when a deployment provisions \
         nothing still has a reader that reaches its owner"
    );
    assert!(
        entries_below(unused.path()).is_empty(),
        "a directory nobody was told about stays empty — the variable is read, never guessed: {:?}",
        entries_below(unused.path())
    );

    owner.release();
}
