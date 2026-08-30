//! Putting a real `cg` command in front of a store somebody else is holding.
//!
//! Two conditions an operator meets on a machine where nothing is broken:
//!
//! - another process is WRITING the store (a running `cg` or a runtime that owns it), and
//! - another process is READING the store and has not finished loading the committed image yet.
//!
//! Both are produced here the only way they can be produced honestly — by other PROCESSES. Reader
//! contention in particular exists only WHILE a committed image is being loaded, so a finished read
//! session holds nothing; the reader is parked inside contextdb's shipped hydration checkpoint
//! (`test-seams`), which stops it after its breadcrumb is published and its read lock is held, and
//! prints the identity it published. Nothing here waits on elapsed time: the harness advances only
//! when a child has printed the marker that says it has arrived.
//!
//! The runtime root is handed to every participant as `XDG_RUNTIME_DIR`, which is where a deployed
//! reader and a deployed writer meet today, so the readers publish into the directory the `cg`
//! process scans.

#![allow(dead_code)]

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use context_graph::{Store, StoreConfig};
use contextdb_engine::ReadSession;
use contextdb_engine::persistence::read_persistence_test_scaffold::arm_read_image_hydration_pause_for_test;

/// Which role a re-invocation of this test binary is playing. Absent in the parent.
pub const CHILD_ROLE_ENV: &str = "CG_CLI_STORE_CONTENTION_CHILD_ROLE";
/// The store every child works on.
pub const CHILD_STORE_ENV: &str = "CG_CLI_STORE_CONTENTION_STORE_PATH";

/// contextdb's own hydration-checkpoint marker, printed by the shipped checkpoint rather than by
/// the child, after the reader's breadcrumb exists and its read lock is held.
const HYDRATION_HELD: &str = "READ_IMAGE_HYDRATION_HELD=";
const RELEASE: &str = "release";
const READER_RELEASED: &str = "CG_CLI_PARKED_READER_RELEASED";
const WRITER_HOLDING: &str = "CG_CLI_WRITER_HOLDING";
const WRITER_RELEASED: &str = "CG_CLI_WRITER_RELEASED";

/// The file a `cg` data root keeps its memory store in.
pub fn store_path_in(data_root: &Path) -> PathBuf {
    data_root.join("context-graph.db")
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn decode_hex(encoded: &str) -> Vec<u8> {
    assert!(
        encoded.len().is_multiple_of(2),
        "hex value `{encoded}` has an odd length"
    );
    encoded
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
        .collect()
}

fn child_store_path() -> PathBuf {
    PathBuf::from(std::env::var_os(CHILD_STORE_ENV).expect("a child is given the store to work on"))
}

/// A store open that asks for no embedder: these files are about who holds the store, and a model
/// download would be a second thing to go wrong.
fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

/// The body of the parked-reader child. A test file registers it under a `#[test]` of its own,
/// because a child is started by naming a test in this same binary.
pub fn run_parked_reader_child_if_asked() -> bool {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("parked-direct-reader") {
        return false;
    }
    arm_read_image_hydration_pause_for_test();
    let session = ReadSession::open(child_store_path())
        .expect("a direct reader opens the committed context-graph store");
    drop(session);
    println!("{READER_RELEASED}");
    std::io::stdout()
        .flush()
        .expect("flush the parked-reader release marker");
    true
}

/// The body of the writer-holder child: it takes the store the way a running context-graph does and
/// keeps it until released.
pub fn run_writer_holder_child_if_asked() -> bool {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("writer-holder") {
        return false;
    }
    let store =
        Store::open(store_config(child_store_path())).expect("the writer child takes the store");
    println!("{WRITER_HOLDING}");
    std::io::stdout()
        .flush()
        .expect("flush the writer-holding marker");
    let mut command = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut command)
        .expect("read the writer release command");
    assert_eq!(command.trim(), RELEASE);
    drop(store);
    println!("{WRITER_RELEASED}");
    std::io::stdout()
        .flush()
        .expect("flush the writer-released marker");
    true
}

/// What a parked reader published about itself, read back from the shipped checkpoint's marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParkedReader {
    pub process_id: u32,
    pub process_name: String,
    pub process_start: u64,
}

/// A child process holding the store, either as a parked direct reader or as a writer.
pub struct HoldingChild {
    child: Child,
    stdout: BufReader<ChildStdout>,
    released: bool,
}

impl HoldingChild {
    fn spawn(role: &str, test_name: &str, store_path: &Path, runtime_directory: &Path) -> Self {
        let mut child =
            Command::new(std::env::current_exe().expect("the current integration-test binary"))
                .arg("--exact")
                .arg(test_name)
                .arg("--nocapture")
                .env(CHILD_ROLE_ENV, role)
                .env(CHILD_STORE_ENV, store_path)
                .env("XDG_RUNTIME_DIR", runtime_directory)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap_or_else(|error| panic!("start the {role} child: {error}"));
        let stdout = child.stdout.take().expect("the holding child's stdout");
        Self {
            child,
            stdout: BufReader::new(stdout),
            released: false,
        }
    }

    pub fn park_a_reader(store_path: &Path, runtime_directory: &Path, test_name: &str) -> Self {
        Self::spawn(
            "parked-direct-reader",
            test_name,
            store_path,
            runtime_directory,
        )
    }

    pub fn hold_as_writer(store_path: &Path, runtime_directory: &Path, test_name: &str) -> Self {
        Self::spawn("writer-holder", test_name, store_path, runtime_directory)
    }

    pub fn process_id(&self) -> u32 {
        self.child.id()
    }

    fn wait_for(&mut self, marker: &str) -> String {
        let mut line = String::new();
        while self
            .stdout
            .read_line(&mut line)
            .expect("read the holding child's markers")
            != 0
        {
            if let Some(index) = line.find(marker) {
                return line[index + marker.len()..].trim().to_owned();
            }
            line.clear();
        }
        panic!("the holding child exited before it reached `{marker}`");
    }

    /// Advance only when the reader says it is holding the store, never on elapsed time.
    pub fn wait_until_parked(&mut self) -> ParkedReader {
        let value = self.wait_for(HYDRATION_HELD);
        assert_ne!(
            value, "unverified",
            "a parked reader that published no identity cannot be the one a refusal names"
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
        let process_name = String::from_utf8(decode_hex(
            fields.next().expect("the marker carries a process name"),
        ))
        .expect("the marker's process name is text");
        ParkedReader {
            process_id,
            process_name,
            process_start,
        }
    }

    /// Advance only when the writer says it has the store.
    pub fn wait_until_holding_as_writer(&mut self) {
        self.wait_for(WRITER_HOLDING);
    }

    pub fn release(mut self) {
        let stdin = self
            .child
            .stdin
            .as_mut()
            .expect("the holding child keeps its release pipe");
        writeln!(stdin, "{RELEASE}").expect("release the holding child");
        stdin.flush().expect("flush the release");
        let mut line = String::new();
        let mut released = false;
        while self
            .stdout
            .read_line(&mut line)
            .expect("read the holding child's completion")
            != 0
        {
            if line.contains(READER_RELEASED) || line.contains(WRITER_RELEASED) {
                released = true;
                break;
            }
            line.clear();
        }
        let status = self.child.wait().expect("wait for the holding child");
        self.released = true;
        assert!(
            released && status.success(),
            "a released holder lets the store go and exits cleanly: released={released}, \
             status={status}"
        );
    }
}

impl Drop for HoldingChild {
    fn drop(&mut self) {
        if !self.released {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// What one `cg` invocation said and how it ended.
pub struct CgRun {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl CgRun {
    /// Everything the operator sees, wherever it was written.
    pub fn everything_said(&self) -> String {
        format!("{}{}", self.stdout, self.stderr)
    }
}

/// Run the compiled `cg` against `data_root`, meeting the holders below `runtime_directory`.
pub fn run_cg_below_runtime_root(
    args: &[&str],
    data_root: &Path,
    runtime_directory: &Path,
) -> CgRun {
    let output = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_cg")))
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", data_root)
        .env("XDG_RUNTIME_DIR", runtime_directory)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("run the compiled cg binary");
    CgRun {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// A runtime root short enough that a local channel pathname below it still fits a Unix socket
/// address, and owner-only so it passes the shipped runtime-root validation.
pub fn runtime_root() -> tempfile::TempDir {
    let root = tempfile::Builder::new()
        .prefix("cgr")
        .tempdir_in("/tmp")
        .expect("a runtime root short enough for local channel addresses");
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700))
        .expect("make the runtime root owner-only");
    root
}

/// A `cg` data root holding a committed store, created by the shipped binary itself so the file
/// every holder and every later command meets is the one `cg` really writes.
pub fn seeded_data_root(runtime_directory: &Path) -> tempfile::TempDir {
    let data_root = tempfile::Builder::new()
        .prefix("cgd")
        .tempdir_in("/tmp")
        .expect("a cg data root");
    let seeding = run_cg_below_runtime_root(&["impact"], data_root.path(), runtime_directory);
    let store_path = store_path_in(data_root.path());
    assert!(
        store_path.exists(),
        "seeding must leave a committed store at {}: cg said {}",
        store_path.display(),
        seeding.everything_said()
    );
    data_root
}

/// Debug renderings of the busy condition that must never reach an operator: a Rust value printed
/// at somebody instead of a sentence written for them.
pub const DEBUG_DUMP_FRAGMENTS: [&str; 4] = [
    "StoreHeldByReaders {",
    "observed_direct_readers:",
    "ReaderIdentity {",
    "process_start:",
];

/// Wording that tells an operator the engine failed, on a machine where nothing has.
pub const READS_AS_A_BROKEN_ENGINE: [&str; 3] = ["unexpected", "engine(", "panicked at"];

pub fn encode_for_marker(bytes: &[u8]) -> String {
    encode_hex(bytes)
}
