//! Node-deployment filesystem primitives for cg.
//!
//! These are the three on-disk touches a cg *node* makes that are NOT part of the embeddable memory
//! core: reading a per-node config file, holding the per-data-root worker lock, and replacing the
//! bounded latest-value event-search worker policy. They live here, one layer out from
//! `context-graph`, precisely so the embeddable core (`context-graph`, which
//! vigil and any WASM target link) stays free of filesystem access — its purity guard forbids any
//! `std::fs` in the core `src/`, because the core must never fetch evidence bytes or probe the disk.
//! A node runtime (the `cg` CLI, or `context_graph::fabric` on a real deployment) consumes these;
//! an embed never does.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Relative name of the bounded latest-value event-search execution policy.
pub const EVENT_SEARCH_WORKER_POLICY: &str = "event-search-worker-policy.json";

/// Relative name of the short lock which serializes worker-policy record updates.
pub const EVENT_SEARCH_WORKER_POLICY_UPDATE_LOCK: &str = "event-search-worker-policy.lock";

/// The byte bound shared by worker records and their reserved refusal reports.
pub const EVENT_SEARCH_WORKER_POLICY_MAX_BYTES: u64 = 4 * 1024;

/// A short-lived guard for one read/modify/write operation on the worker-policy record.
///
/// It deliberately protects only record IO. Callers must drop it before taking a worker lock,
/// opening a store, embedding, or starting a process.
pub struct EventSearchWorkerPolicyUpdateGuard {
    root: PathBuf,
    _file: std::fs::File,
}

/// Take the per-data-root policy-update lock.
pub fn acquire_event_search_worker_policy_update(
    root: &Path,
) -> std::io::Result<EventSearchWorkerPolicyUpdateGuard> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(EVENT_SEARCH_WORKER_POLICY_UPDATE_LOCK))?;
    use fs4::fs_std::FileExt;
    // This is an OS wait, not a polling loop. Policy updates are short, but a contended update
    // must yield to the holder instead of burning a CPU while it waits.
    file.lock_exclusive()?;
    Ok(EventSearchWorkerPolicyUpdateGuard {
        root: root.to_path_buf(),
        _file: file,
    })
}

impl EventSearchWorkerPolicyUpdateGuard {
    /// Read the record while this guard serializes a following replacement.
    pub fn read(&self) -> std::io::Result<Option<String>> {
        read_event_search_worker_policy(&self.root)
    }

    /// Atomically replace the record while this guard remains held.
    pub fn write(&self, contents: &str) -> std::io::Result<()> {
        write_event_search_worker_policy(&self.root, contents)
    }
}

pub fn read_event_search_worker_policy(root: &Path) -> std::io::Result<Option<String>> {
    let path = root.join(EVENT_SEARCH_WORKER_POLICY);
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut bytes = Vec::new();
    file.take(EVENT_SEARCH_WORKER_POLICY_MAX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > EVENT_SEARCH_WORKER_POLICY_MAX_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "event-search worker policy exceeds 4 KiB",
        ));
    }
    String::from_utf8(bytes).map(Some).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("event-search worker policy is not UTF-8: {error}"),
        )
    })
}

/// Atomically replace the bounded event-search worker policy in a data root.
pub fn write_event_search_worker_policy(root: &Path, contents: &str) -> std::io::Result<()> {
    if contents.len() as u64 > EVENT_SEARCH_WORKER_POLICY_MAX_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "event-search worker policy exceeds 4 KiB",
        ));
    }
    let mut temporary = tempfile::NamedTempFile::new_in(root)?;
    temporary.write_all(contents.as_bytes())?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(root.join(EVENT_SEARCH_WORKER_POLICY))
        .map(|_| ())
        .map_err(|error| error.error)
}

/// A held per-data-root worker lock. Dropping it releases the lock. The lock is an OS advisory
/// exclusive lock on a lockfile, so a second holder on the same data root — even in the same
/// process, via a distinct open — is refused while the first is held.
#[derive(Debug)]
pub struct WorkerLock {
    _file: std::fs::File,
}

/// Why the per-data-root worker lock was not taken.
///
/// The two variants are opposite answers to the only question a caller asks — "is my work going to
/// be done?" — so they must never be collapsed into one. A lock somebody holds means a worker is
/// running and this caller is a no-op. Anything else means no worker is running and none can be
/// started, which the caller owes its user.
#[derive(Debug)]
pub enum WorkerLockRefusal {
    /// Another worker holds the lock. It does the work; this caller stands down.
    HeldByAnotherWorker,
    /// The lockfile could not be opened, or the lock could not be asked for. Nothing holds it and
    /// nothing is running because of this.
    Unusable(std::io::Error),
}

impl std::fmt::Display for WorkerLockRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HeldByAnotherWorker => formatter.write_str("another worker holds the lock"),
            Self::Unusable(error) => write!(formatter, "the lock file could not be used: {error}"),
        }
    }
}

/// Take the per-data-root worker lock at `lock_path`.
///
/// A caller that only wants to know whether it holds the lock reads the `Ok`; a caller that must
/// tell a user whether its work was scheduled reads the refusal, because only
/// [`WorkerLockRefusal::HeldByAnotherWorker`] means somebody else is doing it.
pub fn acquire_worker_lock(lock_path: &Path) -> Result<WorkerLock, WorkerLockRefusal> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .map_err(WorkerLockRefusal::Unusable)?;
    use fs4::fs_std::FileExt;
    match file.try_lock_exclusive() {
        Ok(true) => Ok(WorkerLock { _file: file }),
        Ok(false) => Err(WorkerLockRefusal::HeldByAnotherWorker),
        Err(error) => Err(WorkerLockRefusal::Unusable(error)),
    }
}

/// Load a config file's text, mapping a missing file to `Ok(None)` (an absent config is not an
/// error — the node runs with defaults). Any other IO error surfaces as `Err`.
pub fn load_text_if_present(path: &Path) -> std::io::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}

/// How much memory this machine actually has available to a process running here.
///
/// Two numbers, because a container is told a smaller one than the box it runs on: `total` is the
/// machine's installed memory, and `cgroup` is the ceiling the kernel enforces on this process's
/// control group when one is set. The caller takes the smaller of them — a process in a 2 GiB
/// container on a 128 GiB host lives under 2 GiB, and sizing anything off the host's number is how
/// a container gets killed instead of refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MachineMemory {
    /// Installed memory, from `/proc/meminfo` on Linux or `hw.memsize` on macOS. `None` where the
    /// platform does not publish it.
    pub total_bytes: Option<u64>,
    /// The control group's own memory ceiling, when the kernel enforces a finite one.
    pub cgroup_limit_bytes: Option<u64>,
}

/// Read what this machine will actually let a process here allocate.
///
/// Every read is best-effort: a platform that publishes neither number answers `None` for both and
/// the caller falls back to its own floor rather than failing. Nothing here dereferences a source
/// reference or reaches the network — these are the kernel's own published memory facts.
pub fn machine_memory() -> MachineMemory {
    MachineMemory {
        total_bytes: read_installed_memory(),
        cgroup_limit_bytes: read_cgroup_memory_limit(),
    }
}

/// Installed physical memory, in bytes, from the platform's own published source.
fn read_installed_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        read_macos_memsize()
    }
    #[cfg(not(target_os = "macos"))]
    {
        read_proc_meminfo_total()
    }
}

/// Darwin publishes installed physical memory in bytes as the `hw.memsize` sysctl. The system
/// utility is addressed by absolute path, so the answer does not depend on the caller's `PATH`,
/// and no shell is involved.
#[cfg(target_os = "macos")]
fn read_macos_memsize() -> Option<u64> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let bytes = std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    (bytes > 0).then_some(bytes)
}

/// `MemTotal:` from `/proc/meminfo`, in bytes. The file publishes kibibytes.
#[cfg(not(target_os = "macos"))]
fn read_proc_meminfo_total() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kibibytes = text
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kibibytes.checked_mul(1024)
}

/// The memory ceiling on this process's own control group, following the unified (v2) hierarchy
/// first and then the legacy (v1) one. `max` — cgroup v2's word for "no ceiling" — and v1's
/// unlimited sentinel both answer `None`, because neither is a number to size anything from.
fn read_cgroup_memory_limit() -> Option<u64> {
    if let Some(bytes) = unified_cgroup_memory_max() {
        return Some(bytes);
    }
    legacy_cgroup_memory_limit()
}

fn unified_cgroup_memory_max() -> Option<u64> {
    // `/proc/self/cgroup` names this process's path inside the unified hierarchy as `0::<path>`.
    // The ceiling in force is the tightest one on the way down to it, because an ancestor's limit
    // binds every descendant.
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = own
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim()
        .trim_start_matches('/');
    let mut directory = PathBuf::from("/sys/fs/cgroup");
    let mut tightest = read_memory_max(&directory.join("memory.max"));
    for segment in relative.split('/').filter(|segment| !segment.is_empty()) {
        directory.push(segment);
        if let Some(bytes) = read_memory_max(&directory.join("memory.max")) {
            tightest = Some(tightest.map_or(bytes, |held: u64| held.min(bytes)));
        }
    }
    tightest
}

fn read_memory_max(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim();
    if trimmed == "max" {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

fn legacy_cgroup_memory_limit() -> Option<u64> {
    let text = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes").ok()?;
    let bytes = text.trim().parse::<u64>().ok()?;
    // v1 spells "no ceiling" as a number near the top of the address space rather than as a word.
    // Anything at or above a pebibyte is that sentinel, not a limit anyone set.
    (bytes < (1 << 50)).then_some(bytes)
}
