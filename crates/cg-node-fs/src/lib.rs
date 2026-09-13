//! Node-deployment filesystem primitives for cg.
//!
//! These are the two on-disk touches a cg *node* makes that are NOT part of the embeddable memory
//! core: reading a per-node config file, and holding the per-data-root worker lock. They live here,
//! one layer out from `context-graph`, precisely so the embeddable core (`context-graph`, which
//! vigil and any WASM target link) stays free of filesystem access — its purity guard forbids any
//! `std::fs` in the core `src/`, because the core must never fetch evidence bytes or probe the disk.
//! A node runtime (the `cg` CLI, or `context_graph::fabric` on a real deployment) consumes these;
//! an embed never does.

use std::path::Path;

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
