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

/// Acquire the per-data-root worker lock at `lock_path`. Returns `None` when another worker already
/// holds it (the second worker is then a no-op) or when the lockfile cannot be opened.
pub fn acquire_worker_lock(lock_path: &Path) -> Option<WorkerLock> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .ok()?;
    use fs4::fs_std::FileExt;
    match file.try_lock_exclusive() {
        Ok(true) => Some(WorkerLock { _file: file }),
        _ => None,
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
