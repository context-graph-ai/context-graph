//! Keeps the one test in this binary that must run alone from overlapping any sibling test.
//!
//! The reindex test that bounds how many rows the scan reads while 24 writers race it
//! (`reindex::reindex_concurrent_writes_during_reindex_window_land_in_target_space_deterministically`)
//! fails its read budget when other tests in this process run alongside it. nextest already
//! runs it alone through the `runs-alone` test group in `.config/nextest.toml`; plain
//! `cargo test` has no test groups, so this binary carries its own process-wide lock instead.
//! That test holds the lock exclusively; every other test in this binary holds it shared for its
//! whole body, so the exclusive holder never overlaps any of them at any `--test-threads` count.
//!
//! The guards ignore poisoning: a sibling that panicked while holding the lock has already
//! reported its own failure, and the lock guards no data, only overlap.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

static RUNS_ALONE: RwLock<()> = RwLock::new(());

/// Held for the whole body of every test in this binary that may run alongside its siblings.
///
/// It is a named holder rather than a bare lock guard because it is deliberately held across
/// `.await` in the async tests: the point is to cover the test's whole body, and the only waiter
/// on the other side is the single exclusive test, which never holds anything an async test
/// awaits on.
pub struct RunsAlongsideSiblings {
    _guard: RwLockReadGuard<'static, ()>,
}

/// Held for the whole body of the one test that must not overlap any sibling.
pub struct RunsAlone {
    _guard: RwLockWriteGuard<'static, ()>,
}

/// Take the shared side: may run alongside any test except the one that runs alone.
pub fn alongside_siblings() -> RunsAlongsideSiblings {
    RunsAlongsideSiblings {
        _guard: RUNS_ALONE
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    }
}

/// Take the exclusive side: waits until no sibling test is running, then keeps them all out.
pub fn alone() -> RunsAlone {
    RunsAlone {
        _guard: RUNS_ALONE
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    }
}
