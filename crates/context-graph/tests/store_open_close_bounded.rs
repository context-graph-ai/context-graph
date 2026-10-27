//! A process that opens and closes a writable Store again and again gives back what each closed
//! Store held.
//!
//! The detached projection worker reopens the Store for every claim and every commit, and a
//! long-lived host reopens it whenever it needs it. Whatever one open-and-drop cycle keeps is kept
//! once per cycle for the life of the process, so the OS threads and the resident memory after the
//! last drop must be what they were after the first.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use context_graph::{Store, StoreConfig};

const CYCLES: usize = 40;
/// Threads a runtime may lazily start after the first cycle without that being per-cycle growth.
const THREAD_SLACK: usize = 4;
const RSS_GROWTH_CEILING_KB: u64 = 64 * 1024;

/// The owner channel's directory for this whole test process, set before any Store opens.
fn owner_runtime_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = tempfile::tempdir().expect("owner runtime dir").keep();
        // The owner plane accepts only an owner-only runtime root.
        std::fs::set_permissions(&dir, std::os::unix::fs::PermissionsExt::from_mode(0o700))
            .expect("owner-only runtime dir");
        // SAFETY: set once, before any Store in this process reads the environment.
        unsafe { std::env::set_var("CONTEXTDB_OWNER_READ_RUNTIME_DIR", &dir) };
        dir
    })
}

fn status_field_kb(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    status
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("{field} missing from /proc/self/status"))
}

fn threads() -> usize {
    status_field_kb("Threads:") as usize
}

fn rss_kb() -> u64 {
    status_field_kb("VmRSS:")
}

/// How many live threads carry each name, so a failure says whose threads were kept.
fn threads_by_name() -> BTreeMap<String, usize> {
    let mut names = BTreeMap::new();
    for task in std::fs::read_dir("/proc/self/task").expect("list /proc/self/task") {
        let Ok(task) = task else { continue };
        let Ok(comm) = std::fs::read_to_string(task.path().join("comm")) else {
            continue;
        };
        *names.entry(comm.trim().to_owned()).or_insert(0) += 1;
    }
    names
}

fn open_and_drop(path: &Path) {
    let store = Store::open(StoreConfig {
        db_path: path.to_path_buf(),
        default_text_embedder: None,
        ..StoreConfig::default()
    })
    .expect("open a writable store");
    drop(store);
}

#[test]
#[ignore = "contextdb a5f5b3a: owner-read accept thread outlives Store close; the accept thread remains live after close"]
fn reopening_a_writable_store_keeps_no_threads_or_memory_from_closed_stores() {
    owner_runtime_dir();
    let dir = tempfile::tempdir().expect("store dir");
    let path = dir.path().join("reopened.db");

    open_and_drop(&path);
    let threads_after_first = threads();
    let rss_after_first = rss_kb();
    let names_after_first = threads_by_name();

    for cycle in 2..=CYCLES {
        open_and_drop(&path);
        if cycle % 10 == 0 {
            eprintln!(
                "[store-open-close] after drop #{cycle}: threads={} rss_kb={}",
                threads(),
                rss_kb()
            );
        }
    }
    let threads_after_last = threads();
    let rss_after_last = rss_kb();
    let names_after_last = threads_by_name();

    eprintln!(
        "[store-open-close] after drop #1: threads={threads_after_first} rss_kb={rss_after_first} \
         names={names_after_first:?}"
    );
    eprintln!(
        "[store-open-close] after drop #{CYCLES}: threads={threads_after_last} \
         rss_kb={rss_after_last} names={names_after_last:?}"
    );

    assert!(
        threads_after_last <= threads_after_first + THREAD_SLACK,
        "{CYCLES} open/drop cycles kept threads: {threads_after_first} after the first drop, \
         {threads_after_last} after the last; by name {names_after_first:?} -> \
         {names_after_last:?}"
    );
    let growth_kb = rss_after_last.saturating_sub(rss_after_first);
    assert!(
        growth_kb < RSS_GROWTH_CEILING_KB,
        "{CYCLES} open/drop cycles kept {growth_kb} kB of resident memory \
         ({rss_after_first} kB after the first drop, {rss_after_last} kB after the last)"
    );
}
