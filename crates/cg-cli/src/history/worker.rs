//! The detached one-shot event-search worker and the foreground `--search wait` turn.
//!
//! Both run the library's `EventSearchProjectionRunner`; the only thing this module owns is the
//! process boundary: the advisory `event-search-worker.lock` file (held through `cg-node-fs`, and
//! taken by no other worker kind), a silent detached process for background mode, and polling for
//! the same lock in wait mode so there is never a second builder. The Store-enforced projection
//! lease stays the correctness boundary if two callers race.

use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cg_node_fs::{WorkerLock, WorkerLockRefusal, acquire_worker_lock};
use context_graph::owner_control::{
    READER_RELEASE_ALLOWANCE, ReaderRelease, wait_for_reader_release_within,
};
use context_graph::{
    CgError, ContextId, ContextSelector, EmbedderRegistration, EventHistorySearchState,
    EventHistorySearchStatus, EventSearchProfile, EventSearchProjectionRunner,
    EventSearchRunRequest,
};

use super::{EVENT_SEARCH_WORKER_LOCK, db_path, open_reader, search_registration, store_config};

/// The hidden subcommand a background import launches. It is not a user surface.
pub(crate) const WORKER_COMMAND: &str = "__event-search-worker";

/// Bounded passes a detached worker makes before it exits, so a Context whose projection keeps
/// failing cannot keep a process alive.
const WORKER_PASSES: usize = 3;

/// Runner passes `--search wait` makes before reporting that search is still behind.
const FOREGROUND_PASSES: usize = 16;

/// What status says about a Context the worker gave up on because the store stayed held by readers.
/// Fixed text: it names the condition and nothing about this machine.
const HELD_BY_READERS_REFUSAL: &str =
    "the search index could not be built: other processes were reading the store throughout";

/// The longest a refused step pauses before asking again when no reader can be observed to wait on.
const UNOBSERVABLE_BACKOFF_CEILING: Duration = Duration::from_millis(500);

/// Wait for the store's direct readers to let go, inside `deadline`.
///
/// A reader refusing a writer is a moment, not a broken store: readers publish themselves only for
/// the span in which they load a committed image, so the wait below is answered by the last of them
/// letting go. A wait nobody could observe says nothing about whether the store is free, so it is
/// not taken as a reason to stop — the caller simply asks again after a growing pause, inside the
/// same allowance. `false` means the allowance is spent and the caller must report rather than wait.
fn wait_for_readers(root: &Path, deadline: Instant, backoff: &mut Duration) -> bool {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return false;
    }
    match wait_for_reader_release_within(&db_path(root), deadline) {
        ReaderRelease::Released => true,
        ReaderRelease::Stopped => false,
        ReaderRelease::Unobservable(_) => {
            std::thread::sleep((*backoff).min(remaining));
            *backoff = (*backoff * 2).min(UNOBSERVABLE_BACKOFF_CEILING);
            true
        }
    }
}

fn lock(root: &Path) -> Option<WorkerLock> {
    acquire_worker_lock(&root.join(EVENT_SEARCH_WORKER_LOCK)).ok()
}

fn run_request(
    root: &Path,
    context_id: ContextId,
    embedder: EmbedderRegistration,
) -> EventSearchRunRequest {
    EventSearchRunRequest {
        store: store_config(root, None),
        contexts: ContextSelector::one(context_id),
        embedder,
        profile: EventSearchProfile::default(),
        max_events_per_batch: NonZeroU32::new(256).expect("non-zero"),
        max_passages_per_batch: NonZeroU32::new(32).expect("non-zero"),
        max_bytes_per_batch: NonZeroUsize::new(4 * 1024 * 1024).expect("non-zero"),
        lease_duration_ms: NonZeroU64::new(120_000).expect("non-zero"),
    }
}

/// Every Context with uncovered captured events that no live projection is already serving, or
/// `None` when the store would not answer.
///
/// A refusal is not an answer. Direct readers holding the store are waited out here, and a question
/// that still cannot be put is reported as unknown rather than as "no work": only `cg import` and
/// `cg forget` start a worker, so a worker that mistakes a refused read for an empty queue strands
/// every pending Context until the user imports again.
fn contexts_with_pending_work(root: &Path) -> Option<Vec<ContextId>> {
    let deadline = Instant::now() + READER_RELEASE_ALLOWANCE;
    let mut backoff = Duration::from_millis(20);
    loop {
        match pending_once(root) {
            Ok(contexts) => return Some(contexts),
            Err(CgError::StoreHeldByReaders { .. }) => {
                if !wait_for_readers(root, deadline, &mut backoff) {
                    return None;
                }
            }
            Err(_) => return None,
        }
    }
}

fn pending_once(root: &Path) -> context_graph::Result<Vec<ContextId>> {
    let reader = open_reader(root)?;
    let contexts = reader.graph().list_contexts()?;
    let Ok(selector) = ContextSelector::new(contexts.iter().map(|context| context.id)) else {
        return Ok(Vec::new());
    };
    Ok(reader
        .event_history_status(selector)?
        .contexts
        .into_iter()
        .filter(|context| {
            context.search.pending_events > 0
                && context.search.state != EventHistorySearchState::Running
        })
        .map(|context| context.context_id)
        .collect())
}

/// Hand pending work to a detached worker. Returns how it was handed over.
pub(crate) fn kick(root: &Path) -> Result<&'static str, String> {
    match acquire_worker_lock(&root.join(EVENT_SEARCH_WORKER_LOCK)) {
        // A worker already holds the lock; it re-checks for pending work after releasing it, so
        // this invocation's work is still scheduled to it.
        Err(WorkerLockRefusal::HeldByAnotherWorker) => Ok("already_running"),
        // Nothing holds the lock and nothing can take it, so no worker is running and none was
        // started. Saying "already running" here would promise the user a worker that does not
        // exist, and the import would report the search as scheduled when nothing is.
        Err(refusal) => Err(format!(
            "the background search worker could not be started: {refusal}"
        )),
        Ok(held) => {
            drop(held);
            let exe = std::env::current_exe()
                .map_err(|error| format!("cannot locate the cg executable: {error}"))?;
            Command::new(exe)
                .arg(WORKER_COMMAND)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()
                .map(|_| "spawned")
                .map_err(|error| format!("could not start the background search worker: {error}"))
        }
    }
}

/// Catch up each Context once; a Context whose projection fails this pass is not retried in it.
fn catch_up(root: &Path, embedder: &EmbedderRegistration, failed: &mut BTreeSet<ContextId>) {
    let mut attempts = std::collections::BTreeMap::<ContextId, usize>::new();
    loop {
        let Some(known) = contexts_with_pending_work(root) else {
            return;
        };
        let pending = known
            .into_iter()
            .filter(|context_id| !failed.contains(context_id))
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return;
        }
        for context_id in pending {
            serve(root, embedder, context_id, &mut attempts, failed);
        }
    }
}

/// Build one Context's projection, retrying IN PLACE while direct readers refuse it.
///
/// Retrying here rather than by going back to the pending list is the point. An attempt the store
/// refuses part-way can leave this Context's projection lease behind, and a live lease reads back
/// as `Running` — "somebody is already serving this" — so a worker that re-derived its queue would
/// drop the very Context it was started for and exit. Only `cg import` and `cg forget` start a
/// worker, so nothing would pick that Context up again: `cg status` would answer `pending`, with no
/// recorded failure, until the user imported a second time.
fn serve(
    root: &Path,
    embedder: &EmbedderRegistration,
    context_id: ContextId,
    attempts: &mut std::collections::BTreeMap<ContextId, usize>,
    failed: &mut BTreeSet<ContextId>,
) {
    let deadline = Instant::now() + READER_RELEASE_ALLOWANCE;
    let mut backoff = Duration::from_millis(20);
    loop {
        let attempt = attempts.entry(context_id).or_default();
        *attempt += 1;
        match EventSearchProjectionRunner::catch_up(run_request(root, context_id, embedder.clone()))
        {
            // Readers holding the store are a moment, not a broken Context: wait for the last of
            // them to let go and put the same question again. That costs no projection attempt.
            // When the allowance is spent the Context is given up on, but never silently: status
            // must say `failed` with the reason, not `pending` for work nobody is doing.
            Err(CgError::StoreHeldByReaders { .. }) => {
                *attempt -= 1;
                if !wait_for_readers(root, deadline, &mut backoff) {
                    let _ = EventSearchProjectionRunner::record_refusal(
                        run_request(root, context_id, embedder.clone()),
                        HELD_BY_READERS_REFUSAL,
                    );
                    failed.insert(context_id);
                    return;
                }
            }
            Err(_) => {
                failed.insert(context_id);
                return;
            }
            // A Context that still has work after repeated successful passes is left for the next
            // worker rather than spun on here.
            Ok(_) => {
                if *attempt >= WORKER_PASSES {
                    failed.insert(context_id);
                }
                return;
            }
        }
    }
}

/// The detached worker: take the lock or exit, catch up, release, and re-check for work committed
/// while the lock was held so no import's pending work is stranded.
pub(crate) fn handle_event_search_worker(_args: &[String]) -> i32 {
    let root = super::super::data_root();
    let embedder = match search_registration() {
        Ok(embedder) => embedder,
        Err(_) => return 1,
    };
    let mut failed = BTreeSet::new();
    for _ in 0..WORKER_PASSES {
        let Some(held) = lock(&root) else {
            return 0;
        };
        catch_up(&root, &embedder, &mut failed);
        drop(held);
        // A store that would not answer leaves the queue unknown; that costs another pass, never a
        // silent "everything is done".
        if let Some(still) = contexts_with_pending_work(&root)
            && still.iter().all(|context_id| failed.contains(context_id))
        {
            return 0;
        }
    }
    0
}

/// `--search wait`: obtain the worker turn, then run the same runner in the foreground until this
/// Context is caught up or the projection fails.
pub(crate) fn run_in_foreground(
    root: &Path,
    context_id: ContextId,
) -> Result<EventHistorySearchStatus, String> {
    let embedder = search_registration().map_err(|error| error.to_string())?;
    // Only a lock another worker holds is worth waiting for: that worker finishes and lets go. A
    // lock that cannot be used is not a turn that is coming, so waiting for it would hang this
    // invocation forever instead of telling the user their search did not run.
    let held = loop {
        match acquire_worker_lock(&root.join(EVENT_SEARCH_WORKER_LOCK)) {
            Ok(held) => break held,
            Err(WorkerLockRefusal::HeldByAnotherWorker) => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(refusal) => {
                return Err(format!(
                    "the search worker turn could not be taken: {refusal}"
                ));
            }
        }
    };
    let result = (|| {
        for _ in 0..FOREGROUND_PASSES {
            let status = open_reader(root)
                .and_then(|reader| reader.event_history_status(ContextSelector::one(context_id)))
                .map_err(|error| error.to_string())?
                .contexts
                .remove(0)
                .search;
            if status.pending_events == 0 {
                return Ok(status);
            }
            EventSearchProjectionRunner::catch_up(run_request(root, context_id, embedder.clone()))
                .map_err(|error| error.to_string())?;
        }
        Err("search did not catch up with history that kept arriving; run the same import with --search wait again".to_string())
    })();
    drop(held);
    result
}
