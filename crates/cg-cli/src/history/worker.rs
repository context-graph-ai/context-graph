//! The detached one-shot event-search worker and the foreground `--search wait` turn.
//!
//! Both run the library's `EventSearchProjectionRunner`; the only thing this module owns is the
//! process boundary: the advisory `event-search-worker.lock` file (held through `cg-node-fs`, and
//! taken by no other worker kind), a silent detached process for background mode, and polling for
//! the same lock in wait mode so there is never a second builder. The Store-enforced projection
//! lease stays the correctness boundary if two callers race.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cg_node_fs::{
    WorkerLock, WorkerLockRefusal, acquire_event_search_worker_policy_update, acquire_worker_lock,
    read_event_search_worker_policy,
};
use context_graph::owner_control::{
    READER_RELEASE_ALLOWANCE, ReaderRelease, WRITER_TRANSITION_ALLOWANCE,
    wait_for_reader_release_within,
};
use context_graph::{
    CgError, ContextId, ContextSelector, EmbedderRegistration, EventSearchBatchWriteLimit,
    EventSearchFailureDisposition, EventSearchProfile, EventSearchProjectionRunner,
    EventSearchRunOutcome, EventSearchRunRequest, EventSearchSelectedWork,
    EventSearchWorkAssessment, EventSearchWorkPolicy,
};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use super::{
    EVENT_SEARCH_WORKER_LOCK, configured_passage_profile, db_path, open_reader,
    resolved_embedding_workers, resolved_search_work_policy, search_registration, store_config,
};

/// The hidden subcommand a background import launches. It is not a user surface.
pub(crate) const WORKER_COMMAND: &str = "__event-search-worker";

/// Bound unknown queue reads, unchanged work, and consecutive typed coordination refusals.
/// Successful new work spends no retry allowance.
const WORKER_RETRIES: usize = 3;

/// Bound repeated identical work in the foreground, independently of successful new work.
const FOREGROUND_UNCHANGED_RETRIES: usize = 16;

/// The longest a refused step pauses before asking again when no reader can be observed to wait on.
const UNOBSERVABLE_BACKOFF_CEILING: Duration = Duration::from_millis(500);

#[derive(Deserialize, Serialize)]
struct WorkerEnvelope {
    profile: EventSearchProfile,
    embedding_workers: NonZeroUsize,
    #[serde(default)]
    work_policy: EventSearchWorkPolicy,
    ignored_publication: Option<String>,
    /// Only the child handed off by an interrupted foreground turn waits for admission. Normal
    /// detached workers retain their existing one-shot admission semantics.
    #[serde(default)]
    retry_worker_lock: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WorkerPolicyPublication {
    publication: String,
    embedding_workers: NonZeroUsize,
    #[serde(default)]
    work_policy: EventSearchWorkPolicy,
    #[serde(default)]
    pending_background_request: Option<PendingBackgroundRequest>,
    /// Foreground refusals have no wake request and never alter accepted background selections.
    #[serde(default)]
    foreground_read_refusals: Vec<ForegroundReadRefusal>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ForegroundReadRefusal {
    token: String,
    context_id: ContextId,
    /// A refused selection requires verified full current-Context completion on retry.
    selected_work: Option<EventSearchSelectedWork>,
    limit_ms: u64,
    ordinal: u64,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PendingBackgroundRequest {
    token: String,
    /// Keep each import's selected ceiling; derive the shared minimum only at admission.
    #[serde(default)]
    write_requests: Vec<PendingWriteRequest>,
    #[serde(default)]
    write_memory_refusals: BTreeMap<String, BackgroundMemoryRefusal>,
    embedding_workers: NonZeroUsize,
    #[serde(default)]
    work_policy: EventSearchWorkPolicy,
    ignored_publication: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PendingWriteRequest {
    token: String,
    context_id: ContextId,
    memory_limit_bytes: NonZeroUsize,
    #[serde(default = "context_graph::owner_control::default_read_active_time_ms")]
    read_active_time_ms: NonZeroU64,
    /// Consecutive active-time refusals without committed coverage progress, per selection.
    #[serde(default)]
    read_timeout_no_progress: usize,
    #[serde(default)]
    read_timeout_covered_events: u64,
    /// A restart can finish acknowledging an exhausted refusal only after the core recorded it.
    #[serde(default)]
    read_timeout_failure_recorded: bool,
    #[serde(default)]
    read_timeout_ordinal: u64,
    #[serde(default)]
    read_timeout_limit_ms: u64,
    work_policy: EventSearchWorkPolicy,
    selected_work: uuid::Uuid,
    #[serde(default)]
    completed: bool,
    #[serde(default)]
    refused_memory: bool,
    memory_refusal: PendingMemoryRefusal,
    /// A retry supersedes a verified complete refusal but keeps its unretired selection.
    refusal_superseded: bool,
    /// Keep the saved ceiling until selection retirement commits, including after restart.
    selection_retired: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
enum PendingMemoryRefusal {
    NotRecorded,
    Recorded {
        ordinal: u64,
        refusal: BackgroundMemoryRefusal,
    },
}

impl PendingWriteRequest {
    fn has_terminal_refusal(&self) -> bool {
        self.refused_memory || self.read_timeout_no_progress >= WORKER_RETRIES
    }
}

impl PendingBackgroundRequest {
    /// Rendering is derived from actual active identities, never a removed identity's cache.
    fn rebuild_memory_refusals(&mut self) -> Result<(), String> {
        let mut latest = BTreeMap::new();
        for request in &self.write_requests {
            if request.refused_memory && !request.refusal_superseded {
                let PendingMemoryRefusal::Recorded { ordinal, refusal } = &request.memory_refusal
                else {
                    return Err(format!(
                        "active memory refusal {} has no recorded facts",
                        request.token
                    ));
                };
                let entry = latest
                    .entry(request.context_id.to_string())
                    .or_insert((*ordinal, refusal.clone()));
                if *ordinal > entry.0 {
                    *entry = (*ordinal, refusal.clone());
                }
            }
        }
        self.write_memory_refusals = latest
            .into_iter()
            .map(|(context, (_, refusal))| (context, refusal))
            .collect();
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackgroundMemoryRefusal {
    requested_bytes: usize,
    limit_bytes: usize,
}

impl BackgroundMemoryRefusal {
    pub(crate) fn message(&self) -> String {
        CgError::WriteDidNotFitTheMemoryLimit {
            requested_bytes: self.requested_bytes,
            limit_bytes: self.limit_bytes,
        }
        .to_string()
    }
}

struct EmbeddingWorkerPolicyResolver {
    root: PathBuf,
    last_valid: NonZeroUsize,
    last_work_policy: EventSearchWorkPolicy,
    ignored_publication: Option<String>,
}

impl EmbeddingWorkerPolicyResolver {
    fn new(
        root: &Path,
        initial: NonZeroUsize,
        initial_work_policy: EventSearchWorkPolicy,
        ignored_publication: Option<String>,
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            last_valid: initial,
            last_work_policy: initial_work_policy,
            ignored_publication,
        }
    }

    fn current(&self) -> NonZeroUsize {
        self.last_valid
    }

    fn current_work_policy(&self) -> EventSearchWorkPolicy {
        self.last_work_policy
    }

    fn resolve_next_claim(&mut self) -> NonZeroUsize {
        if let Ok(Some(contents)) = read_event_search_worker_policy(&self.root)
            && let Ok(next) = decode_worker_policy(&contents)
            && self.ignored_publication.as_deref() != Some(next.publication.as_str())
        {
            self.last_valid = next.embedding_workers;
            self.last_work_policy = next.work_policy;
        }
        self.last_valid
    }
}

fn decode_worker_policy(contents: &str) -> Result<WorkerPolicyPublication, String> {
    let mut policy: WorkerPolicyPublication =
        serde_json::from_str(contents).map_err(|error| error.to_string())?;
    if let Some(request) = &mut policy.pending_background_request {
        request.rebuild_memory_refusals()?;
    }
    Ok(policy)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Acceptance retains its committed coverage baseline beside its selected frontier"
)]
fn publish_embedding_worker_policy(
    root: &Path,
    context_id: ContextId,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
    selected_work: uuid::Uuid,
    preserve_choice: bool,
    completed_refusal_tokens: &[String],
    covered_events: u64,
) -> Result<PendingBackgroundRequest, String> {
    update_worker_policy(root, |existing| {
        let mut policy = existing.unwrap_or_else(|| WorkerPolicyPublication {
            publication: uuid::Uuid::new_v4().to_string(),
            embedding_workers,
            work_policy,
            pending_background_request: None,
            foreground_read_refusals: Vec::new(),
        });
        if !preserve_choice {
            policy.publication = uuid::Uuid::new_v4().to_string();
            policy.embedding_workers = embedding_workers;
            policy.work_policy = work_policy;
        }
        let previous = policy.pending_background_request.take();
        let (effective_workers, effective_policy, ignored_publication) = if preserve_choice {
            previous
                .as_ref()
                .map(|request| {
                    (
                        request.embedding_workers,
                        request.work_policy,
                        request.ignored_publication.clone(),
                    )
                })
                .unwrap_or((
                    embedding_workers,
                    work_policy,
                    Some(policy.publication.clone()),
                ))
        } else {
            (embedding_workers, work_policy, None)
        };
        let mut write_requests = previous
            .map(|request| request.write_requests)
            .unwrap_or_default();
        // Only an observed refusal whose original saved frontier is complete is superseded.
        // Keep its old ceiling until durable retirement; later and incomplete identities remain
        // active even when the new accepted selection is narrower or completes successfully.
        for request in &mut write_requests {
            if request.context_id == context_id
                && request.has_terminal_refusal()
                && completed_refusal_tokens.contains(&request.token)
            {
                request.refusal_superseded = true;
            }
        }
        // The accepted token supplies the wake for completed ordinary selections.
        write_requests.retain(|request| {
            !request.completed
                || !request.selection_retired
                || (request.has_terminal_refusal() && !request.refusal_superseded)
        });
        let token = uuid::Uuid::new_v4().to_string();
        write_requests.push(PendingWriteRequest {
            token: token.clone(),
            context_id,
            memory_limit_bytes: crate::write_config::in_force().bytes,
            read_active_time_ms: crate::read_config::active_time_in_force(),
            read_timeout_no_progress: 0,
            read_timeout_covered_events: covered_events,
            read_timeout_failure_recorded: false,
            read_timeout_ordinal: 0,
            read_timeout_limit_ms: 0,
            work_policy,
            selected_work,
            completed: false,
            refused_memory: false,
            memory_refusal: PendingMemoryRefusal::NotRecorded,
            refusal_superseded: false,
            selection_retired: false,
        });
        let mut request = PendingBackgroundRequest {
            token,
            write_requests,
            write_memory_refusals: BTreeMap::new(),
            embedding_workers: effective_workers,
            work_policy: effective_policy,
            ignored_publication,
        };
        request.rebuild_memory_refusals()?;
        policy.pending_background_request = Some(request.clone());
        Ok((policy, request))
    })
}

/// Foreground wait publishes its fixed cap for any later background owner but never creates or
/// clears a wake request of its own.
fn publish_foreground_embedding_worker_policy(
    root: &Path,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
) -> Result<(), String> {
    update_worker_policy(root, |existing| {
        let mut policy = existing.unwrap_or_else(|| WorkerPolicyPublication {
            publication: String::new(),
            embedding_workers,
            work_policy,
            pending_background_request: None,
            foreground_read_refusals: Vec::new(),
        });
        policy.publication = uuid::Uuid::new_v4().to_string();
        policy.embedding_workers = embedding_workers;
        policy.work_policy = work_policy;
        Ok((policy, ()))
    })
}

/// Publish a latest background execution choice and a separate coalesced wake request. The update
/// guard prevents a foreground publication or an older worker clear from losing either field.
fn update_worker_policy<T>(
    root: &Path,
    update: impl FnOnce(Option<WorkerPolicyPublication>) -> Result<(WorkerPolicyPublication, T), String>,
) -> Result<T, String> {
    let guard = acquire_event_search_worker_policy_update(root).map_err(|error| {
        format!("the event-search worker policy could not be delivered: {error}")
    })?;
    let existing = guard
        .read()
        .map_err(|error| format!("the event-search worker policy could not be read: {error}"))?
        .map(|contents| decode_worker_policy(&contents))
        .transpose()
        .map_err(|error| format!("the event-search worker policy is malformed: {error}"))?;
    let (policy, result) = update(existing)?;
    if !worker_policy_fits(&policy) {
        return Err("the bounded search request record has no room for another import and its memory refusal; retry after pending requests finish".into());
    }
    let contents = serde_json::to_string(&policy).expect("event-search worker policy serializes");
    guard.write(&contents).map_err(|error| {
        format!("the event-search worker policy could not be delivered: {error}")
    })?;
    Ok(result)
}

/// Preserve room for each accepted request's possible terminal facts and their Context cache.
fn worker_policy_fits(policy: &WorkerPolicyPublication) -> bool {
    let serialized_len = |value: &WorkerPolicyPublication| {
        serde_json::to_vec(value)
            .expect("worker policy serializes")
            .len() as u64
    };
    let maximum = cg_node_fs::EVENT_SEARCH_WORKER_POLICY_MAX_BYTES;
    if serialized_len(policy) > maximum {
        return false;
    }
    let mut reported = policy.clone();
    if let Some(request) = &mut reported.pending_background_request {
        let mut selection_read_limits = BTreeMap::<(ContextId, uuid::Uuid), u64>::new();
        for pending in &request.write_requests {
            let limit = selection_read_limits
                .entry((pending.context_id, pending.selected_work))
                .or_default();
            *limit = (*limit).max(pending.read_active_time_ms.get());
        }
        for pending in &mut request.write_requests {
            let mut read_refused = pending.clone();
            read_refused.read_timeout_no_progress =
                read_refused.read_timeout_no_progress.max(WORKER_RETRIES);
            read_refused.read_timeout_covered_events = u64::MAX;
            read_refused.read_timeout_ordinal = u64::MAX;
            // Assessment can use any owner's allowance, and its refusal limit is saved to
            // every owner of that selection. Preserve any recorded value as well.
            read_refused.read_timeout_limit_ms = read_refused
                .read_timeout_limit_ms
                .max(selection_read_limits[&(pending.context_id, pending.selected_work)]);
            read_refused.read_timeout_failure_recorded = false;

            // Memory can fail after transient read timeouts, so its terminal state must retain
            // room for that accounting too. Its limit comes from the saved writer ceiling (or
            // a lower shared ceiling), while the required bytes remain unconstrained.
            let recorded_limit = match &pending.memory_refusal {
                PendingMemoryRefusal::NotRecorded => 0,
                PendingMemoryRefusal::Recorded { refusal, .. } => refusal.limit_bytes,
            };
            let refusal = BackgroundMemoryRefusal {
                requested_bytes: usize::MAX,
                limit_bytes: pending.memory_limit_bytes.get().max(recorded_limit),
            };
            let mut memory_refused = read_refused.clone();
            memory_refused.read_timeout_no_progress =
                pending.read_timeout_no_progress.max(WORKER_RETRIES - 1);
            memory_refused.completed = true;
            memory_refused.refused_memory = true;
            memory_refused.memory_refusal = PendingMemoryRefusal::Recorded {
                ordinal: u64::MAX,
                refusal: refusal.clone(),
            };
            // Entry lengths add independently: selecting each larger outcome also bounds a
            // mixture of memory and active-time refusals, without combining terminal states.
            let entry_len = |entry: &PendingWriteRequest| {
                serde_json::to_vec(entry)
                    .expect("pending write request serializes")
                    .len()
            };
            *pending = if entry_len(&memory_refused) >= entry_len(&read_refused) {
                memory_refused
            } else {
                read_refused
            };
            // The cache holds only the latest memory refusal per Context. Reserve its largest
            // possible saved limit independently of which entry supplies that latest ordinal.
            let cached = request
                .write_memory_refusals
                .entry(pending.context_id.to_string())
                .or_insert_with(|| refusal.clone());
            cached.requested_bytes = usize::MAX;
            cached.limit_bytes = cached.limit_bytes.max(refusal.limit_bytes);
        }
    }
    serialized_len(&reported) <= maximum
}

fn worker_policy_checked(root: &Path) -> Result<Option<WorkerPolicyPublication>, String> {
    read_event_search_worker_policy(root)
        .map_err(|error| format!("the event-search worker policy could not be read: {error}"))?
        .map(|contents| decode_worker_policy(&contents))
        .transpose()
}

fn next_refusal_ordinal(policy: &WorkerPolicyPublication) -> Result<u64, String> {
    policy
        .foreground_read_refusals
        .iter()
        .map(|request| request.ordinal)
        .chain(
            policy
                .pending_background_request
                .iter()
                .flat_map(|pending| &pending.write_requests)
                .map(|entry| {
                    let memory = match entry.memory_refusal {
                        PendingMemoryRefusal::Recorded { ordinal, .. } => ordinal,
                        PendingMemoryRefusal::NotRecorded => 0,
                    };
                    entry.read_timeout_ordinal.max(memory)
                }),
        )
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| "read refusal ordinal exhausted".into())
}

fn record_foreground_read_refusal(
    root: &Path,
    context_id: ContextId,
    selected_work: Option<EventSearchSelectedWork>,
    limit_ms: u64,
) -> Result<(), String> {
    update_worker_policy(root, |existing| {
        let mut policy = existing.ok_or("the foreground search policy is missing")?;
        let ordinal = next_refusal_ordinal(&policy)?;
        let refusal = ForegroundReadRefusal {
            token: uuid::Uuid::new_v4().to_string(),
            context_id,
            selected_work,
            limit_ms,
            ordinal,
        };
        policy.foreground_read_refusals.push(refusal.clone());
        if !worker_policy_fits(&policy) {
            // Large all-session selections and repeated terminal retries still owe durable status.
            // Collapse only this Context to the conservative full-completion refusal.
            policy
                .foreground_read_refusals
                .retain(|saved| saved.context_id != context_id);
            policy.foreground_read_refusals.push(ForegroundReadRefusal {
                selected_work: None,
                ..refusal
            });
        }
        Ok((policy, ()))
    })
}

fn completed_foreground_read_refusals(
    store: &context_graph::Store,
    context_id: ContextId,
    spec: &context_graph::EventSearchWorkSpec,
    observed: &[ForegroundReadRefusal],
) -> context_graph::Result<Vec<String>> {
    let mut completed = Vec::new();
    for refusal in observed
        .iter()
        .filter(|refusal| refusal.context_id == context_id)
    {
        let selected_work = match &refusal.selected_work {
            Some(selected) => selected.clone(),
            None => store.select_event_search_work(
                context_id,
                EventSearchWorkPolicy {
                    sessions: context_graph::EventSearchSessionScope::All,
                    ..EventSearchWorkPolicy::default()
                },
            )?,
        };
        if store.event_search_selected_work_complete(context_id, spec, &selected_work)? {
            completed.push(refusal.token.clone());
        }
    }
    Ok(completed)
}

fn acknowledge_foreground_read_refusals(root: &Path, observed: &[String]) -> Result<(), String> {
    if observed.is_empty() {
        return Ok(());
    }
    update_worker_policy(root, |existing| {
        let mut policy = existing.ok_or("the foreground search policy is missing")?;
        policy
            .foreground_read_refusals
            .retain(|refusal| !observed.contains(&refusal.token));
        Ok((policy, ()))
    })
}

fn pending_background_request_checked(
    root: &Path,
) -> Result<Option<PendingBackgroundRequest>, String> {
    read_event_search_worker_policy(root)
        .map_err(|error| format!("the event-search worker policy could not be read: {error}"))?
        .map(|contents| decode_worker_policy(&contents))
        .transpose()
        .map(|policy| policy.and_then(|policy| policy.pending_background_request))
}

fn pending_background_request(root: &Path) -> Option<PendingBackgroundRequest> {
    pending_background_request_checked(root).ok().flatten()
}

pub(crate) enum BackgroundWorkerRefusal {
    Memory(BackgroundMemoryRefusal),
    ReadActiveTime { limit_ms: u64 },
}

impl BackgroundWorkerRefusal {
    pub(crate) fn message(&self) -> String {
        match self {
            Self::Memory(refusal) => refusal.message(),
            Self::ReadActiveTime { limit_ms } => format!(
                "{}; committed search progress is retained; run the same import with --search wait again",
                CgError::ReadActiveTimeExceeded {
                    limit_ms: *limit_ms
                },
            ),
        }
    }
}

pub(crate) fn background_memory_refusals(
    root: &Path,
) -> Result<BTreeMap<String, BackgroundWorkerRefusal>, String> {
    let Some(policy) = worker_policy_checked(root)? else {
        return Ok(BTreeMap::new());
    };
    let mut latest = BTreeMap::<String, (u64, BackgroundWorkerRefusal)>::new();
    for refusal in policy.foreground_read_refusals {
        let entry = latest
            .entry(refusal.context_id.to_string())
            .or_insert_with(|| {
                (
                    refusal.ordinal,
                    BackgroundWorkerRefusal::ReadActiveTime {
                        limit_ms: refusal.limit_ms,
                    },
                )
            });
        if refusal.ordinal > entry.0 {
            *entry = (
                refusal.ordinal,
                BackgroundWorkerRefusal::ReadActiveTime {
                    limit_ms: refusal.limit_ms,
                },
            );
        }
    }
    for request in policy
        .pending_background_request
        .into_iter()
        .flat_map(|pending| pending.write_requests)
    {
        if request.refusal_superseded {
            continue;
        }
        let refusal = if request.read_timeout_no_progress >= WORKER_RETRIES {
            Some((
                request.read_timeout_ordinal,
                BackgroundWorkerRefusal::ReadActiveTime {
                    limit_ms: if request.read_timeout_limit_ms == 0 {
                        request.read_active_time_ms.get()
                    } else {
                        request.read_timeout_limit_ms
                    },
                },
            ))
        } else if request.refused_memory {
            match request.memory_refusal {
                PendingMemoryRefusal::Recorded { ordinal, refusal } => {
                    Some((ordinal, BackgroundWorkerRefusal::Memory(refusal)))
                }
                PendingMemoryRefusal::NotRecorded => None,
            }
        } else {
            None
        };
        if let Some(refusal) = refusal {
            match latest.entry(request.context_id.to_string()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(refusal);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if refusal.0 >= entry.get().0 {
                        entry.insert(refusal);
                    }
                }
            }
        }
    }
    let refusals = latest
        .into_iter()
        .map(|(context, (_, refusal))| (context, refusal))
        .collect();
    Ok(refusals)
}

/// Verify this import and each observed refusal on the original admitted writer. The subsequent
/// acknowledgement must not erase a later refusal or an incomplete broader accepted frontier.
pub(crate) fn completed_memory_refusals(
    root: &Path,
    store: &context_graph::Store,
    context_id: ContextId,
    work_policy: EventSearchWorkPolicy,
    profile: &EventSearchProfile,
) -> Result<Vec<String>, CgError> {
    let Some(policy) = worker_policy_checked(root).map_err(CgError::Engine)? else {
        return Ok(Vec::new());
    };
    let pending = policy.pending_background_request.as_ref();
    if !pending.is_some_and(|pending| {
        pending.write_requests.iter().any(|request| {
            request.context_id == context_id
                && !request.refusal_superseded
                && request.has_terminal_refusal()
        })
    }) && !policy
        .foreground_read_refusals
        .iter()
        .any(|request| request.context_id == context_id)
    {
        return Ok(Vec::new());
    }
    let selected = store.select_event_search_work(context_id, work_policy)?;
    if !store.event_search_selected_work_complete(
        context_id,
        &super::search_work_spec(profile),
        &selected,
    )? {
        return Ok(Vec::new());
    }
    let mut observed = completed_foreground_read_refusals(
        store,
        context_id,
        &super::search_work_spec(profile),
        &policy.foreground_read_refusals,
    )?;
    if let Some(pending) = pending {
        observed.extend(completed_saved_memory_refusals(
            store,
            context_id,
            &super::search_work_spec(profile),
            pending,
        )?);
    }
    if !observed.is_empty() {
        reconcile_background_selections(root, store)?;
    }
    Ok(observed)
}

/// Assess exact saved frontiers without acknowledging or retiring any queue identity.
fn completed_saved_memory_refusals(
    store: &context_graph::Store,
    context_id: ContextId,
    spec: &context_graph::EventSearchWorkSpec,
    pending: &PendingBackgroundRequest,
) -> Result<Vec<String>, CgError> {
    let mut observed = Vec::new();
    for request in pending.write_requests.iter().filter(|request| {
        request.context_id == context_id
            && request.has_terminal_refusal()
            && !request.refusal_superseded
    }) {
        let accepted = if request.selection_retired {
            store.event_search_refusal_work(context_id, request.selected_work)?
        } else {
            store.event_search_selected_work(context_id, request.selected_work)?
        };
        if store.event_search_selected_work_complete(context_id, spec, &accepted)? {
            observed.push(request.token.clone());
        }
    }
    Ok(observed)
}

pub(crate) fn acknowledge_completed_memory_refusal(
    root: &Path,
    _context_id: ContextId,
    observed: &[String],
    read_memory_limit_bytes: NonZeroUsize,
) -> Result<(), CgError> {
    if observed.is_empty() {
        return Ok(());
    }
    acknowledge_foreground_read_refusals(root, observed).map_err(CgError::Engine)?;
    mark_completed_requests(
        root,
        None,
        &BTreeSet::new(),
        false,
        read_memory_limit_bytes,
        Some(observed),
        None,
    )?;
    let guard = acquire_event_search_worker_policy_update(root)
        .map_err(|error| CgError::Engine(error.to_string()))?;
    let Some(contents) = guard
        .read()
        .map_err(|error| CgError::Engine(error.to_string()))?
    else {
        return Ok(());
    };
    let mut policy = decode_worker_policy(&contents).map_err(CgError::Engine)?;
    if let Some(request) = &mut policy.pending_background_request {
        request.write_requests.retain(|request| {
            !request.has_terminal_refusal()
                || !request.selection_retired
                || !observed.contains(&request.token)
        });
        request.rebuild_memory_refusals().map_err(CgError::Engine)?;
        if request.write_requests.is_empty() {
            policy.pending_background_request = None;
        }
        guard
            .write(&serde_json::to_string(&policy).expect("worker policy serializes"))
            .map_err(|error| CgError::Engine(error.to_string()))?;
    }
    drop(guard);
    let config = store_config(root, None, read_memory_limit_bytes);
    let store = context_graph::Store::open(config)?;
    reconcile_background_selections(root, &store)?;
    Ok(())
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
pub(crate) fn pause_before_import_acknowledgement() {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = Path::new(&directory);
    if !directory
        .join("pause-before-import-acknowledgement")
        .is_file()
    {
        return;
    }
    let _ = std::fs::write(
        directory.join("import-assessment-complete"),
        b"writer released; refusal identities observed",
    );
    wait_for_test_release(|| directory.join("release-import-acknowledgement").exists());
}

fn record_background_memory_refusal(
    root: &Path,
    context_id: ContextId,
    error: &CgError,
    admitted: &[String],
) -> Result<(), String> {
    let CgError::WriteDidNotFitTheMemoryLimit {
        requested_bytes,
        limit_bytes,
    } = error
    else {
        return Ok(());
    };
    let guard =
        acquire_event_search_worker_policy_update(root).map_err(|error| error.to_string())?;
    let Some(contents) = guard.read().map_err(|error| error.to_string())? else {
        return Ok(());
    };
    let mut policy = decode_worker_policy(&contents)?;
    let ordinal = next_refusal_ordinal(&policy)?;
    if let Some(request) = &mut policy.pending_background_request
        && request
            .write_requests
            .iter()
            .any(|pending| admitted.contains(&pending.token) && !pending.refusal_superseded)
    {
        for pending in &mut request.write_requests {
            if pending.context_id == context_id
                && admitted.contains(&pending.token)
                && !pending.refusal_superseded
            {
                pending.refused_memory = true;
                pending.completed = true;
                pending.memory_refusal = PendingMemoryRefusal::Recorded {
                    ordinal,
                    refusal: BackgroundMemoryRefusal {
                        requested_bytes: *requested_bytes,
                        limit_bytes: *limit_bytes,
                    },
                };
            }
        }
        request.rebuild_memory_refusals()?;
        guard
            .write(&serde_json::to_string(&policy).expect("worker policy serializes"))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Completion is recorded while the long worker turn is still held. A later capture cannot make
/// an earlier finished request's ceiling govern new work, even if this worker is interrupted.
fn mark_completed_requests(
    root: &Path,
    owned: Option<&PendingBackgroundRequest>,
    completed: &BTreeSet<String>,
    served: bool,
    read_memory_limit_bytes: NonZeroUsize,
    retirement_tokens: Option<&[String]>,
    desired_spec: Option<&context_graph::EventSearchWorkSpec>,
) -> Result<(), CgError> {
    let guard = acquire_event_search_worker_policy_update(root)
        .map_err(|error| CgError::Engine(error.to_string()))?;
    let mut has_completed = false;
    let mut retiring = BTreeSet::new();
    if let Some(contents) = guard
        .read()
        .map_err(|error| CgError::Engine(error.to_string()))?
    {
        let mut policy = decode_worker_policy(&contents).map_err(CgError::Engine)?;
        if let Some(pending) = &mut policy.pending_background_request {
            for request in &mut pending.write_requests {
                if request.read_timeout_no_progress >= WORKER_RETRIES
                    && (request.refusal_superseded
                        || retirement_tokens.is_some_and(|tokens| tokens.contains(&request.token)))
                {
                    request.completed = true;
                    request.refusal_superseded = true;
                }
                if completed.contains(&request.token)
                    && owned.is_some_and(|owned| {
                        owned
                            .write_requests
                            .iter()
                            .any(|old| old.token == request.token)
                    })
                {
                    request.completed = true;
                }
                has_completed |= request.completed;
                if request.completed
                    && !request.selection_retired
                    && retirement_tokens.is_none_or(|tokens| tokens.contains(&request.token))
                {
                    retiring.insert(request.selected_work);
                }
            }
        }
        guard
            .write(&serde_json::to_string(&policy).expect("worker policy serializes"))
            .map_err(|error| CgError::Engine(error.to_string()))?;
    }
    drop(guard);
    if !served && !has_completed {
        return Ok(());
    }
    // Each durable selection owns its cleanup admission. Unrelated low ceilings may fail,
    // but cannot retain affordable completed selections. Shared identities use the minimum
    // of all saved ceilings and remain live while any accepted owner is unfinished.
    let mut first_error = None;
    for selection in retiring {
        let result = retire_completed_selection(
            root,
            selection,
            read_memory_limit_bytes,
            owned,
            retirement_tokens,
            desired_spec,
        );
        if let Err(error) = result {
            first_error.get_or_insert(error);
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    Ok(())
}

fn retire_completed_selection(
    root: &Path,
    selection: uuid::Uuid,
    read_memory_limit_bytes: NonZeroUsize,
    owned: Option<&PendingBackgroundRequest>,
    retirement_tokens: Option<&[String]>,
    desired_spec: Option<&context_graph::EventSearchWorkSpec>,
) -> Result<(), CgError> {
    let Some(pending) = pending_background_request_checked(root).map_err(CgError::Engine)? else {
        return Ok(());
    };
    let owners = pending
        .write_requests
        .iter()
        .filter(|request| request.selected_work == selection && !request.selection_retired)
        .collect::<Vec<_>>();
    if owners.is_empty() || owners.iter().any(|request| !request.completed) {
        return Ok(());
    }
    let cleanup_limit = owners
        .iter()
        .map(|request| request.memory_limit_bytes)
        .min()
        .unwrap();
    let mut config = store_config(root, None, read_memory_limit_bytes);
    config.write_memory_limit_bytes = cleanup_limit;
    config.read_active_time_ms = owners
        .iter()
        .map(|request| request.read_active_time_ms)
        .min()
        .unwrap();
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
        let _ = std::fs::write(
            Path::new(&directory).join("cleanup-read-active-time-ms"),
            config.read_active_time_ms.get().to_string(),
        );
        let _ = std::fs::write(
            Path::new(&directory).join("cleanup-write-memory-limit"),
            cleanup_limit.get().to_string(),
        );
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(Path::new(&directory).join("cleanup-write-memory-limits"))
        {
            use std::io::Write;
            let _ = writeln!(file, "{}", cleanup_limit.get());
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(Path::new(&directory).join("cleanup-selection-limits"))
        {
            use std::io::Write;
            let _ = writeln!(file, "{selection} {}", cleanup_limit.get());
        }
    }
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    pause_before_selection_cleanup_for_test();
    let store = context_graph::Store::open(config)?;
    let observed_foreground = worker_policy_checked(root)
        .map_err(CgError::Engine)?
        .map(|policy| policy.foreground_read_refusals)
        .unwrap_or_default();
    let mut completed_refusals = Vec::new();
    if let Some(spec) = desired_spec {
        let contexts = owners
            .iter()
            .filter(|request| {
                !request.has_terminal_refusal()
                    && owned.is_some_and(|owned| {
                        owned
                            .write_requests
                            .iter()
                            .any(|old| old.token == request.token)
                    })
            })
            .map(|request| request.context_id)
            .collect::<BTreeSet<_>>();
        for context_id in contexts {
            completed_refusals.extend(completed_foreground_read_refusals(
                &store,
                context_id,
                spec,
                &observed_foreground,
            )?);
            completed_refusals.extend(completed_saved_memory_refusals(
                &store, context_id, spec, &pending,
            )?);
        }
    }
    let guard = acquire_event_search_worker_policy_update(root)
        .map_err(|error| CgError::Engine(error.to_string()))?;
    let Some(contents) = guard
        .read()
        .map_err(|error| CgError::Engine(error.to_string()))?
    else {
        return Ok(());
    };
    let mut policy = decode_worker_policy(&contents).map_err(CgError::Engine)?;
    policy
        .foreground_read_refusals
        .retain(|refusal| !completed_refusals.contains(&refusal.token));
    let Some(pending) = &mut policy.pending_background_request else {
        return Ok(());
    };
    if pending.write_requests.iter().any(|request| {
        request.selected_work == selection
            && !request.selection_retired
            && (!request.completed || request.memory_limit_bytes < cleanup_limit)
    }) {
        return Ok(());
    }
    for request in &mut pending.write_requests {
        if request.has_terminal_refusal() && completed_refusals.contains(&request.token) {
            request.refusal_superseded = true;
        }
    }
    pending.rebuild_memory_refusals().map_err(CgError::Engine)?;
    let retained = pending
        .write_requests
        .iter()
        .filter(|request| {
            request.selected_work != selection && (!request.completed || !request.selection_retired)
        })
        .map(|request| request.selected_work)
        .collect::<Vec<_>>();
    let refusal_context = pending
        .write_requests
        .iter()
        .find(|request| {
            request.selected_work == selection
                && request.has_terminal_refusal()
                && !request.refusal_superseded
        })
        .map(|request| request.context_id);
    guard
        .write(&serde_json::to_string(&policy).expect("worker policy serializes"))
        .map_err(|error| CgError::Engine(error.to_string()))?;
    drop(guard);
    if let Some(context_id) = refusal_context {
        store.preserve_event_search_refusal_work(context_id, selection)?;
    }
    // Durable completion evidence precedes runnable retirement and queue acknowledgement.
    store.reconcile_event_search_selected_work(&retained)?;
    let guard = acquire_event_search_worker_policy_update(root)
        .map_err(|error| CgError::Engine(error.to_string()))?;
    let Some(contents) = guard
        .read()
        .map_err(|error| CgError::Engine(error.to_string()))?
    else {
        return Ok(());
    };
    let mut policy = decode_worker_policy(&contents).map_err(CgError::Engine)?;
    let Some(pending) = &mut policy.pending_background_request else {
        return Ok(());
    };
    for request in &mut pending.write_requests {
        if request.selected_work == selection {
            request.selection_retired = true;
        }
    }
    // Retirement preserves the running worker's exact non-refusal wake identities until its
    // post-unlock acknowledgement. An import acknowledges only the refusal tokens it observed,
    // so its cleanup cannot remove another worker's completed wake. Superseded refusals still
    // leave the worker queue as soon as their selections retire.
    pending.write_requests.retain(|request| {
        !request.completed
            || !request.selection_retired
            || (request.has_terminal_refusal() && !request.refusal_superseded)
            || (!request.has_terminal_refusal()
                && owned.is_some_and(|owned| {
                    owned
                        .write_requests
                        .iter()
                        .any(|old| old.token == request.token)
                }))
            || retirement_tokens.is_some_and(|tokens| !tokens.contains(&request.token))
    });
    pending.rebuild_memory_refusals().map_err(CgError::Engine)?;
    if pending.write_requests.is_empty() {
        policy.pending_background_request = None;
    }
    let retained_refusals = policy
        .pending_background_request
        .iter()
        .flat_map(|pending| &pending.write_requests)
        .filter(|request| request.has_terminal_refusal())
        .map(|request| request.selected_work)
        .collect::<Vec<_>>();
    guard
        .write(&serde_json::to_string(&policy).expect("worker policy serializes"))
        .map_err(|error| CgError::Engine(error.to_string()))?;
    drop(guard);
    // The successful retry may have acknowledged an already retired refusal selection.
    // Reclaim its completion frontier only after the durable queue update, while this
    // selection's saved cleanup admission and writable handle still own the operation.
    store.reconcile_event_search_refusal_work(&retained_refusals)?;
    Ok(())
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn pause_before_selection_cleanup_for_test() {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = Path::new(&directory);
    if !directory.join("pause-before-selection-cleanup").is_file() {
        return;
    }
    let _ = std::fs::write(
        directory.join("selection-cleanup-entered"),
        b"writer released",
    );
    wait_for_test_release(|| directory.join("release-selection-cleanup").exists());
}

/// Acknowledge finished requests by identity, leaving later appends and unfinished leased work.
/// The wake token remains until after the long worker turn releases, preserving the handoff.
fn clear_background_request_if_current(
    root: &Path,
    owned: &PendingBackgroundRequest,
) -> Result<bool, String> {
    let guard =
        acquire_event_search_worker_policy_update(root).map_err(|error| error.to_string())?;
    let Some(contents) = guard.read().map_err(|error| error.to_string())? else {
        return Ok(false);
    };
    let mut policy = decode_worker_policy(&contents)?;
    let mut unfinished = false;
    if let Some(pending) = &mut policy.pending_background_request {
        if owned.write_requests.is_empty() {
            if pending.token == owned.token {
                policy.pending_background_request = None;
            }
        } else {
            pending.write_requests.retain(|request| {
                let ours = owned
                    .write_requests
                    .iter()
                    .any(|old| old.token == request.token);
                unfinished |= ours && !request.completed;
                !ours
                    || !request.completed
                    || request.has_terminal_refusal()
                    || !request.selection_retired
            });
            pending.rebuild_memory_refusals()?;
            if pending.write_requests.is_empty() {
                policy.pending_background_request = None;
            } else if pending
                .write_requests
                .iter()
                .all(|request| request.completed)
            {
                unfinished = true; // Terminal refusal remains inspectable without another turn.
            }
        }
    }
    guard
        .write(&serde_json::to_string(&policy).expect("worker policy serializes"))
        .map_err(|error| error.to_string())?;
    Ok(unfinished)
}

/// Test-only rendezvous for the live-worker profile handoff proof. It runs after a completed pass
/// has dropped every Store access, while the worker lock is still held, so the next import can
/// prove the handoff without racing a writable Store open.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn pause_after_catch_up_before_unlock() {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    // A finite numbered sequence lets fixtures observe adoption and successive late captures.
    // Without this file, retain the original entered/release one-shot protocol.
    static PASS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let pass = PASS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let numbered = std::fs::read_to_string(directory.join("numbered-passes"))
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok());
    if numbered.is_some_and(|limit| pass > limit) {
        return;
    }
    let entered = directory.join(if numbered.is_some() {
        format!("entered.{pass}")
    } else {
        "entered".into()
    });
    let release = directory.join("release");
    let numbered_release = directory.join(format!("release.{pass}"));
    if std::fs::write(&entered, b"store closed; worker lock held\n").is_err() {
        return;
    }
    wait_for_test_release(|| release.exists() || (numbered.is_some() && numbered_release.exists()));
}

/// Test-only rendezvous immediately before the worker asks the store for work. It lets the CLI
/// journey install a durable orphan lease after the worker has loaded its profile and taken its
/// turn, without depending on how long the embedder takes to initialize.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn pause_before_catch_up() {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    if directory.join("pause-after-work-assessment").is_file() {
        return;
    }
    let entered = directory.join("pre-catch-up-entered");
    let release = directory.join("pre-catch-up-release");
    if std::fs::write(&entered, b"worker lock held; work not yet assessed\n").is_err() {
        return;
    }
    wait_for_test_release(|| release.exists());
}

/// Test-only rendezvous after the worker has read its queue but before it opens the store for the
/// projection. This lets the CLI proof introduce a real writer without making the queue read itself
/// the contended operation.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn pause_after_work_assessment() {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    if !directory.join("pause-after-work-assessment").is_file() {
        return;
    }
    if std::fs::write(
        directory.join("work-assessment-complete"),
        b"pending work assessed; projection not opened\n",
    )
    .is_err()
    {
        return;
    }
    wait_for_test_release(|| directory.join("work-assessment-release").exists());
}

/// Test-only rendezvous after foreground handoff owns or delivers every queued request, while the
/// command's SIGINT listener must still cover its final status refresh.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn pause_after_foreground_handoff_before_refresh() {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    if !directory.join("pause-after-foreground-handoff").is_file() {
        return;
    }
    if std::fs::write(
        directory.join("foreground-handoff-complete"),
        b"queued handoff settled; final status not read\n",
    )
    .is_err()
    {
        return;
    }
    wait_for_test_release(|| directory.join("release-final-status").exists());
}

/// Use the core's existing thread-local allowance override only in a private-feature binary.
/// Production builds neither compile nor inspect this test marker.
#[cfg(feature = "event-search-test-support")]
pub(crate) fn apply_store_open_retry_allowance_for_test() {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER") else {
        return;
    };
    if std::path::Path::new(&directory)
        .join("exhaust-store-open-retry")
        .is_file()
    {
        context_graph::test_support::set_store_open_retry_allowance_for_test(Some(Duration::ZERO));
    }
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn wait_for_test_release(mut released: impl FnMut() -> bool) {
    // This is a test-only cross-process condition handshake. The integration harness owns the
    // bounded test timeout; no local polling clock may disguise a synchronization timeout.
    while !released() {
        std::thread::park_timeout(Duration::from_millis(10));
    }
}

/// Pause before asking a transient condition again, without letting repeated rechecks spin a CPU.
fn pause_before_recheck(backoff: &mut Duration, maximum_pause: Duration) {
    std::thread::sleep((*backoff).min(maximum_pause));
    *backoff = (*backoff * 2).min(UNOBSERVABLE_BACKOFF_CEILING);
}

/// Retry one reader-refused operation within one allowance, preserving every other result.
fn retry_while_readers<T>(
    root: &Path,
    allowance: Duration,
    mut operation: impl FnMut() -> context_graph::Result<T>,
) -> context_graph::Result<T> {
    let started = Instant::now();
    let deadline = started + allowance;
    let mut backoff = Duration::from_millis(20);
    loop {
        match operation() {
            Err(refusal @ CgError::StoreHeldByReaders { .. }) => {
                let remaining = allowance.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return Err(refusal);
                }
                match wait_for_reader_release_within(&db_path(root), deadline) {
                    ReaderRelease::Released => {}
                    ReaderRelease::Stopped => return Err(refusal),
                    ReaderRelease::Unobservable(_) => {
                        let remaining = allowance.saturating_sub(started.elapsed());
                        if remaining.is_zero() {
                            return Err(refusal);
                        }
                        pause_before_recheck(&mut backoff, remaining);
                    }
                }
            }
            result => return result,
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
    profile: &EventSearchProfile,
    embedding_workers: NonZeroUsize,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> EventSearchRunRequest {
    let request = EventSearchRunRequest {
        store: store_config(root, None, read_memory_limit_bytes),
        contexts: ContextSelector::one(context_id),
        embedder,
        profile: profile.clone(),
        embedding_workers,
        max_events_per_batch: NonZeroU32::new(256).expect("non-zero"),
        max_passages_per_batch: NonZeroU32::new(32).expect("non-zero"),
        max_bytes_per_batch: NonZeroUsize::new(4 * 1024 * 1024).expect("non-zero"),
        lease_duration_ms: NonZeroU64::new(120_000).expect("non-zero"),
    };
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
        let _ = std::fs::write(
            std::path::PathBuf::from(directory).join("worker-store-write-memory-limit-bytes"),
            request.store.write_memory_limit_bytes.get().to_string(),
        );
    }
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    observe_embedding_workers_request_for_test(request.embedding_workers);
    request
}

/// Route every CLI projection run through the core's per-batch worker-cap seam, resolving the
/// latest valid policy before each claim.
fn run_observed_with_embedding_worker_policy(
    root: &Path,
    request: EventSearchRunRequest,
    resolver: &mut EmbeddingWorkerPolicyResolver,
    work_policy: EventSearchWorkPolicy,
    selected_work: EventSearchSelectedWork,
    queue_backed: bool,
) -> (EventSearchRunOutcome, Vec<String>) {
    let context_id = request.contexts.context_ids()[0];
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    let read_active_time_ms = request.store.read_active_time_ms;
    let admitted = RefCell::new(Vec::new());
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    {
        observe_actual_write_budget_for_test();
        if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
            let directory = PathBuf::from(directory);
            if std::fs::read_to_string(directory.join("supersede-selected-session")).is_ok_and(
                |session| {
                    selected_work
                        .streams
                        .iter()
                        .any(|stream| stream.session_ref == session.trim())
                },
            ) {
                // Use the core's committed-attempt seam to perform an actual ownership takeover.
                // Its readback produces the same Stale disposition as a competing production run.
                let competing_request = request.clone();
                context_graph::test_support::after_next_event_search_attempt_commit_before_readback_for_test(move |_| {
                    EventSearchProjectionRunner::record_refusal(
                        competing_request,
                        "competing request owns the committed attempt marker",
                    ).expect("competing production attempt takes the committed marker");
                    use std::io::Write;
                    let mut attempts = std::fs::OpenOptions::new().create(true).append(true)
                        .open(directory.join("superseded-selected-attempts")).unwrap();
                    writeln!(attempts, "{context_id}").unwrap();
                });
            }
        }
        if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
            let marker = Path::new(&directory).join("refuse-selected-active-time-count");
            if let Some(count) = std::fs::read_to_string(&marker)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|count| *count > 0)
            {
                let _ = std::fs::write(&marker, (count - 1).to_string());
                context_graph::test_support::fail_next_event_search_membership_capture_with_error_for_test(
                    CgError::ReadActiveTimeExceeded { limit_ms: request.store.read_active_time_ms.get() },
                );
            }
        }
        if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
            let marker = Path::new(&directory).join("refuse-selected-coordination-count");
            if let Some(count) = std::fs::read_to_string(&marker)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|count| *count > 0)
            {
                let _ = std::fs::write(&marker, (count - 1).to_string());
                context_graph::test_support::fail_next_event_search_membership_capture_with_error_for_test(
                    CgError::ReaderCoordinationUnavailable {
                        operation: "prepare selected search work".into(),
                        path: db_path(root).display().to_string(),
                        reason: "deterministic selected-work coordination refusal".into(),
                    },
                );
            }
        }
        if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
            let marker = Path::new(&directory).join("refuse-next-selected-memory");
            if let Some(limit_bytes) = std::fs::read_to_string(&marker)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
            {
                let _ = std::fs::remove_file(marker);
                context_graph::test_support::fail_next_event_search_membership_capture_with_error_for_test(
                    CgError::WriteDidNotFitTheMemoryLimit {
                        requested_bytes: limit_bytes.saturating_add(1), limit_bytes,
                    },
                );
            }
        }
        if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
            && std::fs::remove_file(Path::new(&directory).join("refuse-next-selected-work")).is_ok()
        {
            context_graph::test_support::fail_next_event_search_membership_capture_for_test(
                "selected work received its explicit test refusal",
            );
        }
    }
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    let failure_record_barrier = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
        .map(std::path::PathBuf::from)
        .filter(|directory| {
            directory
                .join("hold-writer-during-failure-record")
                .is_file()
        });
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = failure_record_barrier.clone() {
        context_graph::test_support::set_store_open_retry_allowance_for_test(Some(Duration::ZERO));
        context_graph::test_support::before_next_event_search_failure_record_for_test(move || {
            let _ = std::fs::write(
                directory.join("failure-record-entered"),
                b"runner store closed",
            );
            wait_for_test_release(|| directory.join("failure-record-continue").is_file());
        });
    }
    let observed =
        EventSearchProjectionRunner::catch_up_observed_with_selected_work_and_batch_limits(
            request,
            work_policy,
            selected_work,
            || {
                let embedding_workers = resolver.resolve_next_claim();
                #[cfg(any(
                    context_graph_internal_test_support,
                    feature = "event-search-test-support"
                ))]
                {
                    observe_embedding_workers_batch_for_test(root, embedding_workers);
                }
                #[cfg(not(any(
                    context_graph_internal_test_support,
                    feature = "event-search-test-support"
                )))]
                let _ = root;
                embedding_workers
            },
            |batch| {
                #[cfg(any(
                    context_graph_internal_test_support,
                    feature = "event-search-test-support"
                ))]
                if batch.is_some()
                    && let Some(directory) =
                        std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
                {
                    let directory = Path::new(&directory);
                    if directory.join("actual-write-memory-batches").is_file()
                        && std::fs::remove_file(
                            directory.join("refuse-active-time-after-committed-batch"),
                        )
                        .is_ok()
                    {
                        return Err(CgError::ReadActiveTimeExceeded {
                            limit_ms: read_active_time_ms.get(),
                        });
                    }
                    // Reaching the next batch admission proves the preceding batch committed.
                    if directory.join("actual-write-memory-batches").is_file()
                        && std::fs::remove_file(
                            directory.join("refuse-coordination-after-committed-batch"),
                        )
                        .is_ok()
                    {
                        return Err(CgError::ReaderCoordinationUnavailable {
                            operation: "admit selected search batch".into(),
                            path: db_path(root).display().to_string(),
                            reason: "deterministic selected-work coordination refusal after committed batch".into(),
                        });
                    }
                }
                let pending = pending_background_request_checked(root).map_err(CgError::Engine)?;
                let mut eligible = Vec::new();
                let mut through_sequence = None;
                if let Some(pending) = &pending {
                    for request in &pending.write_requests {
                        if request.context_id != context_id || request.completed {
                            continue;
                        }
                        let shared = match &batch {
                            None => true,
                            Some(batch) => {
                                match (batch.selected_frontier)(request.selected_work)? {
                                    Some(frontier) if frontier > batch.committed_sequence => {
                                        through_sequence = Some(
                                            through_sequence.map_or(frontier, |previous: u64| {
                                                previous.min(frontier)
                                            }),
                                        );
                                        true
                                    }
                                    _ => false,
                                }
                            }
                        };
                        if shared {
                            eligible.push(request);
                        }
                    }
                }
                // Unfinished owners select the physical batch. Until its durable retirement,
                // every owner of that exact selection still supplies its saved ceiling, even
                // when an earlier owner already completed or recorded a refusal.
                let selected_identities = eligible
                    .iter()
                    .map(|request| request.selected_work)
                    .collect::<BTreeSet<_>>();
                let owners = pending
                    .iter()
                    .flat_map(|pending| &pending.write_requests)
                    .filter(|request| {
                        request.context_id == context_id
                            && !request.selection_retired
                            && selected_identities.contains(&request.selected_work)
                    })
                    .collect::<Vec<_>>();
                let limit = owners
                    .iter()
                    .map(|request| request.memory_limit_bytes)
                    .min();
                let limit = match limit {
                    Some(limit) => limit,
                    None => {
                        if queue_backed && batch.is_some() {
                            admitted.borrow_mut().clear();
                            return Err(CgError::Engine("event-search batch has no pending request selecting its source frontier".into()));
                        }
                        if let Some(refusal) = pending.as_ref().and_then(|pending| {
                            pending.write_memory_refusals.get(&context_id.to_string())
                        }) {
                            admitted.borrow_mut().clear();
                            return Err(CgError::WriteDidNotFitTheMemoryLimit {
                                requested_bytes: refusal.requested_bytes,
                                limit_bytes: refusal.limit_bytes,
                            });
                        }
                        crate::write_config::in_force().bytes
                    }
                };
                let governed_identities = owners
                    .iter()
                    .filter(|request| request.memory_limit_bytes == limit)
                    .map(|request| request.selected_work)
                    .collect::<BTreeSet<_>>();
                // A refusal at the shared minimum terminates every unfinished owner of that
                // physical selection. Distinct selections retain their own service obligation.
                *admitted.borrow_mut() = owners
                    .iter()
                    .filter(|request| governed_identities.contains(&request.selected_work))
                    .map(|request| request.token.clone())
                    .collect();
                #[cfg(any(
                    context_graph_internal_test_support,
                    feature = "event-search-test-support"
                ))]
                observe_write_memory_limit_for_test(context_id, limit);
                Ok(EventSearchBatchWriteLimit {
                    memory_limit_bytes: limit,
                    through_sequence,
                })
            },
        );
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Err(error) = &observed.result
        && let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
    {
        let _ = std::fs::write(
            Path::new(&directory).join("projection-error"),
            format!("{error:?}"),
        );
    }
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = failure_record_barrier {
        context_graph::test_support::set_store_open_retry_allowance_for_test(None);
        let _ = std::fs::write(
            directory.join("failure-record-returned"),
            b"core disposition returned",
        );
        wait_for_test_release(|| directory.join("failure-writer-released").is_file());
    }
    (observed, admitted.into_inner())
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observe_actual_write_budget_for_test() {
    let directory = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER").map(PathBuf::from);
    context_graph::test_support::observe_event_search_batch_admission_for_test(directory.map(|directory| {
        Box::new(move |session: &str, source: &str, committed, through, limit| {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(directory.join("actual-write-memory-batches")) {
                let _ = writeln!(file, "{}", serde_json::json!({"session": session, "source": source, "committed": committed, "through": through, "limit": limit}));
            }
        }) as Box<dyn Fn(&str, &str, u64, Option<u64>, usize)>
    }));
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observe_write_memory_limit_for_test(context_id: ContextId, limit: NonZeroUsize) {
    use std::io::Write;
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = Path::new(&directory);
    let _ = std::fs::write(
        directory.join("worker-store-write-memory-limit-bytes"),
        limit.get().to_string(),
    );
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("write-memory-batches"))
    {
        let _ = writeln!(file, "{context_id} {}", limit.get());
    }
}

/// Foreground wait owns its invocation-resolved cap for every claim. It may publish that cap for a
/// detached worker, but a later background publication cannot alter this foreground projection.
fn run_observed_with_fixed_embedding_workers(
    root: &Path,
    request: EventSearchRunRequest,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
) -> EventSearchRunOutcome {
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    let injected_refusal = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
        .map(PathBuf::from)
        .and_then(|directory| {
            let marker = directory.join("refuse-foreground-active-time-count");
            let count = std::fs::read_to_string(&marker)
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|count| *count > 0)?;
            std::fs::write(marker, (count - 1).to_string()).ok()?;
            context_graph::test_support::fail_next_event_search_membership_capture_with_error_for_test(
                CgError::ReadActiveTimeExceeded { limit_ms: request.store.read_active_time_ms.get() },
            );
            Some((directory, count))
        });
    let observed = EventSearchProjectionRunner::catch_up_observed_with_policy_and_embedding_workers(
        request,
        work_policy,
        || {
            #[cfg(any(
                context_graph_internal_test_support,
                feature = "event-search-test-support"
            ))]
            observe_embedding_workers_batch_for_test(root, embedding_workers);
            #[cfg(not(any(
                context_graph_internal_test_support,
                feature = "event-search-test-support"
            )))]
            let _ = root;
            embedding_workers
        },
    );
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some((directory, count)) = injected_refusal
        && matches!(observed.result, Err(CgError::ReadActiveTimeExceeded { .. }))
        && directory
            .join("pause-after-foreground-read-timeout")
            .is_file()
    {
        let _ = std::fs::write(
            directory.join(format!("foreground-read-timeout-{count}")),
            b"runner released its store",
        );
        wait_for_test_release(|| {
            directory
                .join(format!("release-foreground-read-timeout-{count}"))
                .is_file()
        });
    }
    observed
}

/// Internal observation of the exact request built by the production worker path. The existing
/// handoff-barrier directory keeps this out of the public surface and out of non-test builds.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observe_embedding_workers_request_for_test(embedding_workers: NonZeroUsize) {
    use std::io::Write;
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let path = Path::new(&directory).join("embedding-worker-requests");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{}", embedding_workers.get());
    }
}

/// The independent delivery proof opts into this observation with a marker in the existing
/// handoff-barrier directory. It observes the cap carried into each core callback and pauses the
/// first callback at the already-established rendezvous, without adding a production control path.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observe_embedding_workers_batch_for_test(root: &Path, embedding_workers: NonZeroUsize) {
    use std::io::Write;

    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = Path::new(&directory);
    if !directory.join("observe-embedding-worker-batches").is_file() {
        return;
    }
    let path = directory.join("embedding-worker-batches");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{}", embedding_workers.get());
    }
    let _ = root;
    pause_after_catch_up_before_unlock();
}

/// The detached worker starts in its own process group. The test marker records that group so
/// the test can release and terminate its child after a bounded observation window.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn observe_background_worker_process_group_for_test(process_group: u32) {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = Path::new(&directory);
    if directory.join("observe-embedding-worker-batches").is_file() {
        let _ = std::fs::write(
            directory.join("embedding-worker-process-group"),
            process_group.to_string(),
        );
    }
}

/// Every requested Context assessment, including work another live projection is serving, or
/// the typed refusal when the store would not answer.
///
/// A refusal is not an answer. Direct readers holding the store are waited out here, and a question
/// that still cannot be put is reported as unknown rather than as "no work": only `cg import` and
/// `cg forget` start a worker, so a worker that mistakes a refused read for an empty queue strands
/// every pending Context until the user imports again.
fn requested_work_assessments(
    root: &Path,
    desired_profile: &EventSearchProfile,
    work_policy: EventSearchWorkPolicy,
    read_memory_limit_bytes: NonZeroUsize,
    assessed: Option<&PendingBackgroundRequest>,
    suppression: &RequestScopedSuppression,
) -> context_graph::Result<Vec<EventSearchWorkAssessment>> {
    let active_time_ms = assessed
        .iter()
        .flat_map(|pending| &pending.write_requests)
        .filter(|request| suppression.serviceable(request))
        .map(|request| request.read_active_time_ms)
        .min()
        .unwrap_or_else(crate::read_config::active_time_in_force);
    retry_while_readers(root, READER_RELEASE_ALLOWANCE, || {
        pending_once(
            root,
            desired_profile,
            work_policy,
            read_memory_limit_bytes,
            active_time_ms,
        )
    })
}

fn pending_once(
    root: &Path,
    desired_profile: &EventSearchProfile,
    work_policy: EventSearchWorkPolicy,
    read_memory_limit_bytes: NonZeroUsize,
    active_time_ms: NonZeroU64,
) -> context_graph::Result<Vec<EventSearchWorkAssessment>> {
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
        && Path::new(&directory)
            .join("refuse-aggregate-read")
            .is_file()
    {
        eprintln!("deterministic aggregate search read refusal");
        return Err(CgError::Engine(
            "deterministic aggregate search read refusal".into(),
        ));
    }
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
        let marker = Path::new(&directory).join("refuse-aggregate-active-time-count");
        let applies = std::fs::read_to_string(
            Path::new(&directory).join("refuse-aggregate-active-time-limit-ms"),
        )
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .is_none_or(|limit| limit == active_time_ms.get());
        if let Some(count) = std::fs::read_to_string(&marker)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|count| *count > 0 && applies)
        {
            if Path::new(&directory)
                .join("pause-before-aggregate-active-time-refusal")
                .is_file()
            {
                let _ = std::fs::write(
                    Path::new(&directory).join("aggregate-active-time-refusal-entered"),
                    b"observed accepted identities",
                );
                wait_for_test_release(|| {
                    Path::new(&directory)
                        .join("release-aggregate-active-time-refusal")
                        .is_file()
                });
            }
            let _ = std::fs::write(&marker, (count - 1).to_string());
            return Err(CgError::ReadActiveTimeExceeded {
                limit_ms: active_time_ms.get(),
            });
        }
    }
    let reader = context_graph::ConsumerReader::open_with_read_limits(
        db_path(root),
        context_graph::event_history_reader_declaration(),
        read_memory_limit_bytes,
        active_time_ms,
    )?;
    let contexts = reader.graph().list_contexts()?;
    let Ok(selector) = ContextSelector::new(contexts.iter().map(|context| context.id)) else {
        return Ok(Vec::new());
    };
    reader.event_search_work_with_policy(
        selector,
        &super::search_work_spec(desired_profile),
        work_policy,
    )
}

/// Give a standalone forget rebuild its own durable selection before waking the worker.
/// Sharing the queue preserves its obligation alongside imports in other Contexts.
pub(crate) fn kick(
    root: &Path,
    profile: &EventSearchProfile,
    context_id: ContextId,
    read_memory_limit_bytes: NonZeroUsize,
) -> Result<&'static str, String> {
    let embedding_workers = resolved_embedding_workers(root, None)?;
    let work_policy = resolved_search_work_policy(root, None, None)?;
    let store = context_graph::Store::open(store_config(root, None, read_memory_limit_bytes))
        .map_err(|error| error.to_string())?;
    let work_policy = pending_background_request_checked(root)?
        .map_or(work_policy, |pending| pending.work_policy);
    let selected = store
        .select_event_search_work(context_id, work_policy)
        .map_err(|error| error.to_string())?;
    let request = accept_background_work_with_choice(
        root,
        &store,
        context_id,
        embedding_workers,
        work_policy,
        &selected,
        &super::search_work_spec(profile),
        true,
    )?;
    drop(store);
    kick_accepted_background_work(root, profile, request)
}

fn reconcile_background_selections(
    root: &Path,
    store: &context_graph::Store,
) -> Result<(), CgError> {
    let requests = pending_background_request_checked(root)
        .map_err(CgError::Engine)?
        .map(|pending| pending.write_requests)
        .unwrap_or_default();
    let retained = requests
        .iter()
        .filter(|request| !request.completed || !request.selection_retired)
        .map(|request| request.selected_work)
        .collect::<Vec<_>>();
    store.reconcile_event_search_selected_work(&retained)?;
    let refused = requests
        .iter()
        .filter(|request| request.has_terminal_refusal())
        .map(|request| request.selected_work)
        .collect::<Vec<_>>();
    store.reconcile_event_search_refusal_work(&refused)
}

/// Persist selected work only while accepting its bounded queue entry. Publication failures
/// retire the selection on the original writable handle, after the queue guard has released.
pub(crate) fn accept_background_work(
    root: &Path,
    store: &context_graph::Store,
    context_id: ContextId,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
    selected_work: &EventSearchSelectedWork,
    desired_spec: &context_graph::EventSearchWorkSpec,
) -> Result<PendingBackgroundRequest, String> {
    accept_background_work_with_choice(
        root,
        store,
        context_id,
        embedding_workers,
        work_policy,
        selected_work,
        desired_spec,
        false,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "One admission transaction needs the existing acceptance inputs and choice-preservation flag"
)]
fn accept_background_work_with_choice(
    root: &Path,
    store: &context_graph::Store,
    context_id: ContextId,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
    selected_work: &EventSearchSelectedWork,
    desired_spec: &context_graph::EventSearchWorkSpec,
    preserve_choice: bool,
) -> Result<PendingBackgroundRequest, String> {
    reconcile_background_selections(root, store).map_err(|error| error.to_string())?;
    // The queue read releases its guard before any database work. The held writable
    // Store serializes acceptances and retirement, so these identities remain durable.
    let pending = pending_background_request_checked(root)?;
    let completed_refusal_tokens = pending
        .as_ref()
        .map(|pending| completed_saved_memory_refusals(store, context_id, desired_spec, pending))
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let retained_requests = pending
        .into_iter()
        .flat_map(|pending| pending.write_requests)
        .filter(|request| request.context_id == context_id && !request.completed)
        .map(|request| request.selected_work)
        .collect::<Vec<_>>();
    // Coverage already committed before acceptance cannot restore this request's retry allowance.
    // Read through the original writable handle, before publishing any selection identity.
    let covered_events = store
        .event_search_work_with_policy(ContextSelector::one(context_id), desired_spec, work_policy)
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or("accepted search assessment returned no Context")?
        .covered_events;
    let id = store
        .persist_event_search_selected_work_with_retained_requests(
            context_id,
            selected_work,
            &retained_requests,
            desired_spec,
        )
        .map_err(|error| error.to_string())?;
    match publish_embedding_worker_policy(
        root,
        context_id,
        embedding_workers,
        work_policy,
        id,
        preserve_choice,
        &completed_refusal_tokens,
        covered_events,
    ) {
        Ok(request) => Ok(request),
        Err(reason) => {
            store
                .release_event_search_selected_work(context_id, &[id])
                .map_err(|error| format!("{reason}; selected-work cleanup failed: {error}"))?;
            Err(reason)
        }
    }
}

/// Wake only after the accepting import releases its writable handle. A failed spawn retains
/// the accepted request and its durable selection for the next worker.
pub(crate) fn kick_accepted_background_work(
    root: &Path,
    profile: &EventSearchProfile,
    request: PendingBackgroundRequest,
) -> Result<&'static str, String> {
    kick_after_admission(
        root,
        profile,
        request.embedding_workers,
        request.work_policy,
        request.ignored_publication,
    )
}

fn kick_after_admission(
    root: &Path,
    profile: &EventSearchProfile,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
    ignored_publication: Option<String>,
) -> Result<&'static str, String> {
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
            let envelope = WorkerEnvelope {
                profile: profile.clone(),
                embedding_workers,
                work_policy,
                ignored_publication,
                retry_worker_lock: false,
            };
            spawn_worker(envelope)?;
            Ok("spawned")
        }
    }
}

fn spawn_worker(envelope: WorkerEnvelope) -> Result<(), String> {
    let exe = std::env::current_exe()
        .map_err(|error| format!("cannot locate the cg executable: {error}"))?;
    let child = Command::new(exe)
        .arg(WORKER_COMMAND)
        .arg(serde_json::to_string(&envelope).expect("worker envelope serializes"))
        .arg(context_graph::write_budget::WRITE_MEMORY_LIMIT_SWITCH)
        .arg(crate::write_config::in_force().bytes.get().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .map_err(|error| format!("could not start the background search worker: {error}"))?;
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    observe_background_worker_process_group_for_test(child.id());
    #[cfg(not(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    )))]
    let _ = child;
    Ok(())
}

fn acquire_worker_lock_within(root: &Path, allowance: Duration) -> Option<WorkerLock> {
    let started = Instant::now();
    loop {
        match acquire_worker_lock(&root.join(EVENT_SEARCH_WORKER_LOCK)) {
            Ok(held) => return Some(held),
            Err(WorkerLockRefusal::HeldByAnotherWorker)
                if !allowance.saturating_sub(started.elapsed()).is_zero() =>
            {
                std::thread::sleep(
                    Duration::from_millis(20).min(allowance.saturating_sub(started.elapsed())),
                );
            }
            Err(_) => return None,
        }
    }
}

/// After a foreground holder releases the long worker lock, start the ordinary detached worker if
/// a background import or forget requested service while that turn was held. This does not publish
/// a choice or manufacture a wake request.
fn handoff_after_foreground(root: &Path, profile: &EventSearchProfile) -> Result<(), String> {
    let Some(request) = pending_background_request(root) else {
        return Ok(());
    };
    kick_after_admission(
        root,
        profile,
        request.embedding_workers,
        request.work_policy,
        request.ignored_publication,
    )
    .map(|_| ())
}

/// A successful pass is not a retry. Only identical, still-required work after repeated runs is
/// suppressed per accepted selection, so a false work predicate cannot spin and completed
/// selections never consume a later accepted selection's allowance.
struct RequestScopedSuppression {
    token: Option<String>,
    failed: BTreeSet<ContextId>,
    unchanged: BTreeMap<(ContextId, Option<uuid::Uuid>), (EventSearchWorkAssessment, usize)>,
    // Independent of generation assessments and queue publication tokens: a failed attempt may
    // replace a generation, and a later import may publish a new queue token. Neither grants
    // another allowance to the same accepted selection.
    coordination: BTreeMap<(ContextId, Option<uuid::Uuid>), usize>,
    unrecorded: BTreeSet<(ContextId, Option<uuid::Uuid>)>,
    aggregate_reads: BTreeMap<(ContextId, Option<uuid::Uuid>), usize>,
}

struct CatchUpResult {
    known: bool,
    /// Typed active-time reads spend the durable allowance instead of unknown-read suppression.
    active_time_refused: bool,
    unrecordable_failure: bool,
    completed: BTreeSet<String>,
    memory_refused: bool,
    served: bool,
}

impl RequestScopedSuppression {
    fn serviceable(&self, request: &PendingWriteRequest) -> bool {
        !request.completed
            && (request.read_timeout_no_progress < WORKER_RETRIES || request.refusal_superseded)
            && !self
                .unrecorded
                .contains(&(request.context_id, Some(request.selected_work)))
    }

    // Attribute an unanswerable aggregate to the accepted identities that were actually
    // observed before asking. A later queue publication cannot reset an older allowance,
    // and an arrival after that assessment keeps its own allowance.
    fn aggregate_read_failed(&mut self, pending: &PendingBackgroundRequest) -> bool {
        let mut exhausted = false;
        let identities = pending
            .write_requests
            .iter()
            .filter(|request| self.serviceable(request))
            .map(|request| (request.context_id, Some(request.selected_work)))
            .collect::<BTreeSet<_>>();
        for identity in identities {
            let attempts = self.aggregate_reads.entry(identity).or_default();
            *attempts += 1;
            if *attempts >= WORKER_RETRIES {
                self.unrecorded.insert(identity);
                exhausted = true;
            }
        }
        exhausted
    }

    fn selected_read_failed(
        &mut self,
        root: &Path,
        request: &PendingWriteRequest,
        error: &CgError,
        covered_events: u64,
    ) -> bool {
        eprintln!("background selected search read refused: {error}");
        let identity = (request.context_id, Some(request.selected_work));
        if let CgError::ReadActiveTimeExceeded { limit_ms } = error {
            let attempts = record_read_timeout(
                root,
                request.context_id,
                request.selected_work,
                covered_events,
                false,
                *limit_ms,
            )
            .unwrap_or(WORKER_RETRIES);
            if attempts < WORKER_RETRIES {
                #[cfg(any(
                    context_graph_internal_test_support,
                    feature = "event-search-test-support"
                ))]
                pause_after_read_timeout_for_test(root);
                let mut backoff = Duration::from_millis(20).saturating_mul(1 << attempts.min(4));
                pause_before_recheck(&mut backoff, UNOBSERVABLE_BACKOFF_CEILING);
                return false;
            }
        }
        if matches!(error, CgError::ReaderCoordinationUnavailable { .. }) {
            let attempts = self.coordination.entry(identity).or_default();
            *attempts += 1;
            if *attempts < WORKER_RETRIES {
                return false;
            }
        }
        self.unrecorded.insert(identity);
        true
    }

    fn observe(&mut self, request: Option<&PendingBackgroundRequest>) {
        self.coordination
            .retain(|(context_id, selection), _| match selection {
                Some(selection) => request.is_some_and(|pending| {
                    pending.write_requests.iter().any(|entry| {
                        entry.context_id == *context_id
                            && entry.selected_work == *selection
                            && !entry.completed
                    })
                }),
                None => request.is_none(),
            });
        let token = request.map(|request| request.token.clone());
        let publication_changed = self.token != token;
        self.unchanged
            .retain(|(context_id, selection), _| match selection {
                Some(selection) => request.is_some_and(|pending| {
                    pending.write_requests.iter().any(|entry| {
                        entry.context_id == *context_id
                            && entry.selected_work == *selection
                            && !entry.completed
                    })
                }),
                None => !publication_changed,
            });
        if self.token != token {
            self.token = token;
            self.failed.clear();
        }
    }

    fn clear(&mut self) {
        self.failed.clear();
        self.unchanged
            .retain(|(_, selection), _| selection.is_some());
    }
}

fn catch_up(
    root: &Path,
    embedder: &EmbedderRegistration,
    profile: &EventSearchProfile,
    suppression: &mut RequestScopedSuppression,
    embedding_worker_policy: &mut EmbeddingWorkerPolicyResolver,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> CatchUpResult {
    let work_policy = embedding_worker_policy.current_work_policy();
    let pending = match pending_background_request_checked(root) {
        Ok(pending) => pending,
        Err(_) => {
            return CatchUpResult {
                known: false,
                active_time_refused: false,
                unrecordable_failure: true,
                completed: BTreeSet::new(),
                memory_refused: false,
                served: false,
            };
        }
    };
    let mut known = match requested_work_assessments(
        root,
        profile,
        work_policy,
        read_memory_limit_bytes,
        pending.as_ref(),
        suppression,
    ) {
        Ok(known) => known,
        Err(error) => {
            let terminal =
                record_aggregate_read_timeout(root, suppression, &error, pending.as_ref());
            return CatchUpResult {
                known: false,
                active_time_refused: matches!(error, CgError::ReadActiveTimeExceeded { .. })
                    && pending
                        .as_ref()
                        .is_some_and(|pending| !pending.write_requests.is_empty()),
                unrecordable_failure: terminal,
                completed: BTreeSet::new(),
                memory_refused: false,
                served: false,
            };
        }
    };
    let mut selected_policies = BTreeMap::new();
    let mut selected_work = BTreeMap::new();
    let mut selected_identities = BTreeMap::new();
    let mut selected_complete_by_context = BTreeMap::new();
    let mut snapshots = BTreeMap::new();
    let mut unrecordable_failure = false;
    let mut unresolved_read = false;
    let mut active_time_refused = false;
    let mut other_unresolved_read = false;
    if let Some(pending) = &pending {
        for request in &pending.write_requests {
            if !suppression.serviceable(request) {
                continue;
            }
            let selection = (|| {
                #[cfg(any(
                    context_graph_internal_test_support,
                    feature = "event-search-test-support"
                ))]
                if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
                    && std::fs::read_to_string(Path::new(&directory).join("refuse-selected-read"))
                        .is_ok_and(|identity| identity.trim() == request.selected_work.to_string())
                {
                    return Err(CgError::Engine(
                        "deterministic selected-work read refusal".into(),
                    ));
                }
                #[cfg(any(
                    context_graph_internal_test_support,
                    feature = "event-search-test-support"
                ))]
                if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
                    let marker =
                        Path::new(&directory).join("refuse-selected-assessment-active-time-count");
                    if let Some(count) = std::fs::read_to_string(&marker)
                        .ok()
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .filter(|count| *count > 0)
                    {
                        let _ = std::fs::write(&marker, (count - 1).to_string());
                        return Err(CgError::ReadActiveTimeExceeded {
                            limit_ms: request.read_active_time_ms.get(),
                        });
                    }
                }
                let reader = context_graph::ConsumerReader::open_with_read_limits(
                    db_path(root),
                    context_graph::event_history_reader_declaration(),
                    read_memory_limit_bytes,
                    request.read_active_time_ms,
                )?;
                let selection =
                    reader.event_search_selected_work(request.context_id, request.selected_work)?;
                let complete = reader.event_search_selected_work_complete(
                    request.context_id,
                    &super::search_work_spec(profile),
                    &selection,
                )?;
                Ok((selection, complete))
            })();
            let (selection, complete) = match selection {
                Ok(selection) => selection,
                Err(error) => {
                    let terminal = suppression.selected_read_failed(
                        root,
                        request,
                        &error,
                        known
                            .iter()
                            .find(|work| work.context_id == request.context_id)
                            .map_or(request.read_timeout_covered_events, |work| {
                                work.covered_events
                            }),
                    );
                    unrecordable_failure |= terminal;
                    unresolved_read |= !terminal;
                    active_time_refused |= matches!(error, CgError::ReadActiveTimeExceeded { .. });
                    other_unresolved_read |=
                        !terminal && !matches!(error, CgError::ReadActiveTimeExceeded { .. });
                    continue;
                }
            };
            selected_policies
                .entry(request.context_id)
                .or_insert(request.work_policy);
            selected_work
                .entry(request.context_id)
                .or_insert_with(|| selection.clone());
            selected_identities
                .entry(request.context_id)
                .or_insert(request.selected_work);
            selected_complete_by_context
                .entry(request.context_id)
                .or_insert(complete);
            snapshots.insert(request.token.clone(), selection);
        }
    }
    if pending
        .as_ref()
        .is_some_and(|pending| !pending.write_requests.is_empty())
    {
        known.retain(|work| selected_work.contains_key(&work.context_id));
    }
    let selected_complete = |context_id, identity, read_active_time_ms| {
        let reader = context_graph::ConsumerReader::open_with_read_limits(
            db_path(root),
            context_graph::event_history_reader_declaration(),
            read_memory_limit_bytes,
            read_active_time_ms,
        )?;
        let current = reader.event_search_selected_work(context_id, identity)?;
        reader.event_search_selected_work_complete(
            context_id,
            &super::search_work_spec(profile),
            &current,
        )
    };
    for work in &mut known {
        if let Some(complete) = selected_complete_by_context.get(&work.context_id) {
            work.work_required = !complete;
        }
    }
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    pause_after_work_assessment();
    let mut completed = pending
        .iter()
        .flat_map(|pending| &pending.write_requests)
        .filter(|request| {
            !request.completed
                && request.read_timeout_no_progress >= WORKER_RETRIES
                && (request.read_timeout_failure_recorded || request.refusal_superseded)
        })
        .map(|request| request.token.clone())
        .collect::<BTreeSet<_>>();
    let mut memory_refused = false;
    let mut served = false;
    for work in known {
        let context_id = work.context_id;
        let selected_policy = selected_policies
            .get(&context_id)
            .copied()
            .unwrap_or(work_policy);
        let complete = |completed: &mut BTreeSet<String>,
                        suppression: &mut RequestScopedSuppression,
                        unrecordable_failure: &mut bool,
                        unresolved_read: &mut bool,
                        active_time_refused: &mut bool,
                        other_unresolved_read: &mut bool| {
            if let Some(pending) = &pending {
                for request in &pending.write_requests {
                    if request.context_id != context_id || !suppression.serviceable(request) {
                        continue;
                    }
                    if !snapshots.contains_key(&request.token) {
                        continue;
                    }
                    match selected_complete(
                        context_id,
                        request.selected_work,
                        request.read_active_time_ms,
                    ) {
                        Ok(true) => {
                            completed.insert(request.token.clone());
                        }
                        Ok(false) => {}
                        Err(error) => {
                            let terminal = suppression.selected_read_failed(
                                root,
                                request,
                                &error,
                                work.covered_events,
                            );
                            *unrecordable_failure |= terminal;
                            *unresolved_read |= !terminal;
                            *active_time_refused |=
                                matches!(error, CgError::ReadActiveTimeExceeded { .. });
                            *other_unresolved_read |= !terminal
                                && !matches!(error, CgError::ReadActiveTimeExceeded { .. });
                        }
                    }
                }
            }
        };
        if !work.work_required {
            complete(
                &mut completed,
                suppression,
                &mut unrecordable_failure,
                &mut unresolved_read,
                &mut active_time_refused,
                &mut other_unresolved_read,
            );
            continue;
        }
        if work.live_projection_lease {
            continue;
        }
        if suppression.failed.contains(&context_id) {
            continue;
        }
        let selection_id = selected_identities.get(&context_id).copied();
        let repeated = suppression
            .unchanged
            .entry((context_id, selection_id))
            .or_insert((work.clone(), 0));
        if repeated.0 == work {
            repeated.1 += 1;
        } else {
            *repeated = (work.clone(), 1);
        }
        if repeated.1 > WORKER_RETRIES {
            if selection_id.is_some() {
                suppression.unrecorded.insert((context_id, selection_id));
                unrecordable_failure = true;
            } else {
                suppression.failed.insert(context_id);
            }
            continue;
        }
        let coordination_attempts = suppression
            .coordination
            .entry((context_id, selection_id))
            .or_default();
        let (recorded, done, refused, terminal_refusal, coordination_retry) = serve(
            root,
            embedder,
            profile,
            (
                context_id,
                selected_policy,
                selected_work.get(&context_id).cloned(),
                selection_id,
                work.covered_events,
            ),
            &mut suppression.failed,
            coordination_attempts,
            embedding_worker_policy,
            read_memory_limit_bytes,
        );
        served |= !coordination_retry;
        if coordination_retry {
            // The dedicated selection counter bounds these retries even when each refused
            // attempt creates a different assessment. Keep ordinary unchanged-work accounting
            // from suppressing this transient attempt before its own allowance is exhausted.
            suppression.unchanged.remove(&(context_id, selection_id));
        }
        if !recorded {
            suppression.unrecorded.insert((context_id, selection_id));
            if selection_id.is_none() {
                suppression.failed.insert(context_id);
            }
        }
        unrecordable_failure |= !recorded;
        memory_refused |= refused;
        if done {
            suppression.unchanged.remove(&(context_id, selection_id));
            complete(
                &mut completed,
                suppression,
                &mut unrecordable_failure,
                &mut unresolved_read,
                &mut active_time_refused,
                &mut other_unresolved_read,
            );
        }
        if terminal_refusal
            && let Some(request) = pending.as_ref().and_then(|pending| {
                pending.write_requests.iter().find(|request| {
                    request.context_id == context_id
                        && Some(request.selected_work) == selection_id
                        && suppression.serviceable(request)
                })
            })
        {
            // The recorded typed refusal belongs to the exact selection served this turn.
            // Other requests, even in this Context, retain their own work and ceilings.
            completed.insert(request.token.clone());
            if pending.as_ref().is_some_and(|pending| {
                pending.write_requests.iter().any(|next| {
                    next.context_id == context_id && !next.completed && next.token != request.token
                })
            }) {
                suppression.failed.remove(&context_id);
                suppression.unchanged.remove(&(context_id, selection_id));
            }
        }
    }
    CatchUpResult {
        known: !unresolved_read,
        active_time_refused: active_time_refused && !other_unresolved_read,
        unrecordable_failure,
        completed,
        memory_refused,
        served,
    }
}

/// Only a typed active-time refusal spends the durable accepted-selection allowance.
fn record_aggregate_read_timeout(
    root: &Path,
    suppression: &RequestScopedSuppression,
    error: &CgError,
    assessed: Option<&PendingBackgroundRequest>,
) -> bool {
    let CgError::ReadActiveTimeExceeded { limit_ms } = error else {
        return false;
    };
    eprintln!("background search assessment refused: {error}");
    let Some(pending) = assessed else {
        return true;
    };
    let minimum = pending
        .write_requests
        .iter()
        .filter(|request| suppression.serviceable(request))
        .map(|request| request.read_active_time_ms)
        .min();
    let mut terminal = false;
    let mut retry = false;
    let mut identities = BTreeSet::new();
    for request in &pending.write_requests {
        if suppression.serviceable(request)
            && Some(request.read_active_time_ms) == minimum
            && identities.insert((request.context_id, request.selected_work))
        {
            match record_read_timeout(
                root,
                request.context_id,
                request.selected_work,
                request.read_timeout_covered_events,
                false,
                *limit_ms,
            ) {
                Ok(attempts) => {
                    terminal |= attempts >= WORKER_RETRIES;
                    retry |= attempts < WORKER_RETRIES;
                }
                Err(_) => terminal = true,
            }
        }
    }
    if retry {
        #[cfg(any(
            context_graph_internal_test_support,
            feature = "event-search-test-support"
        ))]
        pause_after_read_timeout_for_test(root);
        let mut backoff = Duration::from_millis(40);
        pause_before_recheck(&mut backoff, UNOBSERVABLE_BACKOFF_CEILING);
    }
    terminal
}

/// Save the consecutive refusal allowance with every owner of the exact selected identity.
/// Queue publication and process restarts cannot grant that selection another allowance.
fn record_read_timeout(
    root: &Path,
    context_id: ContextId,
    selection_id: uuid::Uuid,
    covered_events: u64,
    failure_recorded: bool,
    limit_ms: u64,
) -> Result<usize, String> {
    update_worker_policy(root, |existing| {
        let mut policy = existing.ok_or("the accepted search request is missing")?;
        let ordinal = next_refusal_ordinal(&policy)?;
        let pending = policy
            .pending_background_request
            .as_mut()
            .ok_or("the accepted search request is missing")?;
        let attempts = pending
            .write_requests
            .iter()
            .filter(|entry| entry.context_id == context_id && entry.selected_work == selection_id)
            .map(|entry| {
                if covered_events > entry.read_timeout_covered_events {
                    0
                } else {
                    entry.read_timeout_no_progress.saturating_add(1)
                }
            })
            .max()
            .ok_or("the accepted search selection is missing")?;
        for entry in &mut pending.write_requests {
            if entry.context_id == context_id && entry.selected_work == selection_id {
                entry.read_timeout_no_progress = attempts;
                entry.read_timeout_covered_events =
                    entry.read_timeout_covered_events.max(covered_events);
                entry.read_timeout_failure_recorded = failure_recorded;
                entry.read_timeout_ordinal = ordinal;
                entry.read_timeout_limit_ms = limit_ms;
            }
        }
        Ok((policy, attempts))
    })
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn pause_after_read_timeout_for_test(root: &Path) {
    let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") else {
        return;
    };
    let directory = PathBuf::from(directory);
    if directory.join("pause-after-read-timeout").is_file() {
        let _ = std::fs::write(
            directory.join("read-timeout-recovery-entered"),
            db_path(root).as_os_str().as_encoded_bytes(),
        );
        wait_for_test_release(|| directory.join("release-read-timeout-recovery").is_file());
    }
}

/// Build one Context's projection. The core retries the exact physical claim, renewal or commit
/// reopen in place. Typed coordination and active-time refusals may resume durable selected
/// work under their bounded allowances; active-time accounting survives process restart.
#[allow(clippy::too_many_arguments)]
fn serve(
    root: &Path,
    embedder: &EmbedderRegistration,
    profile: &EventSearchProfile,
    (context_id, work_policy, selected_work, selection_id, covered_events): (
        ContextId,
        EventSearchWorkPolicy,
        Option<EventSearchSelectedWork>,
        Option<uuid::Uuid>,
        u64,
    ),
    failed: &mut BTreeSet<ContextId>,
    coordination_attempts: &mut usize,
    embedding_worker_policy: &mut EmbeddingWorkerPolicyResolver,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> (bool, bool, bool, bool, bool) {
    let queue_backed = selected_work.is_some();
    let selected_work = match selected_work {
        Some(selected) => selected,
        None => match open_reader(root, read_memory_limit_bytes)
            .and_then(|reader| reader.select_event_search_work(context_id, work_policy))
        {
            Ok(selected) => selected,
            Err(_) => return (false, false, false, false, false),
        },
    };
    let mut request = run_request(
        root,
        context_id,
        embedder.clone(),
        profile,
        embedding_worker_policy.current(),
        read_memory_limit_bytes,
    );
    if let Some(selection_id) = selection_id {
        let Ok(pending) = pending_background_request_checked(root) else {
            return (false, false, false, false, false);
        };
        if let Some(limit) = pending
            .iter()
            .flat_map(|pending| &pending.write_requests)
            .filter(|entry| {
                entry.context_id == context_id
                    && entry.selected_work == selection_id
                    && !entry.selection_retired
            })
            .map(|entry| entry.read_active_time_ms)
            .min()
        {
            request.store.read_active_time_ms = limit;
        }
    }
    let read_active_time_ms = request.store.read_active_time_ms;
    let (observed, admitted) = run_observed_with_embedding_worker_policy(
        root,
        request,
        embedding_worker_policy,
        work_policy,
        selected_work,
        queue_backed,
    );
    let memory_refused = matches!(
        &observed.result,
        Err(CgError::WriteDidNotFitTheMemoryLimit { .. })
    );
    if let Err(error) = &observed.result {
        eprintln!("background search refused: {error}");
        if record_background_memory_refusal(root, context_id, error, &admitted).is_err() {
            return (false, false, false, false, false);
        }
    }
    if let Err(CgError::ReadActiveTimeExceeded { limit_ms }) = &observed.result {
        // The runner has closed its failed cursor, dropped its Store and released its exact
        // claim before returning. Observe committed progress and persist the allowance before
        // pausing, so a worker restart resumes the same bounded selection.
        let progress = context_graph::ConsumerReader::open_with_read_limits(
            db_path(root),
            context_graph::event_history_reader_declaration(),
            read_memory_limit_bytes,
            read_active_time_ms,
        )
        .and_then(|reader| {
            reader.event_search_work_with_policy(
                ContextSelector::one(context_id),
                &super::search_work_spec(profile),
                work_policy,
            )
        })
        .ok()
        .and_then(|work| work.into_iter().next())
        .map(|work| work.covered_events)
        .unwrap_or(covered_events);
        let attempts = match selection_id {
            Some(selection_id) => match record_read_timeout(
                root,
                context_id,
                selection_id,
                progress,
                observed.failure_disposition == EventSearchFailureDisposition::Recorded,
                *limit_ms,
            ) {
                Ok(attempts) => attempts,
                Err(_) => return (false, false, false, false, false),
            },
            None => {
                *coordination_attempts += 1;
                *coordination_attempts
            }
        };
        if attempts < WORKER_RETRIES {
            #[cfg(any(
                context_graph_internal_test_support,
                feature = "event-search-test-support"
            ))]
            pause_after_read_timeout_for_test(root);
            let mut backoff = Duration::from_millis(20).saturating_mul(1 << attempts.min(4));
            pause_before_recheck(&mut backoff, UNOBSERVABLE_BACKOFF_CEILING);
            return (true, false, false, false, true);
        }
        if observed.failure_disposition != EventSearchFailureDisposition::Recorded {
            return (false, false, false, false, false);
        }
    } else if matches!(
        &observed.result,
        Err(CgError::ReaderCoordinationUnavailable { .. })
    ) {
        *coordination_attempts += 1;
        if *coordination_attempts < WORKER_RETRIES {
            let mut backoff = Duration::from_millis(20);
            pause_before_recheck(&mut backoff, UNOBSERVABLE_BACKOFF_CEILING);
            return (true, false, false, false, true);
        }
        // Only the owning attempt's recorded refusal licenses acknowledgement. An unrecorded
        // or superseded refusal stays pending, and this worker stops rather than spinning.
        if observed.failure_disposition != EventSearchFailureDisposition::Recorded {
            return (false, false, false, false, false);
        }
    } else {
        *coordination_attempts = 0;
    }
    match (observed.result, observed.failure_disposition) {
        (Ok(_), EventSearchFailureDisposition::NoFailure) => (true, true, false, false, false),
        (Err(_), EventSearchFailureDisposition::Recorded) => {
            failed.insert(context_id);
            (true, false, memory_refused, !memory_refused, false)
        }
        // Another attempt owns the generation. It remains responsible for the work and this
        // worker must not turn that takeover into a failed Context.
        (Err(_), EventSearchFailureDisposition::Stale) => (true, false, false, false, false),
        // Only the core knows whether its matching-attempt write committed. A pre-attempt error or
        // refused failure write leaves the request pending and makes the detached worker fail.
        (Err(_), EventSearchFailureDisposition::Unrecorded)
        | (_, EventSearchFailureDisposition::NoFailure)
        | (Ok(_), _) => (false, false, memory_refused, false, false),
    }
}

/// The detached worker: take the lock or exit, catch up, release, and re-check for work committed
/// while the lock was held so no import's pending work is stranded.
pub(crate) fn handle_event_search_worker(
    args: &[String],
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> i32 {
    let root = super::super::data_root();
    let embedder = match search_registration() {
        Ok(embedder) => embedder,
        Err(_) => return 1,
    };
    let [serialized_worker] = args else {
        return 1;
    };
    // New workers carry the initiating request's validated cap and, for a config-only forget
    // kick, the one earlier publication that must not outlive its original worker.
    let Ok(envelope) = serde_json::from_str::<WorkerEnvelope>(serialized_worker) else {
        return 1;
    };
    let (
        mut profile,
        initial_embedding_workers,
        initial_work_policy,
        ignored_publication,
        retry_worker_lock,
    ) = (
        envelope.profile,
        envelope.embedding_workers,
        envelope.work_policy,
        envelope.ignored_publication,
        envelope.retry_worker_lock,
    );
    if profile.validate().is_err() {
        return 1;
    };
    #[cfg(feature = "event-search-test-support")]
    apply_store_open_retry_allowance_for_test();
    let mut embedding_worker_policy = EmbeddingWorkerPolicyResolver::new(
        &root,
        initial_embedding_workers,
        initial_work_policy,
        ignored_publication,
    );
    let mut suppression = RequestScopedSuppression {
        token: None,
        failed: BTreeSet::new(),
        unchanged: BTreeMap::new(),
        coordination: BTreeMap::new(),
        unrecorded: BTreeSet::new(),
        aggregate_reads: BTreeMap::new(),
    };
    let mut unknown_reads = 0;
    let mut worker_failed = false;
    let mut lease_recheck_backoff = Duration::from_millis(20);
    loop {
        #[cfg(any(
            context_graph_internal_test_support,
            feature = "event-search-test-support"
        ))]
        if worker_failed
            && let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER")
        {
            let directory = PathBuf::from(directory);
            if directory
                .join("pause-before-failed-worker-relock")
                .is_file()
            {
                let _ = std::fs::write(directory.join("failed-worker-relock-entered"), b"unlocked");
                wait_for_test_release(|| directory.join("release-failed-worker-relock").is_file());
            }
        }
        let held = if retry_worker_lock {
            let Some(held) = acquire_worker_lock_within(&root, WRITER_TRANSITION_ALLOWANCE) else {
                return 1;
            };
            held
        } else {
            let Some(held) = lock(&root) else {
                return i32::from(worker_failed);
            };
            held
        };
        let serviced_request = pending_background_request(&root);
        suppression.observe(serviced_request.as_ref());
        if let Ok(next) = configured_passage_profile(&root)
            && next != profile
        {
            profile = next;
            suppression.clear();
            lease_recheck_backoff = Duration::from_millis(20);
        }
        #[cfg(any(
            context_graph_internal_test_support,
            feature = "event-search-test-support"
        ))]
        pause_before_catch_up();
        let catch_up = if serviced_request.as_ref().is_some_and(|pending| {
            !pending.write_requests.is_empty()
                && pending.write_requests.iter().all(|request| {
                    request.completed || request.read_timeout_no_progress >= WORKER_RETRIES
                })
        }) {
            CatchUpResult {
                known: true,
                active_time_refused: false,
                unrecordable_failure: serviced_request
                    .iter()
                    .flat_map(|pending| &pending.write_requests)
                    .any(|request| {
                        !request.completed
                            && !request.read_timeout_failure_recorded
                            && !request.refusal_superseded
                    }),
                completed: serviced_request
                    .iter()
                    .flat_map(|pending| &pending.write_requests)
                    .filter(|request| {
                        !request.completed
                            && (request.read_timeout_failure_recorded || request.refusal_superseded)
                    })
                    .map(|request| request.token.clone())
                    .collect(),
                memory_refused: false,
                served: false,
            }
        } else {
            catch_up(
                &root,
                &embedder,
                &profile,
                &mut suppression,
                &mut embedding_worker_policy,
                read_memory_limit_bytes,
            )
        };
        let cleanup_error = catch_up
            .known
            .then(|| {
                mark_completed_requests(
                    &root,
                    serviced_request.as_ref(),
                    &catch_up.completed,
                    catch_up.served,
                    read_memory_limit_bytes,
                    None,
                    Some(&super::search_work_spec(&profile)),
                )
            })
            .and_then(Result::err);
        if let Some(error) = &cleanup_error {
            eprintln!("background search cleanup refused: {error}");
            worker_failed = true;
        }
        #[cfg(any(
            context_graph_internal_test_support,
            feature = "event-search-test-support"
        ))]
        pause_after_catch_up_before_unlock();
        drop(held);

        // Cleanup failure leaves selection retirement pending, but does not cancel accepted
        // service obligations. Reuse the normal queue, suppression, coordination and lease
        // rechecks below for both arrivals and selections present at turn start. Once no
        // serviceable work remains, worker_failed preserves the nonzero result.

        // A failed matching-attempt write must remain pending, but later accepted selections
        // still own service. Suppression follows the selection identity through queue changes.
        worker_failed |= catch_up.unrecordable_failure;
        unknown_reads = if catch_up.known || catch_up.active_time_refused {
            0
        } else {
            unknown_reads + 1
        };
        // Exhaust only the identities assessed under this lock. Check the current queue after
        // unlock before stopping: an import may have accepted new work during the failed read.
        if !catch_up.known
            && let Some(pending) = serviced_request.as_ref()
            && !pending.write_requests.is_empty()
        {
            if !catch_up.active_time_refused {
                worker_failed |= suppression.aggregate_read_failed(pending);
            }
            unknown_reads = 0;
            match pending_background_request_checked(&root) {
                Ok(Some(current))
                    if current
                        .write_requests
                        .iter()
                        .any(|request| suppression.serviceable(request)) => {}
                Ok(_) => return 1,
                Err(_) => return 1,
            }
        } else if unknown_reads >= WORKER_RETRIES {
            return 1;
        }

        if let Some(request) = serviced_request.as_ref() {
            // The long lock is released before this short compare-and-clear. If a newer request
            // arrived while we served this pass, its different token remains and the next loop
            // services it. If no newer request arrived, this is the actual owner acknowledgement.
            match clear_background_request_if_current(&root, request) {
                Ok(false) => {
                    if pending_background_request(&root).is_some_and(|pending| {
                        pending.write_requests.is_empty()
                            || pending
                                .write_requests
                                .iter()
                                .any(|request| suppression.serviceable(request))
                    }) {
                        continue;
                    }
                }
                Ok(true) => {}
                Err(_) => return 1,
            }
        }

        if catch_up.memory_refused
            && pending_background_request(&root).is_some_and(|pending| {
                pending
                    .write_requests
                    .iter()
                    .any(|request| suppression.serviceable(request))
            })
        {
            suppression.clear();
            continue;
        }

        // A queued turn ends when its accepted selections are acknowledged. A standalone
        // turn still rechecks refused reads and live leases below.
        match pending_background_request_checked(&root) {
            Ok(None)
                if serviced_request
                    .as_ref()
                    .is_some_and(|request| !request.write_requests.is_empty()) =>
            {
                return i32::from(worker_failed);
            }
            Ok(Some(pending))
                if !pending.write_requests.is_empty()
                    && pending
                        .write_requests
                        .iter()
                        .all(|request| !suppression.serviceable(request)) =>
            {
                return i32::from(worker_failed);
            }
            Ok(None) | Ok(Some(_)) => {}
            Err(_) => return 1,
        }

        // Releasing before the final read preserves the import handoff: an import that observed
        // our lock either appears in this read, or observes it released and starts a new worker.
        if let Ok(next) = configured_passage_profile(&root)
            && next != profile
        {
            profile = next;
            suppression.clear();
            lease_recheck_backoff = Duration::from_millis(20);
            continue;
        }
        let assessment_request = match pending_background_request_checked(&root) {
            Ok(pending) => pending,
            Err(_) => return 1,
        };
        match requested_work_assessments(
            &root,
            &profile,
            embedding_worker_policy.current_work_policy(),
            read_memory_limit_bytes,
            assessment_request.as_ref(),
            &suppression,
        ) {
            Ok(still) => {
                let pending = pending_background_request(&root);
                let unfailed = still
                    .iter()
                    .filter(|work| {
                        work.work_required
                            && !suppression.failed.contains(&work.context_id)
                            && pending.as_ref().is_none_or(|pending| {
                                pending.write_requests.is_empty()
                                    || pending.write_requests.iter().any(|request| {
                                        request.context_id == work.context_id
                                            && suppression.serviceable(request)
                                    })
                            })
                    })
                    .collect::<Vec<_>>();
                if unfailed.is_empty() {
                    // Another owned selector may already be covered by the pass we just served.
                    // Acknowledge it in its own turn, retaining suppressed unfinished requests.
                    if pending_background_request(&root).is_some_and(|pending| {
                        pending.write_requests.iter().any(|request| {
                            suppression.serviceable(request)
                                && !suppression.failed.contains(&request.context_id)
                        })
                    }) {
                        continue;
                    }
                    return i32::from(worker_failed);
                }
                if unfailed.iter().all(|work| work.live_projection_lease) {
                    // The live holder may be completing normally or may be an interrupted worker
                    // whose durable lease has not expired yet. Dropping our OS lock before this
                    // pause lets the owner finish; the next pass serves expired work without an
                    // additional import.
                    pause_before_recheck(&mut lease_recheck_backoff, UNOBSERVABLE_BACKOFF_CEILING);
                    continue;
                }
                lease_recheck_backoff = Duration::from_millis(20);
            }
            Err(error) => {
                worker_failed |= record_aggregate_read_timeout(
                    &root,
                    &suppression,
                    &error,
                    assessment_request.as_ref(),
                );
                let active_time_refused = matches!(error, CgError::ReadActiveTimeExceeded { .. })
                    && assessment_request
                        .as_ref()
                        .is_some_and(|pending| !pending.write_requests.is_empty());
                // A refused read says nothing about queue exhaustion. Retry it within the
                // existing bound, independently of how many successful passes preceded it.
                if let Some(pending) = assessment_request.as_ref()
                    && !pending.write_requests.is_empty()
                {
                    if !active_time_refused {
                        worker_failed |= suppression.aggregate_read_failed(pending);
                    }
                    match pending_background_request_checked(&root) {
                        Ok(Some(current))
                            if current
                                .write_requests
                                .iter()
                                .any(|request| suppression.serviceable(request)) => {}
                        Ok(_) => return 1,
                        Err(_) => return 1,
                    }
                } else {
                    unknown_reads += 1;
                }
            }
        }
        if unknown_reads >= WORKER_RETRIES {
            return 1;
        }
    }
}

pub(crate) struct ForegroundOutcome {
    pub result: Result<(), String>,
    pub work: Option<EventSearchWorkAssessment>,
    pub embedding_invocations: u64,
}

fn requested_foreground_selection(
    root: &Path,
    context_id: ContextId,
    work_policy: EventSearchWorkPolicy,
    read_memory_limit_bytes: NonZeroUsize,
) -> context_graph::Result<EventSearchSelectedWork> {
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
        let marker = Path::new(&directory).join("refuse-foreground-selection-active-time-count");
        if let Some(count) = std::fs::read_to_string(&marker)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|count| *count > 0)
        {
            let _ = std::fs::write(marker, (count - 1).to_string());
            return Err(CgError::ReadActiveTimeExceeded {
                limit_ms: crate::read_config::active_time_in_force().get(),
            });
        }
    }
    open_reader(root, read_memory_limit_bytes)
        .and_then(|reader| reader.select_event_search_work(context_id, work_policy))
}

fn requested_foreground_work(
    root: &Path,
    context_id: ContextId,
    profile: &EventSearchProfile,
    work_policy: EventSearchWorkPolicy,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> context_graph::Result<EventSearchWorkAssessment> {
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    if let Some(directory) = std::env::var_os("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER") {
        let directory = PathBuf::from(directory);
        let after =
            std::fs::read_to_string(directory.join("foreground-assessment-refusal-after-runs"))
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
        let runs = std::fs::read_to_string(directory.join("embedding-worker-requests"))
            .unwrap_or_default()
            .lines()
            .count();
        let marker = directory.join("refuse-foreground-assessment-active-time-count");
        if runs >= after
            && let Some(count) = std::fs::read_to_string(&marker)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|count| *count > 0)
        {
            let _ = std::fs::write(marker, (count - 1).to_string());
            return Err(CgError::ReadActiveTimeExceeded {
                limit_ms: crate::read_config::active_time_in_force().get(),
            });
        }
    }
    requested_work(
        root,
        context_id,
        profile,
        work_policy,
        read_memory_limit_bytes,
    )
}

fn requested_work(
    root: &Path,
    context_id: ContextId,
    profile: &EventSearchProfile,
    work_policy: EventSearchWorkPolicy,
    read_memory_limit_bytes: NonZeroUsize,
) -> context_graph::Result<EventSearchWorkAssessment> {
    open_reader(root, read_memory_limit_bytes)
        .and_then(|reader| {
            reader.event_search_work_with_policy(
                ContextSelector::one(context_id),
                &super::search_work_spec(profile),
                work_policy,
            )
        })?
        .into_iter()
        .next()
        .ok_or_else(|| CgError::Engine("requested search assessment returned no Context".into()))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ForegroundHandoffPhase {
    Projection,
    MainThread,
    InterruptListener,
    Complete,
}

struct ForegroundInterruptState {
    held: Option<WorkerLock>,
    handoff: ForegroundHandoffPhase,
}

struct ForegroundInterruptShared {
    state: Mutex<ForegroundInterruptState>,
    handoff_changed: Condvar,
}

/// SIGINT has to release the foreground turn while the embedder is still blocked and remain owned
/// by this command through queued handoff and final status refresh. The handoff phase gives either
/// the command thread or its listener sole responsibility for the queued request.
struct ForegroundInterruptGuard {
    stop: Arc<AtomicBool>,
    stop_tx: Option<oneshot::Sender<()>>,
    shared: Arc<ForegroundInterruptShared>,
    listener: Option<thread::JoinHandle<()>>,
}

struct ForegroundHandoffCompletion {
    shared: Arc<ForegroundInterruptShared>,
}

impl Drop for ForegroundHandoffCompletion {
    fn drop(&mut self) {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("foreground interrupt state");
        if state.handoff == ForegroundHandoffPhase::MainThread {
            state.handoff = ForegroundHandoffPhase::Complete;
            self.shared.handoff_changed.notify_all();
        }
    }
}

impl ForegroundInterruptGuard {
    fn arm(root: &Path, profile: &EventSearchProfile) -> Result<Self, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let (stop_tx, mut stop_rx) = oneshot::channel();
        let shared = Arc::new(ForegroundInterruptShared {
            state: Mutex::new(ForegroundInterruptState {
                held: None,
                handoff: ForegroundHandoffPhase::Projection,
            }),
            handoff_changed: Condvar::new(),
        });
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let listener_stop = Arc::clone(&stop);
        let listener_shared = Arc::clone(&shared);
        let root = root.to_path_buf();
        let profile = profile.clone();
        let listener = thread::Builder::new()
            .name("cg-event-search-interrupt".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .enable_time()
                    .build();
                let Ok(runtime) = runtime else {
                    let _ = ready_tx.send(Err("the foreground interrupt listener could not start"));
                    return;
                };
                runtime.block_on(async move {
                    let signal =
                        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt());
                    let Ok(mut signal) = signal else {
                        let _ = ready_tx.send(Err(
                            "the foreground interrupt listener could not register SIGINT",
                        ));
                        return;
                    };
                    let _ = ready_tx.send(Ok(()));
                    tokio::select! {
                        _ = signal.recv() => {
                            if !listener_stop.load(Ordering::SeqCst) {
                                let (held, owns_handoff) = {
                                    let mut state = listener_shared
                                        .state
                                        .lock()
                                        .expect("foreground interrupt state");
                                    loop {
                                        match state.handoff {
                                            ForegroundHandoffPhase::Projection => {
                                                state.handoff =
                                                    ForegroundHandoffPhase::InterruptListener;
                                                break (state.held.take(), true);
                                            }
                                            ForegroundHandoffPhase::MainThread => {
                                                state = listener_shared
                                                    .handoff_changed
                                                    .wait(state)
                                                    .expect("foreground handoff state");
                                            }
                                            ForegroundHandoffPhase::InterruptListener
                                            | ForegroundHandoffPhase::Complete => {
                                                break (None, false);
                                            }
                                        }
                                    }
                                };
                                drop(held);
                                if owns_handoff {
                                    handoff_interrupted_foreground(&root, &profile);
                                }
                                std::process::exit(130);
                            }
                        }
                        _ = &mut stop_rx => {}
                    }
                });
            })
            .map_err(|error| {
                format!("the foreground interrupt listener could not start: {error}")
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                stop,
                stop_tx: Some(stop_tx),
                shared,
                listener: Some(listener),
            }),
            Ok(Err(error)) => {
                let _ = listener.join();
                Err(error.into())
            }
            Err(_) => {
                let _ = listener.join();
                Err("the foreground interrupt listener stopped before registration".into())
            }
        }
    }

    fn hold(&self, worker_lock: WorkerLock) {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("foreground interrupt state");
        if state.handoff == ForegroundHandoffPhase::Projection {
            state.held = Some(worker_lock);
        }
    }

    fn run_handoff<T>(&self, operation: impl FnOnce() -> T) -> Option<T> {
        let held = {
            let mut state = self
                .shared
                .state
                .lock()
                .expect("foreground interrupt state");
            if state.handoff != ForegroundHandoffPhase::Projection {
                return None;
            }
            state.handoff = ForegroundHandoffPhase::MainThread;
            state.held.take()
        };
        drop(held);
        let completion = ForegroundHandoffCompletion {
            shared: Arc::clone(&self.shared),
        };
        let result = operation();
        drop(completion);
        Some(result)
    }

    fn stop_and_join(mut self) {
        self.shared
            .state
            .lock()
            .expect("foreground interrupt state")
            .held
            .take();
        self.stop.store(true, Ordering::SeqCst);
        let _ = self
            .stop_tx
            .take()
            .expect("foreground interrupt stop sender")
            .send(());
        if let Some(listener) = self.listener.take() {
            let _ = listener.join();
        }
    }
}

/// Handoff reads the bounded atomic record directly, launches only its queued request, and leaves it durable if
/// spawning or child admission cannot complete.
fn handoff_interrupted_foreground(root: &Path, profile: &EventSearchProfile) {
    let Some(request) = read_event_search_worker_policy(root)
        .ok()
        .flatten()
        .and_then(|contents| decode_worker_policy(&contents).ok())
        .and_then(|policy| policy.pending_background_request)
    else {
        return;
    };
    let _ = spawn_worker(WorkerEnvelope {
        profile: profile.clone(),
        embedding_workers: request.embedding_workers,
        work_policy: request.work_policy,
        ignored_publication: request.ignored_publication,
        retry_worker_lock: true,
    });
}

/// `--search wait`: obtain the worker turn, then run the same runner in the foreground until this
/// Context is caught up or the projection fails.
pub(crate) fn run_in_foreground(
    root: &Path,
    context_id: ContextId,
    desired_profile: &EventSearchProfile,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
    read_memory_limit_bytes: std::num::NonZeroUsize,
) -> ForegroundOutcome {
    let mut work = None;
    let mut embedding_invocations = 0;
    let interrupt = match ForegroundInterruptGuard::arm(root, desired_profile) {
        Ok(interrupt) => interrupt,
        Err(error) => {
            return ForegroundOutcome {
                result: Err(error),
                work,
                embedding_invocations,
            };
        }
    };
    let result = (|| {
        publish_foreground_embedding_worker_policy(root, embedding_workers, work_policy)?;
        let embedder = search_registration().map_err(|error| error.to_string())?;
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
        interrupt.hold(held);
        // Freeze the refused identities before service. A successful retry may complete their
        // original frontiers, but cannot acknowledge a refusal accepted later in this turn.
        let observed_timeouts = pending_background_request_checked(root)?.map(|mut pending| {
            pending.write_requests.retain(|request| {
                request.context_id == context_id
                    && request.read_timeout_no_progress >= WORKER_RETRIES
                    && !request.refusal_superseded
            });
            pending
        });
        let observed_foreground_refusals = worker_policy_checked(root)?
            .map(|policy| {
                policy
                    .foreground_read_refusals
                    .into_iter()
                    .filter(|refusal| refusal.context_id == context_id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut selected_work = None;
        let mut unchanged = 0;
        let mut read_timeout_no_progress = 0usize;
        let mut read_timeout_covered_events = None;
        let mut read_timeout = None;
        loop {
            let assessment = (|| {
                if selected_work.is_none() {
                    selected_work = Some(requested_foreground_selection(
                        root,
                        context_id,
                        work_policy,
                        read_memory_limit_bytes,
                    )?);
                }
                requested_foreground_work(
                    root,
                    context_id,
                    desired_profile,
                    work_policy,
                    read_memory_limit_bytes,
                )
            })();
            let current = match assessment {
                Ok(current) => current,
                Err(error @ CgError::ReadActiveTimeExceeded { .. }) => {
                    read_timeout_no_progress = read_timeout_no_progress.saturating_add(1);
                    read_timeout = Some(error);
                    if read_timeout_no_progress >= WORKER_RETRIES {
                        break;
                    }
                    let mut backoff = Duration::from_millis(20)
                        .saturating_mul(1 << read_timeout_no_progress.min(4));
                    pause_before_recheck(&mut backoff, UNOBSERVABLE_BACKOFF_CEILING);
                    continue;
                }
                Err(error) => return Err(error.to_string()),
            };
            let required = current.work_required;
            if read_timeout_covered_events.is_some_and(|covered| current.covered_events > covered) {
                read_timeout_no_progress = 0;
            }
            read_timeout_covered_events = Some(
                read_timeout_covered_events
                    .unwrap_or(0)
                    .max(current.covered_events),
            );
            unchanged = if work.as_ref() == Some(&current) {
                unchanged + 1
            } else {
                0
            };
            work = Some(current);
            if !required {
                if !observed_foreground_refusals.is_empty() {
                    let store = context_graph::Store::open(store_config(
                        root,
                        None,
                        read_memory_limit_bytes,
                    ))
                    .map_err(|error| error.to_string())?;
                    let completed = completed_foreground_read_refusals(
                        &store,
                        context_id,
                        &super::search_work_spec(desired_profile),
                        &observed_foreground_refusals,
                    )
                    .map_err(|error| error.to_string())?;
                    drop(store);
                    acknowledge_foreground_read_refusals(root, &completed)?;
                }
                if let Some(pending) = &observed_timeouts
                    && !pending.write_requests.is_empty()
                {
                    let mut config = store_config(root, None, read_memory_limit_bytes);
                    config.read_active_time_ms = pending
                        .write_requests
                        .iter()
                        .map(|request| request.read_active_time_ms)
                        .min()
                        .expect("observed timeout refusals are nonempty");
                    let store =
                        context_graph::Store::open(config).map_err(|error| error.to_string())?;
                    let completed = completed_saved_memory_refusals(
                        &store,
                        context_id,
                        &super::search_work_spec(desired_profile),
                        pending,
                    )
                    .map_err(|error| error.to_string())?;
                    drop(store);
                    acknowledge_completed_memory_refusal(
                        root,
                        context_id,
                        &completed,
                        read_memory_limit_bytes,
                    )
                    .map_err(|error| error.to_string())?;
                }
                return Ok(());
            }
            if unchanged >= FOREGROUND_UNCHANGED_RETRIES {
                return Err("search repeatedly returned without advancing the requested projection; run the same import with --search wait again".into());
            }
            if read_timeout_no_progress >= WORKER_RETRIES {
                break;
            }
            let observed = run_observed_with_fixed_embedding_workers(
                root,
                run_request(
                    root,
                    context_id,
                    embedder.clone(),
                    desired_profile,
                    embedding_workers,
                    read_memory_limit_bytes,
                ),
                embedding_workers,
                work_policy,
            );
            embedding_invocations += observed.embedding_invocations;
            match observed.result {
                Err(error @ CgError::ReadActiveTimeExceeded { .. }) => {
                    read_timeout_no_progress = read_timeout_no_progress.saturating_add(1);
                    read_timeout = Some(error);
                    let mut backoff = Duration::from_millis(20)
                        .saturating_mul(1 << read_timeout_no_progress.min(4));
                    pause_before_recheck(&mut backoff, UNOBSERVABLE_BACKOFF_CEILING);
                }
                result => {
                    result.map_err(|error| error.to_string())?;
                }
            }
        }
        let error = read_timeout.expect("a spent timeout allowance retains its condition");
        let message = format!(
            "{error}; search stopped after {WORKER_RETRIES} active-time refusals without committed coverage progress; inspect cg status and run the same import with --search wait again"
        );
        let CgError::ReadActiveTimeExceeded { limit_ms } = error else {
            unreachable!()
        };
        if let Err(persistence) =
            record_foreground_read_refusal(root, context_id, selected_work, limit_ms)
        {
            return Err(format!(
                "{message}; terminal search status could not be saved: {persistence}"
            ));
        }
        Err(message)
    })();
    // Foreground service owns only its selected Context. Once its long turn is released, hand any
    // coalesced background import or forget to the ordinary detached worker without replacing the
    // published choice; this runs after both foreground success and failure.
    let handoff_error = interrupt
        .run_handoff(|| handoff_after_foreground(root, desired_profile))
        .and_then(|result| result.err());
    #[cfg(any(
        context_graph_internal_test_support,
        feature = "event-search-test-support"
    ))]
    pause_after_foreground_handoff_before_refresh();
    // Refresh after both success and failure. If this read fails, return the last real snapshot
    // with an explicit limitation instead of converting the failed read into made-up progress.
    let mut result = match requested_work(
        root,
        context_id,
        desired_profile,
        work_policy,
        read_memory_limit_bytes,
    ) {
        Ok(current) => {
            let required = current.work_required;
            work = Some(current);
            result.and_then(|()| {
                if required {
                    Err(
                        "search still has requested projection work after the foreground run"
                            .into(),
                    )
                } else {
                    Ok(())
                }
            })
        }
        Err(refresh) => {
            let limitation = format!(
                "requested search coverage could not be refreshed: {refresh}; counts are the last observed snapshot"
            );
            Err(match result {
                Err(error) => format!("{error}; {limitation}"),
                Ok(()) => limitation,
            })
        }
    };
    if let Some(handoff_error) = handoff_error {
        result = Err(match result {
            Ok(()) => handoff_error,
            Err(error) => format!("{error}; {handoff_error}"),
        });
    }
    interrupt.stop_and_join();
    ForegroundOutcome {
        result,
        work,
        embedding_invocations,
    }
}
