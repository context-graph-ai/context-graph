//! Bounded construction of the derived event-history search projection.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use contextdb_core::{Error as EngineError, Value, Wallclock};
use contextdb_engine::{ReadLimits, ReadSession};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use sha2::{Digest, Sha256};

use crate::{
    CgError, ConversationEvent, EVENT_SEARCH_RENDERER_ID, EVENT_SEARCH_RENDERER_VERSION,
    EmbeddingId, EmbeddingTarget, EventCoverage, EventRenderRole, EventSearchBuildReceipt,
    EventSearchBuildRequest, EventSearchBuildTimings, EventSearchEmbedder, EventSearchGenerationId,
    EventSearchGenerationState, EventSearchPassageId, EventSearchProfile,
    EventSearchRebuildRequest, EventSearchRunRequest, EventWatermark,
    InterpretationActivationRefusalReason, InterpretationGenerationId, PassageEventSource, Result,
    Store, TxId,
};

use super::convert::map_engine_err;
use super::encode::{column_index, get_json, get_optional_text, get_text, get_uuid};
use super::event_interpretation::{
    CAPTURED_ANCHOR_COLUMNS, CAPTURED_EVENT_COLUMNS, captured_source_sha256,
};
use super::evidence::producer_value;

const MAX_STALE_RETRIES: usize = 8;

/// The longest a reopen backs off between attempts when no reader can be observed to wait on.
const UNOBSERVABLE_BACKOFF_CEILING: Duration = Duration::from_millis(500);

/// Open the store for one step of a projection run, waiting out direct readers.
///
/// A run claims a batch, CLOSES the store to render and embed, and reopens to commit. A reader that
/// happens to be loading at that reopen must not end the run: the claim it already holds is still
/// this run's, so a refusal here would strand the claim and lock the very next attempt out of its
/// own work with `another event-search worker owns the unexpired claim`. Readers finish, and the
/// reopen then succeeds, so this waits for them inside
/// [`crate::owner_control::READER_RELEASE_ALLOWANCE`] and surfaces the refusal only past it.
///
/// A wait nobody could observe says nothing about whether the store is free, so it is not taken as
/// a reason to give up: readers are short-lived by contract, so the reopen is simply asked again
/// after a growing pause, inside the same allowance.
fn open_for_run(
    config: &crate::StoreConfig,
    embedder: &crate::EmbedderRegistration,
) -> Result<Store> {
    let deadline = Instant::now() + crate::owner_control::READER_RELEASE_ALLOWANCE;
    let mut backoff = Duration::from_millis(20);
    loop {
        match Store::open_with_embedder_registrations(config.clone(), vec![embedder.clone()]) {
            Err(refusal @ CgError::StoreHeldByReaders { .. }) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(refusal);
                }
                match crate::owner_control::wait_for_reader_release_within(
                    &config.db_path,
                    deadline,
                ) {
                    crate::owner_control::ReaderRelease::Released => {}
                    crate::owner_control::ReaderRelease::Stopped => return Err(refusal),
                    crate::owner_control::ReaderRelease::Unobservable(_) => {
                        std::thread::sleep(backoff.min(remaining));
                        backoff = (backoff * 2).min(UNOBSERVABLE_BACKOFF_CEILING);
                    }
                }
            }
            other => return other,
        }
    }
}

/// Record that a projection could not run at all, so `cg status` answers `failed` with the reason
/// rather than `pending` for work no process is doing.
///
/// Status tells a user `pending` when a Context is uncovered and nothing is running — which is also
/// exactly what an abandoned Context looks like. A caller that gives up therefore owes the reader a
/// recorded reason, so the two are distinguishable and a later `cg import` or `cg forget` is the
/// obvious next move. `reason` is caller-authored text: it names a condition, never a path, a
/// process, or anything a source record said.
///
/// A Context whose search is still answering from a ready generation is left alone. Being behind a
/// ready head is `pending`, which is the truth; only a Context with nothing to answer from is
/// marked.
pub(crate) fn record_refusal_standalone(
    request: EventSearchRunRequest,
    reason: &str,
) -> Result<()> {
    let context_ids = request.contexts.context_ids();
    if context_ids.len() != 1 {
        return Err(CgError::Engine(
            "one recorded event-search refusal describes exactly one Context".into(),
        ));
    }
    let context_id = context_ids[0];
    let bounds = bounds(
        request.max_events_per_batch.get(),
        request.max_passages_per_batch.get(),
        request.max_bytes_per_batch.get(),
        request.lease_duration_ms.get(),
    );
    let store = open_for_run(&request.store, &request.embedder)?;
    let generation = ensure_generation(
        &store,
        context_id,
        &request.embedder,
        &request.profile,
        bounds,
        false,
        InterpretationSelection::Ordinary,
    )?;
    if matches!(generation.mode, GenerationMode::ExtendReady) {
        return Ok(());
    }
    store
        .internal_db()
        .execute(
            "UPDATE event_search_generations SET state = $state, sanitized_failure = $failure WHERE id = $id AND context_id = $context_id",
            &HashMap::from([
                ("state".to_string(), Value::Text("failed".into())),
                ("failure".to_string(), Value::Text(reason.to_string())),
                ("id".to_string(), Value::Uuid(generation.id.as_uuid())),
                (
                    "context_id".to_string(),
                    Value::Uuid(context_id.as_uuid()),
                ),
            ]),
        )
        .map_err(map_engine_err)?;
    Ok(())
}

/// What a run needs to let go of the claim it took, by its own token.
struct HeldClaim {
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    owner_token: String,
    lease_epoch: i64,
}

impl HeldClaim {
    fn of(claim: &Claim) -> Self {
        Self {
            context_id: claim.generation.context_id,
            generation_id: claim.generation.id,
            owner_token: claim.owner_token.clone(),
            lease_epoch: claim.lease_epoch,
        }
    }
}

/// Let go of a claim this run took but will not commit.
///
/// A run that returns without committing must not leave its claim standing: the lease is the
/// Store-enforced single-writer guard, so an orphan refuses the next attempt — the retry the caller
/// is about to make — for the rest of the lease, and the Context sits uncovered with nothing
/// running. Only the row this run owns is removed, matched on its own token and epoch, so a claim
/// that has already been taken over is left alone. A late result carrying this token is still
/// refused afterwards, because the commit path requires its own lease row to still be there.
///
/// Best effort by nature: a process that is killed outright releases nothing, and its lease expiry
/// remains the backstop.
fn release_claim(
    config: &crate::StoreConfig,
    embedder: &crate::EmbedderRegistration,
    held: &HeldClaim,
) {
    let Ok(store) = open_for_run(config, embedder) else {
        return;
    };
    let _ = store.internal_db().execute(
        "DELETE FROM event_search_leases WHERE context_id = $context_id AND generation_id = $generation_id AND owner_token = $owner_token AND lease_epoch = $lease_epoch",
        &HashMap::from([
            (
                "context_id".to_string(),
                Value::Uuid(held.context_id.as_uuid()),
            ),
            (
                "generation_id".to_string(),
                Value::Uuid(held.generation_id.as_uuid()),
            ),
            (
                "owner_token".to_string(),
                Value::Text(held.owner_token.clone()),
            ),
            ("lease_epoch".to_string(), Value::Int64(held.lease_epoch)),
        ]),
    );
}

#[cfg(context_graph_internal_test_support)]
type FinalizationPause = Box<dyn FnOnce(&Store)>;

#[cfg(context_graph_internal_test_support)]
thread_local! {
    static FINALIZATION_PAUSE: std::cell::RefCell<Option<FinalizationPause>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `pause` once inside the next event-search head-switch transaction on this thread, after
/// the Context revision is read and guarded and the live event count is checked, before commit.
#[cfg(context_graph_internal_test_support)]
#[doc(hidden)]
pub fn pause_next_event_search_finalization_for_test(pause: impl FnOnce(&Store) + 'static) {
    FINALIZATION_PAUSE.with(|slot| *slot.borrow_mut() = Some(Box::new(pause)));
}
const SOURCE_MEMBER_COLUMNS: &str = "membership_id, ordinal, event_id, connector_id, source_instance_ref, session_ref, source_sequence, effective_interpretation_generation_id, captured_source_sha256, effective_source_sha256";

#[derive(Clone, Copy)]
struct Bounds {
    events: usize,
    passages: usize,
    bytes: usize,
    lease_ms: u64,
}

#[derive(Clone)]
struct InterpretationBinding {
    provenance: InterpretationProvenance,
}

#[derive(Clone, Copy)]
enum InterpretationProvenance {
    CapturedBase,
    Active(InterpretationGenerationId),
    ExplicitStaged(InterpretationGenerationId),
}

#[derive(Clone, Copy)]
enum InterpretationSelection {
    Ordinary,
    ExplicitStaged(InterpretationGenerationId),
}

impl InterpretationBinding {
    fn generation_id(&self) -> Option<InterpretationGenerationId> {
        match self.provenance {
            InterpretationProvenance::CapturedBase => None,
            InterpretationProvenance::Active(id) | InterpretationProvenance::ExplicitStaged(id) => {
                Some(id)
            }
        }
    }
}

#[derive(Clone)]
struct Generation {
    id: EventSearchGenerationId,
    context_id: crate::ContextId,
    bindings: BTreeMap<String, InterpretationBinding>,
    snapshot_sha256: String,
    event_count: u64,
    ordered_source_set_sha256: String,
    source_content_sha256: String,
    target: SourceSnapshot,
    started_at: TxId,
    pending_dual_head_activation: bool,
    mode: GenerationMode,
}

#[derive(Clone)]
struct ClaimedEvent {
    membership_id: uuid::Uuid,
    ordinal: u64,
    event: ConversationEvent,
    effective_interpretation_generation_id: Option<InterpretationGenerationId>,
    captured_source_sha256: String,
    effective_source_sha256: String,
}

struct CapturedRawEvent {
    event: ConversationEvent,
    event_columns: Vec<String>,
    event_row: Vec<Value>,
    anchor_columns: Vec<String>,
    anchor_rows: Vec<Vec<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceSnapshot {
    event_count: u64,
    ordered_source_set_sha256: String,
    source_content_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationMode {
    Building,
    ExtendReady,
}

struct Claim {
    generation: Generation,
    owner_token: String,
    lease_epoch: i64,
    expires_at: i64,
    connector_id: String,
    source_instance_ref: String,
    session_ref: String,
    start_sequence: u64,
    end_sequence: u64,
    cursor_sha256: String,
    passage_ordinal: u64,
    prior_committed_sequence: u64,
    prior_open_tail_start_sequence: Option<u64>,
    /// The cursor's recorded open-tail passage exactly as read, for the commit-time cursor check.
    prior_open_tail_passage_id: Option<uuid::Uuid>,
    /// That passage when this claim reopens its tail: the only passage the claim may retire.
    reopened_tail_passage_id: Option<uuid::Uuid>,
    open_tail_events: Vec<ClaimedEvent>,
    superseded_tail: Vec<SupersededPassage>,
    session_ended: bool,
    tail_can_reopen: bool,
    events: Vec<ClaimedEvent>,
}

struct SupersededPassage {
    id: EventSearchPassageId,
    conditions: Vec<(String, Value)>,
}

#[derive(Clone)]
struct RenderUnit {
    event: ClaimedEvent,
    prefix: String,
    body: String,
}

struct BuiltPassage {
    id: EventSearchPassageId,
    embedding_id: EmbeddingId,
    ordinal: u64,
    first_sequence: u64,
    last_sequence: u64,
    text: String,
    content_sha256: String,
    vector: Vec<f32>,
    dimension: u32,
    producer: crate::EvidenceProducer,
    sources: Vec<PassageEventSource>,
}

struct BuiltBatch {
    claim: Claim,
    passages: Vec<BuiltPassage>,
    eligible_events: u64,
    excluded_events: u64,
    next_open_tail_start_sequence: Option<u64>,
}

struct PendingStream {
    start_ordinal: u64,
    connector_id: String,
    source_instance_ref: String,
    session_ref: String,
    committed_sequence: u64,
    open_tail_start_sequence: Option<u64>,
    open_tail_passage_id: Option<uuid::Uuid>,
}

enum ClaimResult {
    Work(Box<Claim>),
    Complete,
}

enum FinishResult {
    Continue(Generation),
    Rebuild,
    Ready(Box<EventSearchBuildReceipt>),
}

pub(crate) fn catch_up_standalone(
    request: EventSearchRunRequest,
) -> Result<EventSearchBuildReceipt> {
    let context_ids = request.contexts.context_ids();
    if context_ids.len() != 1 {
        return Err(CgError::Engine(
            "one event-search build receipt can describe exactly one Context".into(),
        ));
    }
    let context_id = context_ids[0];
    let bounds = bounds(
        request.max_events_per_batch.get(),
        request.max_passages_per_batch.get(),
        request.max_bytes_per_batch.get(),
        request.lease_duration_ms.get(),
    );
    let mut generation = {
        let store = open_for_run(&request.store, &request.embedder)?;
        let generation = ensure_generation(
            &store,
            context_id,
            &request.embedder,
            &request.profile,
            bounds,
            false,
            InterpretationSelection::Ordinary,
        )?;
        drop(store);
        generation
    };

    let mut retries = 0_u64;
    loop {
        let claim = {
            let store = open_for_run(&request.store, &request.embedder)?;
            let claim = claim_batch(&store, &generation, bounds)?;
            drop(store);
            claim
        };
        let ClaimResult::Work(claim) = claim else {
            let store = open_for_run(&request.store, &request.embedder)?;
            let outcome = finish_or_continue(
                &store,
                &generation,
                &request.embedder,
                &request.profile,
                bounds,
                InterpretationSelection::Ordinary,
                retries,
            );
            let outcome = match outcome {
                Ok(FinishResult::Rebuild) => ensure_generation(
                    &store,
                    context_id,
                    &request.embedder,
                    &request.profile,
                    bounds,
                    true,
                    InterpretationSelection::Ordinary,
                )
                .map(FinishResult::Continue),
                Ok(FinishResult::Ready(receipt)) => {
                    maintain_passage_vectors(&store).map(|()| FinishResult::Ready(receipt))
                }
                other => other,
            };
            drop(store);
            match outcome? {
                FinishResult::Continue(next) => {
                    generation = next;
                    continue;
                }
                FinishResult::Ready(receipt) => return Ok(*receipt),
                FinishResult::Rebuild => unreachable!("rebuild handled while Store was open"),
            }
        };

        // From here the run holds a claim, so every way out of it releases that claim first.
        let held = HeldClaim::of(&claim);
        // No Store exists in this scope: renderer/tokenizer/embedder execution is deliberately
        // between the claim close and the validated commit reopen.
        let built = match render_and_embed(*claim, &request.embedder, &request.profile, bounds) {
            Ok(built) => built,
            Err(error) => {
                release_claim(&request.store, &request.embedder, &held);
                return Err(error);
            }
        };
        let committed = match open_for_run(&request.store, &request.embedder) {
            Ok(store) => {
                let committed =
                    commit_batch(&store, &request.embedder, &request.profile, bounds, built);
                drop(store);
                committed
            }
            Err(error) => {
                release_claim(&request.store, &request.embedder, &held);
                return Err(error);
            }
        };
        if let Err(error) = committed {
            // A stale claim is already gone; this run has nothing left to release and goes round
            // again to take a fresh one.
            if retries < MAX_STALE_RETRIES as u64 && is_stale_claim(&error) {
                retries += 1;
                continue;
            }
            release_claim(&request.store, &request.embedder, &held);
            return Err(error);
        }
    }
}

pub(crate) fn catch_up_store(
    store: &Store,
    request: EventSearchBuildRequest,
) -> Result<EventSearchBuildReceipt> {
    let bounds = bounds(
        request.max_events_per_batch.get(),
        request.max_passages_per_batch.get(),
        request.max_bytes_per_batch.get(),
        request.lease_duration_ms.get(),
    );
    run_with_store(
        store,
        request.context_id,
        request.embedder,
        request.profile,
        bounds,
        false,
        InterpretationSelection::Ordinary,
    )
}

pub(crate) fn rebuild_store(
    store: &Store,
    request: EventSearchRebuildRequest,
) -> Result<EventSearchBuildReceipt> {
    let bounds = bounds(
        request.max_events_per_batch.get(),
        request.max_passages_per_batch.get(),
        request.max_bytes_per_batch.get(),
        request.lease_duration_ms.get(),
    );
    let selection = request
        .interpretation_generation_id
        .map(InterpretationSelection::ExplicitStaged)
        .unwrap_or(InterpretationSelection::Ordinary);
    run_with_store(
        store,
        request.context_id,
        request.embedder,
        request.profile,
        bounds,
        true,
        selection,
    )
}

fn run_with_store(
    store: &Store,
    context_id: crate::ContextId,
    embedder: crate::EmbedderRegistration,
    profile: EventSearchProfile,
    bounds: Bounds,
    force_rebuild: bool,
    selection: InterpretationSelection,
) -> Result<EventSearchBuildReceipt> {
    let mut generation = ensure_generation(
        store,
        context_id,
        &embedder,
        &profile,
        bounds,
        force_rebuild,
        selection,
    )?;
    let mut retries = 0_u64;
    loop {
        let ClaimResult::Work(claim) = claim_batch(store, &generation, bounds)? else {
            match finish_or_continue(
                store,
                &generation,
                &embedder,
                &profile,
                bounds,
                selection,
                retries,
            )? {
                FinishResult::Continue(next) => {
                    generation = next;
                    continue;
                }
                FinishResult::Rebuild => {
                    generation = ensure_generation(
                        store, context_id, &embedder, &profile, bounds, true, selection,
                    )?;
                    continue;
                }
                FinishResult::Ready(receipt) => {
                    maintain_passage_vectors(store)?;
                    return Ok(*receipt);
                }
            }
        };
        // The caller necessarily retains `store`; this wrapper makes no Store-release claim.
        let built = render_and_embed(*claim, &embedder, &profile, bounds)?;
        match commit_batch(store, &embedder, &profile, bounds, built) {
            Ok(()) => {}
            Err(error) if retries < MAX_STALE_RETRIES as u64 && is_stale_claim(&error) => {
                retries += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Leave every passage vector partition with a complete maintained graph before a projection
/// reports ready. ContextDB builds and repairs vector graphs only through maintenance, never inside
/// a write or a query, and persists them with the store; so a find after this, including the first
/// find of a fresh process, searches a built graph instead of falling back to exact work or refusing
/// for want of one. Each maintenance cycle builds every partition still missing its graph; a cycle
/// that builds nothing while partitions remain is reported rather than retried.
fn maintain_passage_vectors(store: &Store) -> Result<()> {
    loop {
        let vector = store
            .db
            .run_maintenance_cycle()
            .map_err(map_engine_err)?
            .vector;
        if vector.remaining_partitions == 0 {
            return Ok(());
        }
        if let Some(failure) = vector.first_failure {
            return Err(CgError::Engine(format!(
                "the search index for {} passage partitions could not be built: {failure:?}",
                vector.remaining_partitions
            )));
        }
        if vector.built_partitions == 0 {
            return Err(CgError::Engine(format!(
                "the search index for {} passage partitions was not built by maintenance",
                vector.remaining_partitions
            )));
        }
    }
}

/// Test seam: run the maintenance a projection runs before it reports ready, for passage rows a
/// test or benchmark wrote directly.
#[cfg(context_graph_internal_test_support)]
impl Store {
    #[doc(hidden)]
    pub fn maintain_event_search_vectors_for_test(&self) -> Result<()> {
        maintain_passage_vectors(self)
    }
}

fn bounds(events: u32, passages: u32, bytes: usize, lease_ms: u64) -> Bounds {
    let ceiling = crate::owner_control::context_graph_read_limits();
    Bounds {
        events: usize::try_from(events)
            .unwrap_or(usize::MAX)
            .min(usize::try_from(ceiling.cursor_page_rows).unwrap_or(usize::MAX)),
        passages: usize::try_from(passages).unwrap_or(usize::MAX),
        bytes: bytes.min(usize::try_from(ceiling.cursor_page_bytes).unwrap_or(usize::MAX)),
        lease_ms,
    }
}

fn ensure_generation(
    store: &Store,
    context_id: crate::ContextId,
    embedder: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    force_rebuild: bool,
    selection: InterpretationSelection,
) -> Result<Generation> {
    ensure_event_context_revision(store, context_id)?;
    admit_projection_embedding_space(store, embedder)?;
    let mut connectors = event_connectors(store, context_id, bounds)?;
    let explicit_connector = match selection {
        InterpretationSelection::Ordinary => None,
        InterpretationSelection::ExplicitStaged(id) => {
            let connector = explicit_staged_connector(store, context_id, id, bounds)?;
            connectors.insert(connector.clone());
            Some(connector)
        }
    };
    let bindings = interpretation_bindings(
        store,
        context_id,
        &connectors,
        selection,
        explicit_connector.as_deref(),
        bounds,
    )?;
    let snapshot = snapshot_json(&bindings);
    let snapshot_sha256 = hash_json(&snapshot);
    if !force_rebuild {
        if matches!(selection, InterpretationSelection::Ordinary)
            && let Some(mut existing) = current_ready_head(
                store,
                context_id,
                &snapshot,
                &snapshot_sha256,
                embedder,
                profile,
                bounds,
            )?
        {
            existing.target =
                capture_source_membership(store, context_id, &bindings, Some(existing.id), bounds)?;
            return Ok(existing);
        }
        if let Some((mut existing, was_capturing)) = incomplete_generation(
            store,
            context_id,
            &snapshot,
            &snapshot_sha256,
            embedder,
            profile,
            bounds,
        )? {
            let captured =
                capture_source_membership(store, context_id, &bindings, Some(existing.id), bounds)?;
            if was_capturing {
                finalize_capture_attempt(store, context_id, existing.id, &captured)?;
                existing.event_count = captured.event_count;
                existing.ordered_source_set_sha256 = captured.ordered_source_set_sha256.clone();
                existing.source_content_sha256 = captured.source_content_sha256.clone();
            }
            existing.target = captured;
            return Ok(existing);
        }
        let observed = capture_source_membership(store, context_id, &bindings, None, bounds)?;
        if let Some(existing) = matching_generation(
            store,
            context_id,
            &snapshot_sha256,
            &observed,
            embedder,
            profile,
            bounds,
        )? {
            return Ok(existing);
        }
    }

    let id = EventSearchGenerationId::new_v7();
    let started_at = insert_capture_attempt(
        store,
        context_id,
        id,
        &bindings,
        &snapshot,
        &snapshot_sha256,
        embedder,
        profile,
    )?;
    let captured = capture_source_membership(store, context_id, &bindings, Some(id), bounds)?;
    finalize_capture_attempt(store, context_id, id, &captured)?;
    Ok(Generation {
        id,
        context_id,
        bindings,
        snapshot_sha256,
        event_count: captured.event_count,
        ordered_source_set_sha256: captured.ordered_source_set_sha256.clone(),
        source_content_sha256: captured.source_content_sha256.clone(),
        target: captured,
        started_at,
        pending_dual_head_activation: false,
        mode: GenerationMode::Building,
    }
    .with_activation_state())
}

fn ensure_event_context_revision(store: &Store, context_id: crate::ContextId) -> Result<()> {
    for attempt in 0..=MAX_STALE_RETRIES {
        let tx = store.begin_tx()?;
        let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
        let existing = match store.internal_db().execute_in_tx(
            tx,
            "SELECT revision_tx FROM event_context_revisions WHERE context_id = $context_id",
            &params,
        ) {
            Ok(existing) => existing,
            Err(error) => {
                let _ = store.internal_db().rollback(tx);
                return Err(map_engine_err(error));
            }
        };
        if let Some(row) = existing.rows.first() {
            if existing.rows.len() != 1 {
                let _ = store.internal_db().rollback(tx);
                return Err(CgError::Engine(
                    "the event Context revision primary key returned more than one row".into(),
                ));
            }
            if let Err(error) = get_txid(row, &existing.columns, "revision_tx") {
                let _ = store.internal_db().rollback(tx);
                return Err(error);
            }
            store.internal_db().rollback(tx).map_err(map_engine_err)?;
            return Ok(());
        }
        let inserted = store.internal_db().execute_in_tx(
            tx,
            "INSERT INTO event_context_revisions (context_id, revision_tx) VALUES ($context_id, $revision_tx)",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("revision_tx".into(), Value::TxId(tx)),
            ]),
        );
        if let Err(error) = inserted {
            let _ = store.internal_db().rollback(tx);
            if is_event_revision_initialization_race(&error) && attempt < MAX_STALE_RETRIES {
                continue;
            }
            return Err(map_engine_err(error));
        }
        match store.internal_db().commit(tx) {
            Ok(()) => return Ok(()),
            Err(error)
                if is_event_revision_initialization_race(&error) && attempt < MAX_STALE_RETRIES =>
            {
                let _ = store.internal_db().rollback(tx);
            }
            Err(error) => {
                let _ = store.internal_db().rollback(tx);
                return Err(map_engine_err(error));
            }
        }
    }
    Err(CgError::Engine(
        "event-search Context revision initialization did not converge".into(),
    ))
}

fn is_event_revision_initialization_race(error: &EngineError) -> bool {
    matches!(
        error,
        EngineError::UniqueViolation { table, column }
            if table == "event_context_revisions" && column == "context_id"
    ) || matches!(error, EngineError::ConditionalUpdateConflict { .. })
}

#[allow(clippy::too_many_arguments)]
fn insert_capture_attempt(
    store: &Store,
    context_id: crate::ContextId,
    id: EventSearchGenerationId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    snapshot: &JsonValue,
    snapshot_sha256: &str,
    embedder: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
) -> Result<TxId> {
    let interpretation_id = bindings
        .values()
        .filter_map(InterpretationBinding::generation_id)
        .next()
        .filter(|_| {
            bindings
                .values()
                .filter_map(InterpretationBinding::generation_id)
                .count()
                == 1
        });
    let tx = store.begin_tx()?;
    let placeholder = format!("capturing:{}", id.as_uuid());
    let result = store.internal_db().execute_in_tx(
        tx,
        "INSERT INTO event_search_generations (id, context_id, interpretation_generation_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, started_at, completed_at, sanitized_failure) VALUES ($id, $context_id, $interpretation_generation_id, $snapshot, $snapshot_sha256, $zero, $ordered_sha256, $source_sha256, $zero, $renderer_id, $renderer_version, $profile, $space, $encoding, $state, $zero, $zero, $zero, $zero, $zero, $zero, $started_at, $null, $null)",
        &HashMap::from([
            ("id".into(), Value::Uuid(id.as_uuid())),
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            (
                "interpretation_generation_id".into(),
                interpretation_id
                    .map(|id| Value::Uuid(id.as_uuid()))
                    .unwrap_or(Value::Null),
            ),
            ("snapshot".into(), Value::Json(snapshot.clone())),
            (
                "snapshot_sha256".into(),
                Value::Text(snapshot_sha256.to_owned()),
            ),
            ("ordered_sha256".into(), Value::Text(placeholder.clone())),
            ("source_sha256".into(), Value::Text(placeholder)),
            (
                "renderer_id".into(),
                Value::Text(EVENT_SEARCH_RENDERER_ID.into()),
            ),
            (
                "renderer_version".into(),
                Value::Text(EVENT_SEARCH_RENDERER_VERSION.into()),
            ),
            (
                "profile".into(),
                Value::Json(serde_json::to_value(profile).map_err(json_error)?),
            ),
            (
                "space".into(),
                Value::Text(embedder.space.embedding_space_id.clone()),
            ),
            (
                "encoding".into(),
                Value::Text(embedder.embedder.encoding_profile_id()),
            ),
            ("state".into(), Value::Text("capturing".into())),
            ("zero".into(), Value::Int64(0)),
            ("started_at".into(), Value::TxId(tx)),
            ("null".into(), Value::Null),
        ]),
    );
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(map_engine_err(error));
    }
    store.internal_db().commit(tx).map_err(map_engine_err)?;
    Ok(tx)
}

fn finalize_capture_attempt(
    store: &Store,
    context_id: crate::ContextId,
    id: EventSearchGenerationId,
    snapshot: &SourceSnapshot,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        let generation = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT context_id, capture_ordinal, state FROM event_search_generations WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        let row = generation
            .rows
            .first()
            .ok_or_else(|| stale_claim("event-search capture attempt disappeared"))?;
        if generation.rows.len() != 1
            || get_u64(row, &generation.columns, "capture_ordinal")? != snapshot.event_count
            || get_text(row, &generation.columns, "state")? != "capturing"
        {
            return Err(stale_claim(
                "event-search capture membership did not seal exactly",
            ));
        }
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_generations SET event_count = $event_count, ordered_source_set_sha256 = $ordered_sha256, source_content_sha256 = $source_sha256, state = $state WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    (
                        "event_count".into(),
                        integer(snapshot.event_count, "event count")?,
                    ),
                    (
                        "ordered_sha256".into(),
                        Value::Text(snapshot.ordered_source_set_sha256.clone()),
                    ),
                    (
                        "source_sha256".into(),
                        Value::Text(snapshot.source_content_sha256.clone()),
                    ),
                    ("state".into(), Value::Text("building".into())),
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)
}

fn admit_projection_embedding_space(
    store: &Store,
    registration: &crate::EmbedderRegistration,
) -> Result<()> {
    let expected_binding = format!(
        "{}:event_search_passages:vector_text",
        registration.space.embedding_space_id
    );
    let mut actual_bindings = registration
        .space
        .bindings
        .iter()
        .map(|binding| {
            format!(
                "{}:{}:{}",
                binding.embedding_space_id, binding.table_name, binding.vector_column
            )
        })
        .collect::<Vec<_>>();
    actual_bindings.sort();
    if actual_bindings != [expected_binding.clone()] {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "bindings",
            expected: expected_binding,
            actual: actual_bindings.join(","),
        });
    }
    if registration.space.default_for_evidence_kind.is_some() {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "default_for_evidence_kind",
            expected: String::new(),
            actual: format!("{:?}", registration.space.default_for_evidence_kind),
        });
    }
    let configured = store
        .schema_embedding_spaces
        .iter()
        .find(|space| space.embedding_space_id == registration.space.embedding_space_id)
        .ok_or_else(|| CgError::EmbeddingSpaceConfigMismatch {
            field: "bindings",
            expected: String::new(),
            actual: expected_binding,
        })?;
    super::validate_embedding_space_decl_matches(configured, &registration.space)?;
    let embedder_declared = registration.embedder.embedding_space();
    super::validate_embedding_space_decl_matches(&registration.space, &embedder_declared)
}

impl Generation {
    fn with_activation_state(mut self) -> Self {
        self.pending_dual_head_activation = self.bindings.values().any(|binding| {
            matches!(
                binding.provenance,
                InterpretationProvenance::ExplicitStaged(_)
            )
        });
        self
    }

    fn expected_state(&self) -> &'static str {
        match self.mode {
            GenerationMode::Building => "building",
            GenerationMode::ExtendReady => "ready",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn matching_generation(
    store: &Store,
    context_id: crate::ContextId,
    snapshot_sha256: &str,
    snapshot: &SourceSnapshot,
    embedder: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
) -> Result<Option<Generation>> {
    let result = query_rows(
        store,
        bounds,
        "SELECT id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, state, started_at FROM event_search_generations WHERE context_id = $context_id AND interpretation_snapshot_sha256 = $snapshot_sha256 AND event_count = $event_count AND ordered_source_set_sha256 = $ordered_sha256 AND source_content_sha256 = $source_sha256 AND renderer_id = $renderer_id AND renderer_version = $renderer_version AND renderer_profile = $profile AND embedding_space_id = $space AND encoding_profile = $encoding AND state = $building ORDER BY started_at DESC, id DESC LIMIT 1",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            (
                "snapshot_sha256".into(),
                Value::Text(snapshot_sha256.into()),
            ),
            (
                "event_count".into(),
                integer(snapshot.event_count, "event count")?,
            ),
            (
                "ordered_sha256".into(),
                Value::Text(snapshot.ordered_source_set_sha256.clone()),
            ),
            (
                "source_sha256".into(),
                Value::Text(snapshot.source_content_sha256.clone()),
            ),
            (
                "renderer_id".into(),
                Value::Text(EVENT_SEARCH_RENDERER_ID.into()),
            ),
            (
                "renderer_version".into(),
                Value::Text(EVENT_SEARCH_RENDERER_VERSION.into()),
            ),
            (
                "profile".into(),
                Value::Json(serde_json::to_value(profile).map_err(json_error)?),
            ),
            (
                "space".into(),
                Value::Text(embedder.space.embedding_space_id.clone()),
            ),
            (
                "encoding".into(),
                Value::Text(embedder.embedder.encoding_profile_id()),
            ),
            ("building".into(), Value::Text("building".into())),
        ]),
    )?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    let state = get_text(row, &result.columns, "state")?;
    if state != "building" {
        return Ok(None);
    }
    if get_u64(row, &result.columns, "capture_ordinal")? != snapshot.event_count
        || get_u64(row, &result.columns, "event_count")? != snapshot.event_count
        || get_text(row, &result.columns, "ordered_source_set_sha256")?
            != snapshot.ordered_source_set_sha256
        || get_text(row, &result.columns, "source_content_sha256")?
            != snapshot.source_content_sha256
    {
        return Ok(None);
    }
    let persisted_snapshot = get_json(row, &result.columns, "interpretation_snapshot")?;
    if hash_json(&persisted_snapshot) != snapshot_sha256 {
        return Err(stale_claim(
            "event-search interpretation snapshot hash does not match its persisted snapshot",
        ));
    }
    let bindings = bindings_from_snapshot(&persisted_snapshot)?;
    Ok(Some(
        Generation {
            id: EventSearchGenerationId::from(get_uuid(row, &result.columns, "id")?),
            context_id,
            bindings,
            snapshot_sha256: snapshot_sha256.into(),
            event_count: snapshot.event_count,
            ordered_source_set_sha256: snapshot.ordered_source_set_sha256.clone(),
            source_content_sha256: snapshot.source_content_sha256.clone(),
            target: snapshot.clone(),
            started_at: get_txid(row, &result.columns, "started_at")?,
            pending_dual_head_activation: false,
            mode: GenerationMode::Building,
        }
        .with_activation_state(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn current_ready_head(
    store: &Store,
    context_id: crate::ContextId,
    snapshot: &JsonValue,
    snapshot_sha256: &str,
    embedder: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
) -> Result<Option<Generation>> {
    let head = query_rows(
        store,
        bounds,
        "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
    )?;
    let Some(head_row) = head.rows.first() else {
        return Ok(None);
    };
    if head.rows.len() != 1 {
        return Err(stale_claim("event-search current head is not unique"));
    }
    let id = EventSearchGenerationId::from(get_uuid(head_row, &head.columns, "generation_id")?);
    let result = query_rows(
        store,
        bounds,
        "SELECT context_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, started_at FROM event_search_generations WHERE id = $id",
        &HashMap::from([("id".into(), Value::Uuid(id.as_uuid()))]),
    )?;
    let Some(row) = result.rows.first() else {
        return Err(stale_claim(
            "event-search current head generation disappeared",
        ));
    };
    if result.rows.len() != 1 {
        return Err(stale_claim(
            "event-search current head generation is not unique",
        ));
    }
    let persisted_snapshot = get_json(row, &result.columns, "interpretation_snapshot")?;
    let event_count = get_u64(row, &result.columns, "event_count")?;
    let ordered_source_set_sha256 = get_text(row, &result.columns, "ordered_source_set_sha256")?;
    let source_content_sha256 = get_text(row, &result.columns, "source_content_sha256")?;
    if get_uuid(row, &result.columns, "context_id")? != context_id.as_uuid()
        || get_text(row, &result.columns, "state")? != "ready"
        || get_text(row, &result.columns, "interpretation_snapshot_sha256")? != snapshot_sha256
        || persisted_snapshot != *snapshot
        || hash_json(&persisted_snapshot) != snapshot_sha256
        || get_text(row, &result.columns, "renderer_id")? != EVENT_SEARCH_RENDERER_ID
        || get_text(row, &result.columns, "renderer_version")? != EVENT_SEARCH_RENDERER_VERSION
        || get_json(row, &result.columns, "renderer_profile")?
            != serde_json::to_value(profile).map_err(json_error)?
        || get_text(row, &result.columns, "embedding_space_id")?
            != embedder.space.embedding_space_id
        || get_text(row, &result.columns, "encoding_profile")?
            != embedder.embedder.encoding_profile_id()
    {
        return Ok(None);
    }
    if get_u64(row, &result.columns, "capture_ordinal")? < event_count {
        return Err(stale_claim(
            "event-search ready head capture ordinal trails its searchable watermark",
        ));
    }
    Ok(Some(
        Generation {
            id,
            context_id,
            bindings: bindings_from_snapshot(&persisted_snapshot)?,
            snapshot_sha256: snapshot_sha256.into(),
            event_count,
            ordered_source_set_sha256: ordered_source_set_sha256.clone(),
            source_content_sha256: source_content_sha256.clone(),
            target: SourceSnapshot {
                event_count,
                ordered_source_set_sha256,
                source_content_sha256,
            },
            started_at: get_txid(row, &result.columns, "started_at")?,
            pending_dual_head_activation: false,
            mode: GenerationMode::ExtendReady,
        }
        .with_activation_state(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn incomplete_generation(
    store: &Store,
    context_id: crate::ContextId,
    snapshot: &JsonValue,
    snapshot_sha256: &str,
    embedder: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
) -> Result<Option<(Generation, bool)>> {
    let result = query_rows(
        store,
        bounds,
        "SELECT id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, started_at FROM event_search_generations WHERE context_id = $context_id AND interpretation_snapshot_sha256 = $snapshot_sha256 AND renderer_id = $renderer_id AND renderer_version = $renderer_version AND renderer_profile = $profile AND embedding_space_id = $space AND encoding_profile = $encoding AND state != $ready AND state != $failed AND state != $retired ORDER BY started_at DESC, id DESC LIMIT 1",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            (
                "snapshot_sha256".into(),
                Value::Text(snapshot_sha256.into()),
            ),
            (
                "renderer_id".into(),
                Value::Text(EVENT_SEARCH_RENDERER_ID.into()),
            ),
            (
                "renderer_version".into(),
                Value::Text(EVENT_SEARCH_RENDERER_VERSION.into()),
            ),
            (
                "profile".into(),
                Value::Json(serde_json::to_value(profile).map_err(json_error)?),
            ),
            (
                "space".into(),
                Value::Text(embedder.space.embedding_space_id.clone()),
            ),
            (
                "encoding".into(),
                Value::Text(embedder.embedder.encoding_profile_id()),
            ),
            ("ready".into(), Value::Text("ready".into())),
            ("failed".into(), Value::Text("failed".into())),
            ("retired".into(), Value::Text("retired".into())),
        ]),
    )?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    let state = get_text(row, &result.columns, "state")?;
    if state != "capturing" && state != "building" {
        return Ok(None);
    }
    let persisted_snapshot = get_json(row, &result.columns, "interpretation_snapshot")?;
    if persisted_snapshot != *snapshot || hash_json(&persisted_snapshot) != snapshot_sha256 {
        return Err(stale_claim(
            "incomplete event-search generation snapshot changed under its identity",
        ));
    }
    let event_count = get_u64(row, &result.columns, "event_count")?;
    let ordered_source_set_sha256 = get_text(row, &result.columns, "ordered_source_set_sha256")?;
    let source_content_sha256 = get_text(row, &result.columns, "source_content_sha256")?;
    Ok(Some((
        Generation {
            id: EventSearchGenerationId::from(get_uuid(row, &result.columns, "id")?),
            context_id,
            bindings: bindings_from_snapshot(&persisted_snapshot)?,
            snapshot_sha256: snapshot_sha256.into(),
            event_count,
            ordered_source_set_sha256: ordered_source_set_sha256.clone(),
            source_content_sha256: source_content_sha256.clone(),
            target: SourceSnapshot {
                event_count,
                ordered_source_set_sha256,
                source_content_sha256,
            },
            started_at: get_txid(row, &result.columns, "started_at")?,
            pending_dual_head_activation: false,
            mode: GenerationMode::Building,
        }
        .with_activation_state(),
        state == "capturing",
    )))
}

fn event_connectors(
    store: &Store,
    context_id: crate::ContextId,
    bounds: Bounds,
) -> Result<BTreeSet<String>> {
    let mut connectors = BTreeSet::new();
    stream_rows(
        store,
        bounds,
        "SELECT connector_id FROM events WHERE context_id = $context_id ORDER BY connector_id ASC, source_instance_ref ASC, seq ASC",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        |columns, row| {
            connectors.insert(get_text(row, columns, "connector_id")?);
            Ok(())
        },
    )?;
    Ok(connectors)
}

fn interpretation_bindings(
    store: &Store,
    context_id: crate::ContextId,
    connectors: &BTreeSet<String>,
    selection: InterpretationSelection,
    explicit_connector: Option<&str>,
    bounds: Bounds,
) -> Result<BTreeMap<String, InterpretationBinding>> {
    let mut bindings = BTreeMap::new();
    for connector in connectors {
        let provenance = match (selection, explicit_connector == Some(connector.as_str())) {
            (InterpretationSelection::ExplicitStaged(id), true) => {
                validate_ready_interpretation(store, context_id, connector, id, bounds)?;
                InterpretationProvenance::ExplicitStaged(id)
            }
            _ => resolve_search_binding(store, context_id, connector, bounds)?,
        };
        bindings.insert(connector.clone(), InterpretationBinding { provenance });
    }
    Ok(bindings)
}

fn explicit_staged_connector(
    store: &Store,
    context_id: crate::ContextId,
    id: InterpretationGenerationId,
    bounds: Bounds,
) -> Result<String> {
    let result = query_rows(
        store,
        bounds,
        "SELECT connector_id, event_count, capture_ordinal, build_ordinal, state FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("id".into(), Value::Uuid(id.as_uuid())),
        ]),
    )?;
    let row = result.rows.first().ok_or(CgError::NotFound {
        id: Some(id.to_string()),
        missing: Some("interpretation generation"),
    })?;
    if result.rows.len() != 1 {
        return Err(CgError::Engine(
            "explicit interpretation generation lookup was not unique".into(),
        ));
    }
    validate_ready_interpretation_row(row, &result.columns, id)?;
    get_text(row, &result.columns, "connector_id")
}

fn resolve_search_binding(
    store: &Store,
    context_id: crate::ContextId,
    connector: &str,
    bounds: Bounds,
) -> Result<InterpretationProvenance> {
    let head = query_rows(
        store,
        bounds,
        "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
        ]),
    )?;
    let Some(row) = head.rows.first() else {
        return Ok(InterpretationProvenance::CapturedBase);
    };
    if head.rows.len() != 1 {
        return Err(CgError::Engine(
            "active interpretation head lookup was not unique".into(),
        ));
    }
    let id = InterpretationGenerationId::from(get_uuid(row, &head.columns, "generation_id")?);
    validate_ready_interpretation(store, context_id, connector, id, bounds)?;
    Ok(InterpretationProvenance::Active(id))
}

fn validate_ready_interpretation(
    store: &Store,
    context_id: crate::ContextId,
    connector: &str,
    id: InterpretationGenerationId,
    bounds: Bounds,
) -> Result<()> {
    let result = query_rows(
        store,
        bounds,
        "SELECT connector_id, event_count, capture_ordinal, build_ordinal, state FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id AND id = $id",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
            ("id".into(), Value::Uuid(id.as_uuid())),
        ]),
    )?;
    let row = result.rows.first().ok_or(CgError::NotFound {
        id: Some(id.to_string()),
        missing: Some("ready interpretation generation"),
    })?;
    if result.rows.len() != 1 {
        return Err(CgError::Engine(
            "selected interpretation generation lookup was not unique".into(),
        ));
    }
    validate_ready_interpretation_row(row, &result.columns, id)
}

fn validate_ready_interpretation_row(
    row: &[Value],
    columns: &[String],
    id: InterpretationGenerationId,
) -> Result<()> {
    let event_count = get_u64(row, columns, "event_count")?;
    if get_text(row, columns, "state")? != "ready"
        || get_u64(row, columns, "capture_ordinal")? != event_count
        || get_u64(row, columns, "build_ordinal")? != event_count
    {
        return Err(CgError::Engine(format!(
            "interpretation generation {id} is not a completely sealed ready input"
        )));
    }
    Ok(())
}

fn snapshot_json(bindings: &BTreeMap<String, InterpretationBinding>) -> JsonValue {
    JsonValue::Object(
        bindings
            .iter()
            .map(|(connector, binding)| {
                let value = match binding.provenance {
                    InterpretationProvenance::CapturedBase => json!({"captured_base": true}),
                    InterpretationProvenance::Active(id) => {
                        json!({"generation_id": id.to_string(), "active": true})
                    }
                    InterpretationProvenance::ExplicitStaged(id) => {
                        json!({"generation_id": id.to_string(), "active": false})
                    }
                };
                (connector.clone(), value)
            })
            .collect::<JsonMap<_, _>>(),
    )
}

fn bindings_from_snapshot(snapshot: &JsonValue) -> Result<BTreeMap<String, InterpretationBinding>> {
    let object = snapshot
        .as_object()
        .ok_or_else(|| stale_claim("event-search interpretation snapshot is not an object"))?;
    object
        .iter()
        .map(|(connector, value)| {
            let value = value.as_object().ok_or_else(|| {
                stale_claim("event-search interpretation binding is not an object")
            })?;
            let provenance = if value.get("captured_base") == Some(&JsonValue::Bool(true)) {
                InterpretationProvenance::CapturedBase
            } else {
                let raw_id = value
                    .get("generation_id")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| stale_claim("event-search interpretation binding has no ID"))?;
                let uuid = uuid::Uuid::parse_str(raw_id).map_err(|_| {
                    stale_claim("event-search interpretation binding has an invalid ID")
                })?;
                let id = InterpretationGenerationId::from(uuid);
                match value.get("active").and_then(JsonValue::as_bool) {
                    Some(true) => InterpretationProvenance::Active(id),
                    Some(false) => InterpretationProvenance::ExplicitStaged(id),
                    None => {
                        return Err(stale_claim(
                            "event-search interpretation binding has no provenance",
                        ));
                    }
                }
            };
            Ok((connector.clone(), InterpretationBinding { provenance }))
        })
        .collect()
}

fn capture_source_membership(
    store: &Store,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    generation_id: Option<EventSearchGenerationId>,
    bounds: Bounds,
) -> Result<SourceSnapshot> {
    let mut next_ordinal = generation_id
        .map(|generation_id| {
            generation_capture_high_water(store, context_id, generation_id, bounds)
        })
        .transpose()?
        .unwrap_or(0);
    let session = store
        .internal_db()
        .read_session(read_limits(bounds))
        .map_err(map_engine_err)?;
    let mut cursor = session
        .open_cursor(
            &format!(
                "SELECT {CAPTURED_EVENT_COLUMNS} FROM events WHERE context_id = $context_id ORDER BY connector_id ASC, source_instance_ref ASC, seq ASC"
            ),
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    let mut count = 0_u64;
    let mut ordered = Sha256::new();
    ordered.update(b"context-graph/event-search-ordered-source-set/v1");
    let mut content = Sha256::new();
    content.update(b"context-graph/event-search-source-content/v1");
    let captured = (|| -> Result<()> {
        let first = cursor.first_page();
        let mut has_more = first.has_more;
        capture_search_page(
            store,
            &session,
            context_id,
            bindings,
            generation_id,
            bounds,
            &first.columns,
            &first.rows,
            &mut count,
            &mut next_ordinal,
            &mut ordered,
            &mut content,
        )?;
        while has_more {
            let page = cursor
                .fetch(NonZeroUsize::new(bounds.events.max(1)))
                .map_err(map_engine_err)?;
            has_more = page.has_more;
            capture_search_page(
                store,
                &session,
                context_id,
                bindings,
                generation_id,
                bounds,
                &page.columns,
                &page.rows,
                &mut count,
                &mut next_ordinal,
                &mut ordered,
                &mut content,
            )?;
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    captured?;
    closed?;
    let observed = SourceSnapshot {
        event_count: count,
        ordered_source_set_sha256: finish_hash(ordered),
        source_content_sha256: finish_hash(content),
    };
    if let Some(generation_id) = generation_id {
        let persisted = fold_source_membership(store, context_id, generation_id, bounds)?;
        let capture_high_water =
            generation_capture_high_water(store, context_id, generation_id, bounds)?;
        if persisted.event_count != observed.event_count
            || capture_high_water != persisted.event_count
        {
            return Err(stale_claim(
                "event-search persisted membership differs from the fixed source snapshot",
            ));
        }
        Ok(persisted)
    } else {
        Ok(observed)
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_search_page(
    store: &Store,
    session: &ReadSession,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    generation_id: Option<EventSearchGenerationId>,
    bounds: Bounds,
    columns: &[String],
    rows: &[Vec<Value>],
    count: &mut u64,
    next_ordinal: &mut u64,
    ordered: &mut Sha256,
    content: &mut Sha256,
) -> Result<()> {
    for row in rows {
        let raw = captured_raw_from_row(session, bounds, context_id, columns, row)?;
        let captured_sha256 = captured_source_sha256(
            &raw.event_columns,
            &raw.event_row,
            &raw.anchor_columns,
            &raw.anchor_rows,
        );
        let (event, effective_generation_id, effective_sha256) =
            effective_captured_event(session, bindings, raw.event, &captured_sha256)?;
        let candidate = ClaimedEvent {
            membership_id: uuid::Uuid::now_v7(),
            ordinal: generation_id.map_or(*count, |_| *next_ordinal),
            event,
            effective_interpretation_generation_id: effective_generation_id,
            captured_source_sha256: captured_sha256,
            effective_source_sha256: effective_sha256,
        };
        let member = if let Some(generation_id) = generation_id {
            let member = reuse_or_persist_search_membership(
                store,
                context_id,
                generation_id,
                &candidate,
                bounds,
            )?;
            *next_ordinal =
                (*next_ordinal).max(member.ordinal.checked_add(1).ok_or_else(|| {
                    CgError::Engine("event-search capture ordinal exhausted u64".into())
                })?);
            member
        } else {
            candidate
        };
        if generation_id.is_none() {
            hash_part(ordered, &member.ordinal.to_be_bytes());
            hash_part(ordered, member.event.id.as_uuid().as_bytes());
            hash_part(ordered, member.event.connector_id.as_bytes());
            hash_part(ordered, member.event.source_instance_ref.as_bytes());
            hash_part(ordered, member.event.session_ref.as_bytes());
            hash_part(ordered, &member.event.seq.to_be_bytes());
            hash_part(ordered, member.captured_source_sha256.as_bytes());
            hash_part(content, member.effective_source_sha256.as_bytes());
        }
        *count = count
            .checked_add(1)
            .ok_or_else(|| CgError::Engine("event-search source count exhausted u64".into()))?;
    }
    Ok(())
}

fn reuse_or_persist_search_membership(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    member: &ClaimedEvent,
    bounds: Bounds,
) -> Result<ClaimedEvent> {
    let result = query_rows(
        store,
        bounds,
        &format!(
            "SELECT context_id, generation_id, {SOURCE_MEMBER_COLUMNS} FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id"
        ),
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
            ("event_id".into(), Value::Uuid(member.event.id.as_uuid())),
        ]),
    )?;
    if let Some(row) = result.rows.first() {
        if result.rows.len() != 1
            || get_uuid(row, &result.columns, "context_id")? != context_id.as_uuid()
            || get_uuid(row, &result.columns, "generation_id")? != generation_id.as_uuid()
            || get_uuid(row, &result.columns, "event_id")? != member.event.id.as_uuid()
            || get_text(row, &result.columns, "connector_id")? != member.event.connector_id
            || get_text(row, &result.columns, "source_instance_ref")?
                != member.event.source_instance_ref
            || get_text(row, &result.columns, "session_ref")? != member.event.session_ref
            || get_u64(row, &result.columns, "source_sequence")? != member.event.seq
            || get_optional_interpretation_id(
                row,
                &result.columns,
                "effective_interpretation_generation_id",
            )? != member.effective_interpretation_generation_id
            || get_text(row, &result.columns, "captured_source_sha256")?
                != member.captured_source_sha256
            || get_text(row, &result.columns, "effective_source_sha256")?
                != member.effective_source_sha256
        {
            return Err(stale_claim(
                "retained event-search membership changed for its immutable event identity",
            ));
        }
        let mut retained = member.clone();
        retained.membership_id = get_uuid(row, &result.columns, "membership_id")?;
        retained.ordinal = get_u64(row, &result.columns, "ordinal")?;
        return Ok(retained);
    }
    persist_search_membership(store, context_id, generation_id, member)?;
    Ok(member.clone())
}

fn generation_capture_high_water(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    bounds: Bounds,
) -> Result<u64> {
    let result = query_rows(
        store,
        bounds,
        "SELECT context_id, capture_ordinal FROM event_search_generations WHERE context_id = $context_id AND id = $id",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("id".into(), Value::Uuid(generation_id.as_uuid())),
        ]),
    )?;
    let row = result
        .rows
        .first()
        .ok_or_else(|| stale_claim("event-search capture generation disappeared"))?;
    if result.rows.len() != 1
        || get_uuid(row, &result.columns, "context_id")? != context_id.as_uuid()
    {
        return Err(stale_claim(
            "event-search capture generation lookup was not unique",
        ));
    }
    get_u64(row, &result.columns, "capture_ordinal")
}

fn fold_source_membership(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    bounds: Bounds,
) -> Result<SourceSnapshot> {
    let mut count = 0_u64;
    let mut ordered = Sha256::new();
    ordered.update(b"context-graph/event-search-ordered-source-set/v1");
    let mut content = Sha256::new();
    content.update(b"context-graph/event-search-source-content/v1");
    stream_rows(
        store,
        bounds,
        &format!(
            "SELECT {SOURCE_MEMBER_COLUMNS} FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal ASC"
        ),
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
        ]),
        |columns, row| {
            let ordinal = get_u64(row, columns, "ordinal")?;
            if ordinal != count {
                return Err(stale_claim(
                    "event-search persisted membership has an ordinal gap",
                ));
            }
            hash_part(&mut ordered, &ordinal.to_be_bytes());
            hash_part(&mut ordered, get_uuid(row, columns, "event_id")?.as_bytes());
            hash_part(
                &mut ordered,
                get_text(row, columns, "connector_id")?.as_bytes(),
            );
            hash_part(
                &mut ordered,
                get_text(row, columns, "source_instance_ref")?.as_bytes(),
            );
            hash_part(
                &mut ordered,
                get_text(row, columns, "session_ref")?.as_bytes(),
            );
            hash_part(
                &mut ordered,
                &get_u64(row, columns, "source_sequence")?.to_be_bytes(),
            );
            hash_part(
                &mut ordered,
                get_text(row, columns, "captured_source_sha256")?.as_bytes(),
            );
            hash_part(
                &mut content,
                get_text(row, columns, "effective_source_sha256")?.as_bytes(),
            );
            count = count
                .checked_add(1)
                .ok_or_else(|| CgError::Engine("event-search source count exhausted u64".into()))?;
            Ok(())
        },
    )?;
    Ok(SourceSnapshot {
        event_count: count,
        ordered_source_set_sha256: finish_hash(ordered),
        source_content_sha256: finish_hash(content),
    })
}

fn captured_raw_from_row(
    session: &ReadSession,
    bounds: Bounds,
    context_id: crate::ContextId,
    columns: &[String],
    row: &[Value],
) -> Result<CapturedRawEvent> {
    let event = decode_event(columns, row)?;
    if event.context_id != context_id {
        return Err(stale_claim(
            "event-search source cursor crossed its Context boundary",
        ));
    }
    let mut cursor = session
        .open_cursor(
            &format!(
                "SELECT {CAPTURED_ANCHOR_COLUMNS} FROM event_anchors WHERE event_id = $event_id AND context_id = $context_id ORDER BY declaration_version ASC, key ASC, identity_value ASC"
            ),
            &HashMap::from([
                ("event_id".into(), Value::Uuid(event.id.as_uuid())),
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    let mut anchor_columns = Vec::new();
    let mut anchor_rows = Vec::new();
    let mut used = 0_usize;
    let read = (|| -> Result<()> {
        let first = cursor.first_page();
        anchor_columns = first.columns.clone();
        let mut has_more = first.has_more;
        append_captured_anchor_rows(
            bounds,
            &first.columns,
            &first.rows,
            &mut used,
            &mut anchor_rows,
        )?;
        while has_more {
            let page = cursor
                .fetch(NonZeroUsize::new(bounds.events.max(1)))
                .map_err(map_engine_err)?;
            has_more = page.has_more;
            append_captured_anchor_rows(
                bounds,
                &page.columns,
                &page.rows,
                &mut used,
                &mut anchor_rows,
            )?;
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    read?;
    closed?;
    Ok(CapturedRawEvent {
        event,
        event_columns: columns.to_vec(),
        event_row: row.to_vec(),
        anchor_columns,
        anchor_rows,
    })
}

fn append_captured_anchor_rows(
    bounds: Bounds,
    columns: &[String],
    rows: &[Vec<Value>],
    used: &mut usize,
    captured: &mut Vec<Vec<Value>>,
) -> Result<()> {
    for row in rows {
        let bytes = canonical_json_bytes(
            &serde_json::to_value((columns, row))
                .expect("persisted anchor values always serialize"),
        )
        .len();
        *used = used.saturating_add(bytes);
        if *used > bounds.bytes {
            return Err(CgError::Engine(format!(
                "captured event-search anchor set exceeds max_bytes_per_batch {}",
                bounds.bytes
            )));
        }
        captured.push(row.clone());
    }
    Ok(())
}

fn effective_captured_event(
    session: &ReadSession,
    bindings: &BTreeMap<String, InterpretationBinding>,
    mut event: ConversationEvent,
    captured_sha256: &str,
) -> Result<(
    ConversationEvent,
    Option<InterpretationGenerationId>,
    String,
)> {
    let Some(selected_id) = bindings
        .get(&event.connector_id)
        .and_then(InterpretationBinding::generation_id)
    else {
        let hash = effective_source_hash(&event, None, None);
        return Ok((event, None, hash));
    };
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(event.context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(selected_id.as_uuid())),
        ("event_id".into(), Value::Uuid(event.id.as_uuid())),
    ]);
    let membership = session
        .execute(
            "SELECT source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
            &params,
        )
        .map_err(map_engine_err)?;
    let Some(member) = membership.rows.first() else {
        let hash = effective_source_hash(&event, None, None);
        return Ok((event, None, hash));
    };
    if membership.rows.len() != 1
        || get_text(member, &membership.columns, "source_row_sha256")? != captured_sha256
    {
        return Err(stale_claim(
            "selected interpretation membership does not match the captured raw event",
        ));
    }
    let effective = session
        .execute(
            "SELECT captured_content_sha256, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, coverage, effective_sha256 FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
            &params,
        )
        .map_err(map_engine_err)?;
    let row = effective
        .rows
        .first()
        .ok_or_else(|| stale_claim("selected interpretation member has no effective value"))?;
    if effective.rows.len() != 1
        || get_text(row, &effective.columns, "captured_content_sha256")? != event.content_hash
    {
        return Err(stale_claim(
            "selected interpretation value does not match captured event content",
        ));
    }
    let effective_row_sha256 = get_text(row, &effective.columns, "effective_sha256")?;
    apply_effective_row(&mut event, &effective.columns, row)?;
    let hash = effective_source_hash(&event, Some(selected_id), Some(&effective_row_sha256));
    Ok((event, Some(selected_id), hash))
}

fn effective_source_hash(
    event: &ConversationEvent,
    generation_id: Option<InterpretationGenerationId>,
    effective_row_sha256: Option<&str>,
) -> String {
    hash_json(&json!([
        "context-graph/event-search-effective-input/v1",
        event,
        generation_id.map(|id| id.to_string()),
        effective_row_sha256,
    ]))
}

fn persist_search_membership(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    member: &ClaimedEvent,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        let generation = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT context_id, capture_ordinal, state FROM event_search_generations WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(generation_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        let row = generation
            .rows
            .first()
            .ok_or_else(|| stale_claim("event-search capture attempt disappeared"))?;
        let state = get_text(row, &generation.columns, "state")?;
        if generation.rows.len() != 1
            || get_u64(row, &generation.columns, "capture_ordinal")? != member.ordinal
            || (state != "capturing" && state != "building" && state != "ready")
        {
            return Err(stale_claim(
                "event-search capture ordinal changed before membership append",
            ));
        }
        if state == "ready" {
            let head = store
                .internal_db()
                .execute_in_tx(
                    tx,
                    "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
                    &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
                )
                .map_err(map_engine_err)?;
            if head.rows.len() != 1
                || get_uuid(&head.rows[0], &head.columns, "generation_id")?
                    != generation_id.as_uuid()
            {
                return Err(stale_claim(
                    "event-search ready generation stopped being the current head during capture",
                ));
            }
        }
        let generation_guard = exact_conditions(
            row,
            &generation.columns,
            &["context_id", "capture_ordinal", "state"],
        )?;
        if !store
            .internal_db()
            .guard_row_conditions_in_tx(
                tx,
                "event_search_generations",
                "id",
                &Value::Uuid(generation_id.as_uuid()),
                &generation_guard,
            )
            .map_err(map_engine_err)?
        {
            return Err(stale_claim(
                "event-search generation changed during membership append",
            ));
        }
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "INSERT INTO event_search_source_events (membership_id, context_id, generation_id, ordinal, event_id, connector_id, source_instance_ref, session_ref, source_sequence, effective_interpretation_generation_id, captured_source_sha256, effective_source_sha256) VALUES ($membership_id, $context_id, $generation_id, $ordinal, $event_id, $connector_id, $source, $session, $sequence, $effective_generation_id, $captured_sha256, $effective_sha256)",
                &HashMap::from([
                    ("membership_id".into(), Value::Uuid(member.membership_id)),
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                    ("ordinal".into(), integer(member.ordinal, "capture ordinal")?),
                    ("event_id".into(), Value::Uuid(member.event.id.as_uuid())),
                    ("connector_id".into(), Value::Text(member.event.connector_id.clone())),
                    ("source".into(), Value::Text(member.event.source_instance_ref.clone())),
                    ("session".into(), Value::Text(member.event.session_ref.clone())),
                    ("sequence".into(), integer(member.event.seq, "source sequence")?),
                    (
                        "effective_generation_id".into(),
                        member
                            .effective_interpretation_generation_id
                            .map(|id| Value::Uuid(id.as_uuid()))
                            .unwrap_or(Value::Null),
                    ),
                    (
                        "captured_sha256".into(),
                        Value::Text(member.captured_source_sha256.clone()),
                    ),
                    (
                        "effective_sha256".into(),
                        Value::Text(member.effective_source_sha256.clone()),
                    ),
                ]),
            )
            .map_err(map_engine_err)?;
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_generations SET capture_ordinal = $next_ordinal WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    (
                        "next_ordinal".into(),
                        integer(
                            member.ordinal.checked_add(1).ok_or_else(|| {
                                CgError::Engine(
                                    "event-search capture ordinal exhausted u64".into(),
                                )
                            })?,
                            "capture ordinal",
                        )?,
                    ),
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(generation_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)
}

fn claim_batch(store: &Store, generation: &Generation, bounds: Bounds) -> Result<ClaimResult> {
    let Some(pending) = first_pending_stream(store, generation, bounds)? else {
        return Ok(ClaimResult::Complete);
    };
    let PendingStream {
        start_ordinal,
        connector_id: connector,
        source_instance_ref: source,
        session_ref: session,
        committed_sequence: committed,
        open_tail_start_sequence: open_tail_start,
        open_tail_passage_id,
    } = pending;
    let result = query_rows(
        store,
        bounds,
        &format!(
            "SELECT {SOURCE_MEMBER_COLUMNS} FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND ordinal >= $start_ordinal ORDER BY ordinal ASC LIMIT {}",
            bounds.events
        ),
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            (
                "start_ordinal".into(),
                integer(start_ordinal, "claim start ordinal")?,
            ),
            ("connector_id".into(), Value::Text(connector.clone())),
            ("source".into(), Value::Text(source.clone())),
        ]),
    )?;
    let mut events = Vec::new();
    let mut used = 0_usize;
    for row in &result.rows {
        let member_connector = get_text(row, &result.columns, "connector_id")?;
        let member_source = get_text(row, &result.columns, "source_instance_ref")?;
        let member_session = get_text(row, &result.columns, "session_ref")?;
        let member_sequence = get_u64(row, &result.columns, "source_sequence")?;
        if member_connector != connector || member_source != source {
            return Err(stale_claim(
                "event-search source claim crossed its selected stream",
            ));
        }
        if member_session != session || member_sequence <= committed {
            break;
        }
        let event = load_claimed_member(store, generation, &result.columns, row, bounds)?;
        let measured = event_body(&event.event)
            .map(|body| event_prefix(&event.event).len().saturating_add(body.len()))
            .unwrap_or(0);
        if measured > bounds.bytes {
            return Err(CgError::Engine(format!(
                "event-search source event {} exceeds max_bytes_per_batch {}",
                event.event.id, bounds.bytes
            )));
        }
        if !events.is_empty() && used.saturating_add(measured) > bounds.bytes {
            break;
        }
        used = used.saturating_add(measured);
        events.push(event);
    }
    if events.is_empty() {
        return Err(CgError::Engine(
            "event-search claim found no bounded source event".into(),
        ));
    }
    let start_sequence = events.first().expect("non-empty claim").event.seq;
    let end_sequence = events.last().expect("non-empty claim").event.seq;
    let reopen_tail_start = reopenable_tail_start(
        store,
        generation,
        &connector,
        &source,
        &session,
        open_tail_start,
        bounds,
    )?;
    let open_tail_events = load_open_tail(
        store,
        generation,
        &connector,
        &source,
        &session,
        committed,
        reopen_tail_start,
        bounds,
    )?;
    // Only the passage the cursor recorded as its open tail is re-rendered by this claim. A closed
    // passage whose events are retained as ordinary overlap is never reopened and stays current.
    let reopened_tail_passage_id = reopen_tail_start.and(open_tail_passage_id);
    let superseded_tail = load_superseded_tail(
        store,
        generation,
        &connector,
        &source,
        &session,
        reopened_tail_passage_id,
        bounds,
    )?;
    let session_ended = !has_later_session_event(
        store,
        generation,
        &connector,
        &source,
        &session,
        end_sequence,
        bounds,
    )?;
    let tail_can_reopen =
        !has_later_source_event(store, generation, &connector, &source, end_sequence, bounds)?;
    let cursor_sha256 = claim_hash(open_tail_events.iter().chain(events.iter()));
    let passage_ordinal = generation_passage_count(store, generation.id, bounds)?;
    let owner_token = uuid::Uuid::now_v7().to_string();
    let now = now_millis()?;
    let expires_at = now
        .checked_add(
            i64::try_from(bounds.lease_ms)
                .map_err(|_| CgError::Engine("event-search lease exceeds TIMESTAMP".into()))?,
        )
        .ok_or_else(|| CgError::Engine("event-search lease expiry exhausted TIMESTAMP".into()))?;
    let tx = store.begin_tx()?;
    if let Err(error) = guard_generation_for_claim_in_tx(store, tx, generation) {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    let existing = store.internal_db().execute_in_tx(
        tx,
        "SELECT owner_token, lease_epoch, expires_at FROM event_search_leases WHERE context_id = $context_id AND generation_id = $generation_id",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(generation.context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
        ]),
    ).map_err(map_engine_err)?;
    let lease_epoch = if let Some(row) = existing.rows.first() {
        let persisted_expiry = get_timestamp(row, &existing.columns, "expires_at")?;
        if persisted_expiry > now {
            let _ = store.internal_db().rollback(tx);
            return Err(stale_claim(
                "another event-search worker owns the unexpired claim",
            ));
        }
        let epoch = get_i64(row, &existing.columns, "lease_epoch")?
            .checked_add(1)
            .ok_or_else(|| CgError::Engine("event-search lease epoch exhausted INTEGER".into()))?;
        store.internal_db().execute_in_tx(
            tx,
            "DELETE FROM event_search_leases WHERE context_id = $context_id AND generation_id = $generation_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(generation.context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ]),
        ).map_err(map_engine_err)?;
        epoch
    } else {
        1
    };
    let insert = store.internal_db().execute_in_tx(
        tx,
        "INSERT INTO event_search_leases (context_id, generation_id, owner_token, lease_epoch, claimed_connector_id, claimed_source_instance_ref, claimed_start_sequence, claimed_end_sequence, cursor_sha256, expires_at) VALUES ($context_id, $generation_id, $owner_token, $lease_epoch, $connector_id, $source, $start, $end, $cursor_sha256, $expires_at)",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(generation.context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ("owner_token".into(), Value::Text(owner_token.clone())),
            ("lease_epoch".into(), Value::Int64(lease_epoch)),
            ("connector_id".into(), Value::Text(connector.clone())),
            ("source".into(), Value::Text(source.clone())),
            ("start".into(), integer(start_sequence, "claim start")?),
            ("end".into(), integer(end_sequence, "claim end")?),
            ("cursor_sha256".into(), Value::Text(cursor_sha256.clone())),
            ("expires_at".into(), Value::Timestamp(expires_at)),
        ]),
    );
    if let Err(error) = insert {
        let _ = store.internal_db().rollback(tx);
        return Err(map_engine_err(error));
    }
    store.internal_db().commit(tx).map_err(map_engine_err)?;
    Ok(ClaimResult::Work(Box::new(Claim {
        generation: generation.clone(),
        owner_token,
        lease_epoch,
        expires_at,
        connector_id: connector,
        source_instance_ref: source,
        session_ref: session,
        start_sequence,
        end_sequence,
        cursor_sha256,
        passage_ordinal,
        prior_committed_sequence: committed,
        prior_open_tail_start_sequence: open_tail_start,
        prior_open_tail_passage_id: open_tail_passage_id,
        reopened_tail_passage_id,
        open_tail_events,
        superseded_tail,
        session_ended,
        tail_can_reopen,
        events,
    })))
}

fn guard_generation_for_claim_in_tx(
    store: &Store,
    tx: TxId,
    generation: &Generation,
) -> Result<()> {
    let result = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT context_id, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, state FROM event_search_generations WHERE id = $id",
            &HashMap::from([(
                "id".into(),
                Value::Uuid(generation.id.as_uuid()),
            )]),
        )
        .map_err(map_engine_err)?;
    let row = result
        .rows
        .first()
        .ok_or_else(|| stale_claim("event-search generation disappeared before claim"))?;
    if result.rows.len() != 1
        || get_uuid(row, &result.columns, "context_id")? != generation.context_id.as_uuid()
        || get_u64(row, &result.columns, "event_count")? != generation.event_count
        || get_text(row, &result.columns, "ordered_source_set_sha256")?
            != generation.ordered_source_set_sha256
        || get_text(row, &result.columns, "source_content_sha256")?
            != generation.source_content_sha256
        || get_u64(row, &result.columns, "capture_ordinal")? != generation.target.event_count
        || get_text(row, &result.columns, "state")? != generation.expected_state()
    {
        return Err(stale_claim("event-search generation changed before claim"));
    }
    let conditions = exact_conditions(
        row,
        &result.columns,
        &[
            "context_id",
            "event_count",
            "ordered_source_set_sha256",
            "source_content_sha256",
            "capture_ordinal",
            "state",
        ],
    )?;
    if !store
        .internal_db()
        .guard_row_conditions_in_tx(
            tx,
            "event_search_generations",
            "id",
            &Value::Uuid(generation.id.as_uuid()),
            &conditions,
        )
        .map_err(map_engine_err)?
    {
        return Err(stale_claim("event-search generation changed during claim"));
    }
    if generation.mode == GenerationMode::ExtendReady {
        let head = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
                &HashMap::from([(
                    "context_id".into(),
                    Value::Uuid(generation.context_id.as_uuid()),
                )]),
            )
            .map_err(map_engine_err)?;
        if head.rows.len() != 1
            || get_uuid(&head.rows[0], &head.columns, "generation_id")? != generation.id.as_uuid()
        {
            return Err(stale_claim(
                "event-search ready generation stopped being the current head before claim",
            ));
        }
    }
    Ok(())
}

fn load_claimed_member(
    store: &Store,
    generation: &Generation,
    columns: &[String],
    row: &[Value],
    bounds: Bounds,
) -> Result<ClaimedEvent> {
    let session = store
        .internal_db()
        .read_session(read_limits(bounds))
        .map_err(map_engine_err)?;
    let membership_id = get_uuid(row, columns, "membership_id")?;
    let ordinal = get_u64(row, columns, "ordinal")?;
    let event_id = crate::EventId::from(get_uuid(row, columns, "event_id")?);
    let selected_id =
        get_optional_interpretation_id(row, columns, "effective_interpretation_generation_id")?;
    let result = session
        .execute(
            &format!(
                "SELECT {CAPTURED_EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND id = $event_id"
            ),
            &HashMap::from([
                (
                    "context_id".into(),
                    Value::Uuid(generation.context_id.as_uuid()),
                ),
                ("event_id".into(), Value::Uuid(event_id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    let event_row = result
        .rows
        .first()
        .ok_or_else(|| stale_claim("sealed event-search source event disappeared"))?;
    if result.rows.len() != 1 {
        return Err(stale_claim(
            "sealed event-search source event lookup was not unique",
        ));
    }
    let raw = captured_raw_from_row(
        &session,
        bounds,
        generation.context_id,
        &result.columns,
        event_row,
    )?;
    let captured_sha256 = captured_source_sha256(
        &raw.event_columns,
        &raw.event_row,
        &raw.anchor_columns,
        &raw.anchor_rows,
    );
    let bindings = selected_id.map_or_else(BTreeMap::new, |id| {
        BTreeMap::from([(
            raw.event.connector_id.clone(),
            InterpretationBinding {
                provenance: InterpretationProvenance::Active(id),
            },
        )])
    });
    let (event, actual_selected_id, effective_sha256) =
        effective_captured_event(&session, &bindings, raw.event, &captured_sha256)?;
    if event.id != event_id
        || event.connector_id != get_text(row, columns, "connector_id")?
        || event.source_instance_ref != get_text(row, columns, "source_instance_ref")?
        || event.session_ref != get_text(row, columns, "session_ref")?
        || event.seq != get_u64(row, columns, "source_sequence")?
        || actual_selected_id != selected_id
        || captured_sha256 != get_text(row, columns, "captured_source_sha256")?
        || effective_sha256 != get_text(row, columns, "effective_source_sha256")?
    {
        return Err(stale_claim(
            "sealed event-search membership no longer matches its exact source input",
        ));
    }
    Ok(ClaimedEvent {
        membership_id,
        ordinal,
        event,
        effective_interpretation_generation_id: selected_id,
        captured_source_sha256: captured_sha256,
        effective_source_sha256: effective_sha256,
    })
}

fn first_pending_stream(
    store: &Store,
    generation: &Generation,
    bounds: Bounds,
) -> Result<Option<PendingStream>> {
    let mut pending = None;
    stream_rows(
        store,
        bounds,
        "SELECT ordinal, connector_id, source_instance_ref, session_ref, source_sequence FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal ASC",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
        ]),
        |columns, row| {
            if pending.is_some() {
                return Ok(());
            }
            let connector = get_text(row, columns, "connector_id")?;
            let source = get_text(row, columns, "source_instance_ref")?;
            let session = get_text(row, columns, "session_ref")?;
            let seq = get_u64(row, columns, "source_sequence")?;
            let (committed, open_tail_start, open_tail_passage_id) =
                cursor_state(store, generation, &connector, &source, bounds)?;
            if seq > committed {
                pending = Some(PendingStream {
                    start_ordinal: get_u64(row, columns, "ordinal")?,
                    connector_id: connector,
                    source_instance_ref: source,
                    session_ref: session,
                    committed_sequence: committed,
                    open_tail_start_sequence: open_tail_start,
                    open_tail_passage_id,
                });
            }
            Ok(())
        },
    )?;
    Ok(pending)
}

fn cursor_state(
    store: &Store,
    generation: &Generation,
    connector: &str,
    source: &str,
    bounds: Bounds,
) -> Result<(u64, Option<u64>, Option<uuid::Uuid>)> {
    let result = query_rows(
        store,
        bounds,
        "SELECT committed_sequence, open_tail_start_sequence, open_tail_passage_id FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
            ("source".into(), Value::Text(source.into())),
        ]),
    )?;
    result.rows.first().map_or(Ok((0, None, None)), |row| {
        Ok((
            get_u64(row, &result.columns, "committed_sequence")?,
            get_optional_u64(row, &result.columns, "open_tail_start_sequence")?,
            get_optional_uuid(row, &result.columns, "open_tail_passage_id")?,
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn reopenable_tail_start(
    store: &Store,
    generation: &Generation,
    connector: &str,
    source: &str,
    session: &str,
    open_tail_start: Option<u64>,
    bounds: Bounds,
) -> Result<Option<u64>> {
    let Some(start) = open_tail_start else {
        return Ok(None);
    };
    let result = query_rows(
        store,
        bounds,
        "SELECT session_ref FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND source_sequence = $start",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
            ("source".into(), Value::Text(source.into())),
            ("start".into(), integer(start, "open tail start")?),
        ]),
    )?;
    let row = result
        .rows
        .first()
        .ok_or_else(|| stale_claim("event-search open tail member disappeared"))?;
    if result.rows.len() != 1 {
        return Err(stale_claim(
            "event-search open tail member lookup was not unique",
        ));
    }
    if get_text(row, &result.columns, "session_ref")? == session {
        Ok(Some(start))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn load_open_tail(
    store: &Store,
    generation: &Generation,
    connector: &str,
    source: &str,
    session: &str,
    committed: u64,
    open_tail_start: Option<u64>,
    bounds: Bounds,
) -> Result<Vec<ClaimedEvent>> {
    let Some(start) = open_tail_start else {
        return Ok(Vec::new());
    };
    if start > committed {
        return Err(stale_claim(
            "event-search open tail starts after its committed cursor",
        ));
    }
    let mut tail = Vec::new();
    stream_rows(
        store,
        bounds,
        &format!(
            "SELECT {SOURCE_MEMBER_COLUMNS} FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session AND source_sequence >= $start AND source_sequence <= $committed ORDER BY ordinal ASC"
        ),
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
            ("source".into(), Value::Text(source.into())),
            ("session".into(), Value::Text(session.into())),
            ("start".into(), integer(start, "open tail start")?),
            (
                "committed".into(),
                integer(committed, "committed sequence")?,
            ),
        ]),
        |columns, row| {
            tail.push(load_claimed_member(
                store, generation, columns, row, bounds,
            )?);
            Ok(())
        },
    )?;
    if tail.first().map(|event| event.event.seq) != Some(start)
        || tail.last().map(|event| event.event.seq) != Some(committed)
    {
        return Err(stale_claim("event-search open tail is incomplete"));
    }
    Ok(tail)
}

#[allow(clippy::too_many_arguments)]
fn load_superseded_tail(
    store: &Store,
    generation: &Generation,
    connector: &str,
    source: &str,
    session: &str,
    open_tail_passage_id: Option<uuid::Uuid>,
    bounds: Bounds,
) -> Result<Vec<SupersededPassage>> {
    let Some(passage_id) = open_tail_passage_id else {
        return Ok(Vec::new());
    };
    let mut passages = Vec::new();
    stream_rows(
        store,
        bounds,
        "SELECT id, context_id, generation_id, connector_id, session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_current, created_at FROM event_search_passages WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session AND is_current = $current AND id = $passage_id ORDER BY passage_ordinal ASC",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
            ("source".into(), Value::Text(source.into())),
            ("session".into(), Value::Text(session.into())),
            ("current".into(), Value::Bool(true)),
            ("passage_id".into(), Value::Uuid(passage_id)),
        ]),
        |columns, row| {
            passages.push(SupersededPassage {
                id: EventSearchPassageId::from(get_uuid(row, columns, "id")?),
                conditions: exact_conditions(
                    row,
                    columns,
                    &[
                        "context_id",
                        "generation_id",
                        "connector_id",
                        "session_ref",
                        "source_instance_ref",
                        "passage_ordinal",
                        "first_event_sequence",
                        "last_event_sequence",
                        "rendered_text",
                        "content_sha256",
                        "vector_text",
                        "dimension",
                        "is_current",
                        "created_at",
                    ],
                )?,
            });
            Ok(())
        },
    )?;
    Ok(passages)
}

#[allow(clippy::too_many_arguments)]
fn has_later_session_event(
    store: &Store,
    generation: &Generation,
    connector: &str,
    source: &str,
    session: &str,
    after: u64,
    bounds: Bounds,
) -> Result<bool> {
    let result = query_rows(
        store,
        bounds,
        "SELECT source_sequence FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session AND source_sequence > $after ORDER BY ordinal ASC LIMIT 1",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
            ("source".into(), Value::Text(source.into())),
            ("session".into(), Value::Text(session.into())),
            ("after".into(), integer(after, "claimed sequence")?),
        ]),
    )?;
    Ok(!result.rows.is_empty())
}

fn has_later_source_event(
    store: &Store,
    generation: &Generation,
    connector: &str,
    source: &str,
    after: u64,
    bounds: Bounds,
) -> Result<bool> {
    let result = query_rows(
        store,
        bounds,
        "SELECT source_sequence FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND source_sequence > $after ORDER BY ordinal ASC LIMIT 1",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
            ("connector_id".into(), Value::Text(connector.into())),
            ("source".into(), Value::Text(source.into())),
            ("after".into(), integer(after, "claimed sequence")?),
        ]),
    )?;
    Ok(!result.rows.is_empty())
}

fn generation_passage_count(
    store: &Store,
    id: EventSearchGenerationId,
    bounds: Bounds,
) -> Result<u64> {
    let result = query_rows(
        store,
        bounds,
        "SELECT passage_count FROM event_search_generations WHERE id = $id",
        &HashMap::from([("id".into(), Value::Uuid(id.as_uuid()))]),
    )?;
    result
        .rows
        .first()
        .map(|row| get_u64(row, &result.columns, "passage_count"))
        .transpose()?
        .ok_or(CgError::NotFound {
            id: Some(id.to_string()),
            missing: Some("event search generation"),
        })
}

fn render_and_embed(
    claim: Claim,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
) -> Result<BuiltBatch> {
    let eligible = u64::try_from(
        claim
            .events
            .iter()
            .filter(|event| event_body(&event.event).is_some())
            .count(),
    )
    .map_err(|_| CgError::Engine("eligible event count exceeds u64".into()))?;
    let hard = usize::try_from(profile.hard_passage_tokens.get())
        .unwrap_or(usize::MAX)
        .min(
            registration
                .embedder
                .event_search_capability()
                .map(|capability| capability.token_limit())
                .ok_or_else(|| CgError::EventSearchPreparationUnsupported {
                    embedding_space_id: registration.space.embedding_space_id.clone(),
                })?,
        );
    let target = usize::try_from(profile.target_passage_tokens.get()).unwrap_or(usize::MAX);
    let ordinary_overlap =
        usize::try_from(profile.ordinary_overlap_events.get()).unwrap_or(usize::MAX);
    let mut passages = Vec::new();
    // The cursor tail is the already-formed group left by the preceding claim. Restoring it as
    // group state avoids replaying cuts whose passages were committed with that claim.
    let mut ordinary = claim
        .open_tail_events
        .iter()
        .filter_map(render_unit)
        .collect::<Vec<_>>();
    if !ordinary.is_empty() && prepared_group_tokens(registration, &ordinary, None)? > hard {
        return Err(stale_claim(
            "event-search open tail exceeds its recorded hard passage limit",
        ));
    }
    let mut excluded = 0_u64;
    for event in &claim.events {
        let Some(unit) = render_unit(event) else {
            excluded += 1;
            continue;
        };
        let unit_tokens = prepared_unit_tokens(registration, &unit)?;
        if unit_tokens > hard {
            emit_ordinary(
                &claim,
                &mut ordinary,
                &mut passages,
                registration,
                hard,
                ordinary_overlap,
                false,
            )?;
            let remaining = bounds.passages.saturating_sub(passages.len());
            let slices = build_oversized_passages(
                &claim,
                claim.passage_ordinal
                    + u64::try_from(passages.len())
                        .map_err(|_| CgError::Engine("passage ordinal exceeds u64".into()))?,
                unit,
                registration,
                profile,
                hard,
                remaining,
            )?;
            passages.extend(slices);
            continue;
        }
        if !ordinary.is_empty() {
            let current_tokens = prepared_group_tokens(registration, &ordinary, None)?;
            let candidate_tokens = prepared_group_tokens(registration, &ordinary, Some(&unit))?;
            let preferred_user_boundary = current_tokens >= target
                && unit.event.event.render_role == EventRenderRole::HumanMessage;
            if candidate_tokens > hard || preferred_user_boundary {
                emit_ordinary(
                    &claim,
                    &mut ordinary,
                    &mut passages,
                    registration,
                    hard,
                    ordinary_overlap,
                    true,
                )?;
                if !ordinary.is_empty()
                    && prepared_group_tokens(registration, &ordinary, Some(&unit))? > hard
                {
                    return Err(CgError::Engine(
                        "event-search ordinary overlap leaves no room for the next event".into(),
                    ));
                }
            }
        }
        ordinary.push(unit);
    }
    let final_ordinary_start = ordinary.first().map(|unit| unit.event.event.seq);
    let next_open_tail_start_sequence = if claim.session_ended {
        emit_ordinary(
            &claim,
            &mut ordinary,
            &mut passages,
            registration,
            hard,
            ordinary_overlap,
            false,
        )?;
        if claim.tail_can_reopen {
            final_ordinary_start
        } else {
            None
        }
    } else {
        final_ordinary_start
    };
    if passages.len() > bounds.passages {
        return Err(CgError::Engine(format!(
            "event-search claim renders {} passages, exceeding max_passages_per_batch {}",
            passages.len(),
            bounds.passages
        )));
    }
    Ok(BuiltBatch {
        claim,
        passages,
        eligible_events: eligible,
        excluded_events: excluded,
        next_open_tail_start_sequence,
    })
}

fn render_unit(event: &ClaimedEvent) -> Option<RenderUnit> {
    event_body(&event.event).map(|body| RenderUnit {
        event: event.clone(),
        prefix: event_prefix(&event.event),
        body,
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_ordinary(
    claim: &Claim,
    group: &mut Vec<RenderUnit>,
    passages: &mut Vec<BuiltPassage>,
    registration: &crate::EmbedderRegistration,
    hard: usize,
    overlap_events: usize,
    retain_overlap: bool,
) -> Result<()> {
    if group.is_empty() {
        return Ok(());
    }
    let ordinal = claim.passage_ordinal
        + u64::try_from(passages.len())
            .map_err(|_| CgError::Engine("passage ordinal exceeds u64".into()))?;
    passages.push(build_ordinary_passage(
        claim,
        ordinal,
        group,
        registration,
        hard,
    )?);
    if retain_overlap {
        let keep_from = group.len().saturating_sub(overlap_events);
        *group = group[keep_from..].to_vec();
    } else {
        group.clear();
    }
    Ok(())
}

fn prepared_unit_tokens(
    registration: &crate::EmbedderRegistration,
    unit: &RenderUnit,
) -> Result<usize> {
    registration
        .embedder
        .prepare_document(&format!("{}{}", unit.prefix, unit.body))
        .map(|prepared| prepared.token_count())
}

fn prepared_group_tokens(
    registration: &crate::EmbedderRegistration,
    group: &[RenderUnit],
    next: Option<&RenderUnit>,
) -> Result<usize> {
    let text = render_group_text(group.iter().chain(next));
    registration
        .embedder
        .prepare_document(&text)
        .map(|prepared| prepared.token_count())
}

fn render_group_text<'a>(units: impl Iterator<Item = &'a RenderUnit>) -> String {
    let mut text = String::new();
    for unit in units {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&unit.prefix);
        text.push_str(&unit.body);
    }
    text
}

fn build_ordinary_passage(
    claim: &Claim,
    ordinal: u64,
    units: &[RenderUnit],
    registration: &crate::EmbedderRegistration,
    hard: usize,
) -> Result<BuiltPassage> {
    let mut text = String::new();
    let mut sources = Vec::new();
    for (source_order, unit) in units.iter().enumerate() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&unit.prefix);
        let start = text.len();
        text.push_str(&unit.body);
        let end = text.len();
        sources.push(source_link(
            claim,
            unit,
            source_order,
            None,
            None,
            start,
            end,
        )?);
    }
    finish_passage(claim, ordinal, text, sources, registration, hard)
}

#[allow(clippy::too_many_arguments)]
fn build_oversized_passages(
    claim: &Claim,
    first_ordinal: u64,
    unit: RenderUnit,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    hard: usize,
    maximum: usize,
) -> Result<Vec<BuiltPassage>> {
    let overlap =
        usize::try_from(profile.oversized_event_overlap_tokens.get()).unwrap_or(usize::MAX);
    let prepared_body = registration.embedder.prepare_document(&unit.body)?;
    let body_spans = prepared_body
        .byte_boundaries()
        .iter()
        .copied()
        .filter(|(start, end)| {
            start < end
                && *end <= unit.body.len()
                && unit.body.is_char_boundary(*start)
                && unit.body.is_char_boundary(*end)
        })
        .collect::<Vec<_>>();
    let mut candidate_ends = body_spans
        .iter()
        .map(|(_, end)| *end)
        .chain(std::iter::once(unit.body.len()))
        .filter(|end| *end <= unit.body.len() && unit.body.is_char_boundary(*end))
        .collect::<Vec<_>>();
    candidate_ends.sort_unstable();
    candidate_ends.dedup();
    let mut start = 0_usize;
    let mut passages = Vec::new();
    while start < unit.body.len() {
        if passages.len() == maximum {
            return Err(CgError::Engine(format!(
                "oversized event requires more than max_passages_per_batch {maximum}"
            )));
        }
        let mut selected_end = None;
        for end in candidate_ends
            .iter()
            .rev()
            .copied()
            .filter(|end| *end > start)
        {
            let text = format!("{}{}", unit.prefix, &unit.body[start..end]);
            if registration.embedder.prepare_document(&text)?.token_count() <= hard {
                selected_end = Some(end);
                break;
            }
        }
        let end = selected_end;
        let Some(end) = end else {
            return Err(CgError::Engine(
                "event-search renderer label leaves no model token for source text".into(),
            ));
        };
        let text = format!("{}{}", unit.prefix, &unit.body[start..end]);
        let passage_start = unit.prefix.len();
        let passage_end = text.len();
        let ordinal = first_ordinal
            + u64::try_from(passages.len())
                .map_err(|_| CgError::Engine("passage ordinal exceeds u64".into()))?;
        let source = source_link(
            claim,
            &unit,
            0,
            Some(start),
            Some(end),
            passage_start,
            passage_end,
        )?;
        passages.push(finish_passage(
            claim,
            ordinal,
            text,
            vec![source],
            registration,
            hard,
        )?);
        if end == unit.body.len() {
            break;
        }
        let emitted_spans = body_spans
            .iter()
            .copied()
            .filter(|(span_start, span_end)| *span_start >= start && *span_end <= end)
            .collect::<Vec<_>>();
        let next = emitted_spans
            .len()
            .checked_sub(overlap)
            .and_then(|index| emitted_spans.get(index))
            .map(|(span_start, _)| *span_start)
            .ok_or_else(|| CgError::Engine("event-search oversized overlap underflow".into()))?;
        if next <= start {
            return Err(CgError::Engine(
                "event-search oversized overlap prevents forward progress".into(),
            ));
        }
        start = next;
    }
    Ok(passages)
}

fn source_link(
    claim: &Claim,
    unit: &RenderUnit,
    source_order: usize,
    event_start: Option<usize>,
    event_end: Option<usize>,
    passage_start: usize,
    passage_end: usize,
) -> Result<PassageEventSource> {
    Ok(PassageEventSource {
        context_id: claim.generation.context_id,
        passage_id: EventSearchPassageId::from(uuid::Uuid::nil()),
        event_id: unit.event.event.id,
        connector_id: unit.event.event.connector_id.clone(),
        session_ref: unit.event.event.session_ref.clone(),
        source_instance_ref: unit.event.event.source_instance_ref.clone(),
        source_order: u64::try_from(source_order)
            .map_err(|_| CgError::Engine("passage source order exceeds u64".into()))?,
        event_start_byte: event_start
            .map(u64::try_from)
            .transpose()
            .map_err(|_| CgError::Engine("event source offset exceeds u64".into()))?,
        event_end_byte: event_end
            .map(u64::try_from)
            .transpose()
            .map_err(|_| CgError::Engine("event source offset exceeds u64".into()))?,
        passage_start_byte: u64::try_from(passage_start)
            .map_err(|_| CgError::Engine("passage offset exceeds u64".into()))?,
        passage_end_byte: u64::try_from(passage_end)
            .map_err(|_| CgError::Engine("passage offset exceeds u64".into()))?,
    })
}

fn finish_passage(
    claim: &Claim,
    ordinal: u64,
    text: String,
    mut sources: Vec<PassageEventSource>,
    registration: &crate::EmbedderRegistration,
    hard: usize,
) -> Result<BuiltPassage> {
    let prepared = registration.embedder.prepare_document(&text)?;
    if prepared.token_count() > hard {
        return Err(CgError::Engine(format!(
            "rendered event-search passage has {} tokens, exceeding resolved hard limit {hard}",
            prepared.token_count()
        )));
    }
    let output = registration.embedder.embed_prepared_document(&prepared)?;
    if output.embedding_space_id != registration.space.embedding_space_id
        || output.dimension != registration.space.dimension
        || output.vector.len() != output.dimension
        || output.vector.iter().any(|value| !value.is_finite())
    {
        return Err(CgError::Engine(
            "event-search embedder returned an invalid passage vector".into(),
        ));
    }
    let id = EventSearchPassageId::new_v7();
    for source in &mut sources {
        source.passage_id = id;
    }
    let target = EmbeddingTarget::EventSearchPassage(id);
    if !matches!(target, EmbeddingTarget::EventSearchPassage(target_id) if target_id == id) {
        return Err(CgError::EventSearchPassageTargetRequiresProjection { passage_id: id });
    }
    let first_sequence = sources
        .first()
        .and_then(|source| {
            claim
                .open_tail_events
                .iter()
                .chain(&claim.events)
                .find(|event| event.event.id == source.event_id)
        })
        .map(|event| event.event.seq)
        .ok_or_else(|| CgError::Engine("event-search passage has no first source".into()))?;
    let last_sequence = sources
        .last()
        .and_then(|source| {
            claim
                .open_tail_events
                .iter()
                .chain(&claim.events)
                .find(|event| event.event.id == source.event_id)
        })
        .map(|event| event.event.seq)
        .ok_or_else(|| CgError::Engine("event-search passage has no last source".into()))?;
    Ok(BuiltPassage {
        id,
        embedding_id: EmbeddingId::new_v7(),
        ordinal,
        first_sequence,
        last_sequence,
        text: text.clone(),
        content_sha256: sha256(text.as_bytes()),
        vector: output.vector,
        dimension: u32::try_from(output.dimension)
            .map_err(|_| CgError::Engine("event-search embedding dimension exceeds u32".into()))?,
        producer: output.producer,
        sources,
    })
}

fn commit_batch(
    store: &Store,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    built: BuiltBatch,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let outcome = commit_batch_in_tx(store, registration, profile, bounds, tx, &built);
    if let Err(error) = outcome {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(|error| match error {
        contextdb_core::Error::ConditionalUpdateConflict { .. } => {
            stale_claim("event-search claim changed before commit")
        }
        other => map_engine_err(other),
    })
}

fn commit_batch_in_tx(
    store: &Store,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    tx: TxId,
    built: &BuiltBatch,
) -> Result<()> {
    let claim = &built.claim;
    if now_millis()? >= claim.expires_at {
        return Err(stale_claim("event-search claim expired before commit"));
    }
    let lease = store.internal_db().execute_in_tx(
        tx,
        "SELECT generation_id, context_id, owner_token, lease_epoch, claimed_connector_id, claimed_source_instance_ref, claimed_start_sequence, claimed_end_sequence, cursor_sha256, expires_at FROM event_search_leases WHERE context_id = $context_id AND generation_id = $generation_id",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(claim.generation.context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
        ]),
    ).map_err(map_engine_err)?;
    let row = lease
        .rows
        .first()
        .ok_or_else(|| stale_claim("event-search claim no longer exists"))?;
    if lease.rows.len() != 1
        || get_text(row, &lease.columns, "owner_token")? != claim.owner_token
        || get_i64(row, &lease.columns, "lease_epoch")? != claim.lease_epoch
        || get_text(row, &lease.columns, "claimed_connector_id")? != claim.connector_id
        || get_text(row, &lease.columns, "claimed_source_instance_ref")?
            != claim.source_instance_ref
        || get_u64(row, &lease.columns, "claimed_start_sequence")? != claim.start_sequence
        || get_u64(row, &lease.columns, "claimed_end_sequence")? != claim.end_sequence
        || get_text(row, &lease.columns, "cursor_sha256")? != claim.cursor_sha256
        || get_timestamp(row, &lease.columns, "expires_at")? != claim.expires_at
    {
        return Err(stale_claim("event-search claim ownership or range changed"));
    }
    let guard_values = exact_conditions(
        row,
        &lease.columns,
        &[
            "context_id",
            "owner_token",
            "lease_epoch",
            "claimed_connector_id",
            "claimed_source_instance_ref",
            "claimed_start_sequence",
            "claimed_end_sequence",
            "cursor_sha256",
            "expires_at",
        ],
    )?;
    let guarded = store
        .internal_db()
        .guard_row_conditions_in_tx(
            tx,
            "event_search_leases",
            "generation_id",
            &Value::Uuid(claim.generation.id.as_uuid()),
            &guard_values,
        )
        .map_err(map_engine_err)?;
    if !guarded {
        return Err(stale_claim("event-search claim changed during commit"));
    }

    let generation = store.internal_db().execute_in_tx(
        tx,
        "SELECT context_id, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, passage_count FROM event_search_generations WHERE id = $id",
        &HashMap::from([("id".into(), Value::Uuid(claim.generation.id.as_uuid()))]),
    ).map_err(map_engine_err)?;
    let generation_row = generation
        .rows
        .first()
        .ok_or_else(|| stale_claim("event-search generation disappeared"))?;
    if get_uuid(generation_row, &generation.columns, "context_id")?
        != claim.generation.context_id.as_uuid()
        || get_text(
            generation_row,
            &generation.columns,
            "interpretation_snapshot_sha256",
        )? != claim.generation.snapshot_sha256
        || get_u64(generation_row, &generation.columns, "event_count")?
            != claim.generation.event_count
        || get_text(
            generation_row,
            &generation.columns,
            "ordered_source_set_sha256",
        )? != claim.generation.ordered_source_set_sha256
        || get_text(generation_row, &generation.columns, "source_content_sha256")?
            != claim.generation.source_content_sha256
        || get_u64(generation_row, &generation.columns, "capture_ordinal")?
            != claim.generation.target.event_count
        || get_text(generation_row, &generation.columns, "renderer_id")? != EVENT_SEARCH_RENDERER_ID
        || get_text(generation_row, &generation.columns, "renderer_version")?
            != EVENT_SEARCH_RENDERER_VERSION
        || get_json(generation_row, &generation.columns, "renderer_profile")?
            != serde_json::to_value(profile).map_err(json_error)?
        || get_text(generation_row, &generation.columns, "embedding_space_id")?
            != registration.space.embedding_space_id
        || get_text(generation_row, &generation.columns, "encoding_profile")?
            != registration.embedder.encoding_profile_id()
        || get_text(generation_row, &generation.columns, "state")?
            != claim.generation.expected_state()
    {
        return Err(stale_claim("event-search generation changed before commit"));
    }
    let generation_guard = exact_conditions(
        generation_row,
        &generation.columns,
        &[
            "context_id",
            "interpretation_snapshot_sha256",
            "event_count",
            "ordered_source_set_sha256",
            "source_content_sha256",
            "capture_ordinal",
            "renderer_id",
            "renderer_version",
            "renderer_profile",
            "embedding_space_id",
            "encoding_profile",
            "state",
            "eligible_event_count",
            "rendered_event_count",
            "embedded_event_count",
            "excluded_event_count",
            "passage_count",
        ],
    )?;
    if !store
        .internal_db()
        .guard_row_conditions_in_tx(
            tx,
            "event_search_generations",
            "id",
            &Value::Uuid(claim.generation.id.as_uuid()),
            &generation_guard,
        )
        .map_err(map_engine_err)?
    {
        return Err(stale_claim("event-search generation changed during commit"));
    }
    if claim.generation.mode == GenerationMode::ExtendReady {
        let head = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
                &HashMap::from([(
                    "context_id".into(),
                    Value::Uuid(claim.generation.context_id.as_uuid()),
                )]),
            )
            .map_err(map_engine_err)?;
        if head.rows.len() != 1
            || get_uuid(&head.rows[0], &head.columns, "generation_id")?
                != claim.generation.id.as_uuid()
        {
            return Err(stale_claim(
                "event-search ready generation stopped being the current head before commit",
            ));
        }
    }

    let reloaded = reload_claimed_events_in_tx(store, tx, claim, bounds)?;
    if claim_hash(reloaded.iter()) != claim.cursor_sha256 {
        return Err(stale_claim(
            "event-search source input changed before commit",
        ));
    }

    for passage in &built.passages {
        store.internal_db().execute_in_tx(
            tx,
            "INSERT INTO event_search_passages (id, context_id, generation_id, connector_id, session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_current, created_at) VALUES ($id, $context_id, $generation_id, $connector_id, $session, $source, $ordinal, $first, $last, $text, $sha, $vector, $dimension, $current, $created_at)",
            &HashMap::from([
                ("id".into(), Value::Uuid(passage.id.as_uuid())),
                ("context_id".into(), Value::Uuid(claim.generation.context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
                ("connector_id".into(), Value::Text(claim.connector_id.clone())),
                ("session".into(), Value::Text(claim.session_ref.clone())),
                ("source".into(), Value::Text(claim.source_instance_ref.clone())),
                ("ordinal".into(), integer(passage.ordinal, "passage ordinal")?),
                ("first".into(), integer(passage.first_sequence, "first event sequence")?),
                ("last".into(), integer(passage.last_sequence, "last event sequence")?),
                ("text".into(), Value::Text(passage.text.clone())),
                ("sha".into(), Value::Text(passage.content_sha256.clone())),
                ("vector".into(), Value::Vector(passage.vector.clone())),
                ("dimension".into(), Value::Int64(i64::from(passage.dimension))),
                ("current".into(), Value::Bool(true)),
                ("created_at".into(), Value::TxId(tx)),
            ]),
        ).map_err(map_engine_err)?;
        for source in &passage.sources {
            store.internal_db().execute_in_tx(
                tx,
                "INSERT INTO event_search_passage_sources (context_id, generation_id, passage_id, event_id, connector_id, session_ref, source_instance_ref, source_order, event_start_byte, event_end_byte, passage_start_byte, passage_end_byte) VALUES ($context_id, $generation_id, $passage_id, $event_id, $connector_id, $session, $source, $source_order, $event_start, $event_end, $passage_start, $passage_end)",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(source.context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
                    ("passage_id".into(), Value::Uuid(passage.id.as_uuid())),
                    ("event_id".into(), Value::Uuid(source.event_id.as_uuid())),
                    ("connector_id".into(), Value::Text(source.connector_id.clone())),
                    ("session".into(), Value::Text(source.session_ref.clone())),
                    ("source".into(), Value::Text(source.source_instance_ref.clone())),
                    ("source_order".into(), integer(source.source_order, "source order")?),
                    ("event_start".into(), optional_integer(source.event_start_byte, "event start")?),
                    ("event_end".into(), optional_integer(source.event_end_byte, "event end")?),
                    ("passage_start".into(), integer(source.passage_start_byte, "passage start")?),
                    ("passage_end".into(), integer(source.passage_end_byte, "passage end")?),
                ]),
            ).map_err(map_engine_err)?;
        }
        store.internal_db().execute_in_tx(
            tx,
            "INSERT INTO event_search_embedding_handles (id, context_id, generation_id, passage_id, embedding_space_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $generation_id, $passage_id, $space, $producer, $producer_system, $created_at)",
            &HashMap::from([
                ("id".into(), Value::Uuid(passage.embedding_id.as_uuid())),
                ("context_id".into(), Value::Uuid(claim.generation.context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
                ("passage_id".into(), Value::Uuid(passage.id.as_uuid())),
                ("space".into(), Value::Text(registration.space.embedding_space_id.clone())),
                ("producer".into(), producer_value(&passage.producer)),
                ("producer_system".into(), Value::Text(passage.producer.system.clone())),
                ("created_at".into(), Value::TxId(tx)),
            ]),
        ).map_err(map_engine_err)?;
    }
    if !built.passages.is_empty() {
        for superseded in &claim.superseded_tail {
            if !store
                .internal_db()
                .guard_row_conditions_in_tx(
                    tx,
                    "event_search_passages",
                    "id",
                    &Value::Uuid(superseded.id.as_uuid()),
                    &superseded.conditions,
                )
                .map_err(map_engine_err)?
            {
                return Err(stale_claim(
                    "event-search open tail changed before atomic retirement",
                ));
            }
            store
                .internal_db()
                .execute_in_tx(
                    tx,
                    "UPDATE event_search_passages SET is_current = $current WHERE id = $id",
                    &HashMap::from([
                        ("current".into(), Value::Bool(false)),
                        ("id".into(), Value::Uuid(superseded.id.as_uuid())),
                    ]),
                )
                .map_err(map_engine_err)?;
        }
    }
    // The tail passage is the final group emitted at session end while its tail can reopen. A claim
    // that emits nothing leaves the recorded tail current, so it carries forward; any other claim
    // that emits has just retired it.
    let next_open_tail_passage_id = if built.passages.is_empty() {
        built
            .next_open_tail_start_sequence
            .and(claim.reopened_tail_passage_id)
    } else if claim.session_ended && built.next_open_tail_start_sequence.is_some() {
        built.passages.last().map(|passage| passage.id.as_uuid())
    } else {
        None
    };
    upsert_cursor_in_tx(
        store,
        tx,
        claim,
        built.next_open_tail_start_sequence,
        next_open_tail_passage_id,
    )?;
    let eligible = get_u64(generation_row, &generation.columns, "eligible_event_count")?
        .checked_add(built.eligible_events)
        .ok_or_else(|| CgError::Engine("eligible event count exhausted u64".into()))?;
    let excluded = get_u64(generation_row, &generation.columns, "excluded_event_count")?
        .checked_add(built.excluded_events)
        .ok_or_else(|| CgError::Engine("excluded event count exhausted u64".into()))?;
    let passage_count = get_u64(generation_row, &generation.columns, "passage_count")?
        .checked_add(
            u64::try_from(built.passages.len())
                .map_err(|_| CgError::Engine("passage count exceeds u64".into()))?,
        )
        .ok_or_else(|| CgError::Engine("passage count exhausted u64".into()))?;
    store.internal_db().execute_in_tx(
        tx,
        "UPDATE event_search_generations SET eligible_event_count = $eligible, rendered_event_count = $eligible, embedded_event_count = $eligible, excluded_event_count = $excluded, passage_count = $passage_count WHERE id = $id",
        &HashMap::from([
            ("eligible".into(), integer(eligible, "eligible count")?),
            ("excluded".into(), integer(excluded, "excluded count")?),
            ("passage_count".into(), integer(passage_count, "passage count")?),
            ("id".into(), Value::Uuid(claim.generation.id.as_uuid())),
        ]),
    ).map_err(map_engine_err)?;
    store.internal_db().execute_in_tx(
        tx,
        "DELETE FROM event_search_leases WHERE context_id = $context_id AND generation_id = $generation_id",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(claim.generation.context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
        ]),
    ).map_err(map_engine_err)?;
    Ok(())
}

fn reload_claimed_events_in_tx(
    store: &Store,
    tx: TxId,
    claim: &Claim,
    bounds: Bounds,
) -> Result<Vec<ClaimedEvent>> {
    let mut reloaded = Vec::with_capacity(
        claim
            .open_tail_events
            .len()
            .saturating_add(claim.events.len()),
    );
    for expected in claim.open_tail_events.iter().chain(claim.events.iter()) {
        let membership = store
            .internal_db()
            .execute_in_tx(
                tx,
                &format!(
                    "SELECT context_id, generation_id, {SOURCE_MEMBER_COLUMNS} FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal = $ordinal"
                ),
                &HashMap::from([
                    (
                        "context_id".into(),
                        Value::Uuid(claim.generation.context_id.as_uuid()),
                    ),
                    (
                        "generation_id".into(),
                        Value::Uuid(claim.generation.id.as_uuid()),
                    ),
                    ("ordinal".into(), integer(expected.ordinal, "claim ordinal")?),
                ]),
            )
            .map_err(map_engine_err)?;
        let member_row = membership
            .rows
            .first()
            .ok_or_else(|| stale_claim("sealed event-search membership disappeared"))?;
        if membership.rows.len() != 1
            || get_uuid(member_row, &membership.columns, "membership_id")? != expected.membership_id
            || get_uuid(member_row, &membership.columns, "context_id")?
                != claim.generation.context_id.as_uuid()
            || get_uuid(member_row, &membership.columns, "generation_id")?
                != claim.generation.id.as_uuid()
            || get_u64(member_row, &membership.columns, "ordinal")? != expected.ordinal
            || get_uuid(member_row, &membership.columns, "event_id")? != expected.event.id.as_uuid()
            || get_text(member_row, &membership.columns, "connector_id")?
                != expected.event.connector_id
            || get_text(member_row, &membership.columns, "source_instance_ref")?
                != expected.event.source_instance_ref
            || get_text(member_row, &membership.columns, "session_ref")?
                != expected.event.session_ref
            || get_u64(member_row, &membership.columns, "source_sequence")? != expected.event.seq
            || get_optional_interpretation_id(
                member_row,
                &membership.columns,
                "effective_interpretation_generation_id",
            )? != expected.effective_interpretation_generation_id
            || get_text(member_row, &membership.columns, "captured_source_sha256")?
                != expected.captured_source_sha256
            || get_text(member_row, &membership.columns, "effective_source_sha256")?
                != expected.effective_source_sha256
        {
            return Err(stale_claim(
                "sealed event-search membership changed before commit",
            ));
        }
        let member_guard = exact_conditions(
            member_row,
            &membership.columns,
            &[
                "context_id",
                "generation_id",
                "ordinal",
                "event_id",
                "connector_id",
                "source_instance_ref",
                "session_ref",
                "source_sequence",
                "effective_interpretation_generation_id",
                "captured_source_sha256",
                "effective_source_sha256",
            ],
        )?;
        if !store
            .internal_db()
            .guard_row_conditions_in_tx(
                tx,
                "event_search_source_events",
                "membership_id",
                &Value::Uuid(expected.membership_id),
                &member_guard,
            )
            .map_err(map_engine_err)?
        {
            return Err(stale_claim(
                "sealed event-search membership changed during commit",
            ));
        }
        reloaded.push(reload_claimed_member_input_in_tx(
            store, tx, claim, expected, bounds,
        )?);
    }
    Ok(reloaded)
}

fn reload_claimed_member_input_in_tx(
    store: &Store,
    tx: TxId,
    claim: &Claim,
    expected: &ClaimedEvent,
    bounds: Bounds,
) -> Result<ClaimedEvent> {
    let event_result = store
        .internal_db()
        .execute_in_tx(
            tx,
            &format!(
                "SELECT {CAPTURED_EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND id = $event_id"
            ),
            &HashMap::from([
                (
                    "context_id".into(),
                    Value::Uuid(claim.generation.context_id.as_uuid()),
                ),
                ("event_id".into(), Value::Uuid(expected.event.id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    let event_row = event_result
        .rows
        .first()
        .ok_or_else(|| stale_claim("sealed event-search source event disappeared"))?;
    if event_result.rows.len() != 1 {
        return Err(stale_claim(
            "sealed event-search source event lookup was not unique",
        ));
    }
    let mut event = decode_event(&event_result.columns, event_row)?;
    let anchor_limit = usize::try_from(expected.event.anchor_count)
        .unwrap_or(usize::MAX.saturating_sub(1))
        .saturating_add(1);
    let anchors = store
        .internal_db()
        .execute_in_tx(
            tx,
            &format!(
                "SELECT {CAPTURED_ANCHOR_COLUMNS} FROM event_anchors WHERE event_id = $event_id AND context_id = $context_id ORDER BY declaration_version ASC, key ASC, identity_value ASC LIMIT {anchor_limit}"
            ),
            &HashMap::from([
                ("event_id".into(), Value::Uuid(expected.event.id.as_uuid())),
                (
                    "context_id".into(),
                    Value::Uuid(claim.generation.context_id.as_uuid()),
                ),
            ]),
        )
        .map_err(map_engine_err)?;
    let anchor_bytes = canonical_json_bytes(
        &serde_json::to_value((&anchors.columns, &anchors.rows))
            .expect("persisted anchor values always serialize"),
    )
    .len();
    if anchor_bytes > bounds.bytes {
        return Err(stale_claim(
            "sealed event-search anchor input exceeds the commit bound",
        ));
    }
    let captured_sha256 = captured_source_sha256(
        &event_result.columns,
        event_row,
        &anchors.columns,
        &anchors.rows,
    );
    if captured_sha256 != expected.captured_source_sha256 {
        return Err(stale_claim(
            "raw event or normalized anchor changed after search capture",
        ));
    }
    let mut effective_row_sha256 = None;
    if let Some(generation_id) = expected.effective_interpretation_generation_id {
        let params = HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(claim.generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
            ("event_id".into(), Value::Uuid(expected.event.id.as_uuid())),
        ]);
        let source_member = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let source_row = source_member.rows.first().ok_or_else(|| {
            stale_claim("selected interpretation membership disappeared before commit")
        })?;
        if source_member.rows.len() != 1
            || get_text(source_row, &source_member.columns, "source_row_sha256")? != captured_sha256
        {
            return Err(stale_claim(
                "selected interpretation membership changed before commit",
            ));
        }
        let effective = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT captured_content_sha256, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, coverage, effective_sha256 FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let effective_row = effective.rows.first().ok_or_else(|| {
            stale_claim("selected interpretation value disappeared before commit")
        })?;
        if effective.rows.len() != 1
            || get_text(effective_row, &effective.columns, "captured_content_sha256")?
                != event.content_hash
        {
            return Err(stale_claim(
                "selected interpretation value changed before commit",
            ));
        }
        effective_row_sha256 = Some(get_text(
            effective_row,
            &effective.columns,
            "effective_sha256",
        )?);
        apply_effective_row(&mut event, &effective.columns, effective_row)?;
    }
    let effective_sha256 = effective_source_hash(
        &event,
        expected.effective_interpretation_generation_id,
        effective_row_sha256.as_deref(),
    );
    if effective_sha256 != expected.effective_source_sha256 {
        return Err(stale_claim(
            "effective event-search input changed after capture",
        ));
    }
    Ok(ClaimedEvent {
        membership_id: expected.membership_id,
        ordinal: expected.ordinal,
        event,
        effective_interpretation_generation_id: expected.effective_interpretation_generation_id,
        captured_source_sha256: captured_sha256,
        effective_source_sha256: effective_sha256,
    })
}

fn apply_effective_row(
    event: &mut ConversationEvent,
    columns: &[String],
    row: &[Value],
) -> Result<()> {
    event.occurred_at = get_optional_timestamp(row, columns, "occurred_at")?;
    event.author = get_text(row, columns, "author")?;
    event.kind = get_text(row, columns, "kind")?;
    event.render_role = parse_role(&get_text(row, columns, "render_role")?)?;
    event.interaction_ref = get_optional_text(row, columns, "interaction_ref")?;
    event.text = get_optional_text(row, columns, "text")?;
    event.payload = get_json(row, columns, "payload")?;
    event.anchors = get_json(row, columns, "anchors")?;
    event.coverage = parse_coverage(&get_text(row, columns, "coverage")?)?;
    Ok(())
}

fn upsert_cursor_in_tx(
    store: &Store,
    tx: TxId,
    claim: &Claim,
    next_open_tail_start_sequence: Option<u64>,
    next_open_tail_passage_id: Option<uuid::Uuid>,
) -> Result<()> {
    let params = HashMap::from([
        (
            "context_id".into(),
            Value::Uuid(claim.generation.context_id.as_uuid()),
        ),
        (
            "generation_id".into(),
            Value::Uuid(claim.generation.id.as_uuid()),
        ),
        (
            "connector_id".into(),
            Value::Text(claim.connector_id.clone()),
        ),
        (
            "source".into(),
            Value::Text(claim.source_instance_ref.clone()),
        ),
        (
            "committed".into(),
            integer(claim.end_sequence, "committed sequence")?,
        ),
        ("watermark".into(), Value::Text(claim.cursor_sha256.clone())),
        (
            "open_tail".into(),
            optional_integer(next_open_tail_start_sequence, "open tail start")?,
        ),
        (
            "open_tail_passage".into(),
            next_open_tail_passage_id.map_or(Value::Null, Value::Uuid),
        ),
        ("updated_at".into(), Value::TxId(tx)),
    ]);
    let existing = store.internal_db().execute_in_tx(
        tx,
        "SELECT committed_sequence, open_tail_start_sequence, open_tail_passage_id FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source",
        &params,
    ).map_err(map_engine_err)?;
    if let Some(row) = existing.rows.first() {
        if get_u64(row, &existing.columns, "committed_sequence")? != claim.prior_committed_sequence
            || get_optional_u64(row, &existing.columns, "open_tail_start_sequence")?
                != claim.prior_open_tail_start_sequence
            || get_optional_uuid(row, &existing.columns, "open_tail_passage_id")?
                != claim.prior_open_tail_passage_id
        {
            return Err(stale_claim(
                "event-search cursor or open tail changed before commit",
            ));
        }
    } else if claim.prior_committed_sequence != 0
        || claim.prior_open_tail_start_sequence.is_some()
        || claim.prior_open_tail_passage_id.is_some()
    {
        return Err(stale_claim("event-search cursor disappeared before commit"));
    }
    let sql = if existing.rows.is_empty() {
        "INSERT INTO event_search_cursors (context_id, generation_id, connector_id, source_instance_ref, committed_sequence, open_tail_start_sequence, open_tail_passage_id, watermark_sha256, updated_at) VALUES ($context_id, $generation_id, $connector_id, $source, $committed, $open_tail, $open_tail_passage, $watermark, $updated_at)"
    } else {
        "UPDATE event_search_cursors SET committed_sequence = $committed, open_tail_start_sequence = $open_tail, open_tail_passage_id = $open_tail_passage, watermark_sha256 = $watermark, updated_at = $updated_at WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source"
    };
    store
        .internal_db()
        .execute_in_tx(tx, sql, &params)
        .map_err(map_engine_err)?;
    Ok(())
}

fn finish_or_continue(
    store: &Store,
    generation: &Generation,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    selection: InterpretationSelection,
    retry_count: u64,
) -> Result<FinishResult> {
    let mut connectors = event_connectors(store, generation.context_id, bounds)?;
    let explicit_connector = match selection {
        InterpretationSelection::Ordinary => None,
        InterpretationSelection::ExplicitStaged(id) => {
            let connector = explicit_staged_connector(store, generation.context_id, id, bounds)?;
            connectors.insert(connector.clone());
            Some(connector)
        }
    };
    let bindings = interpretation_bindings(
        store,
        generation.context_id,
        &connectors,
        selection,
        explicit_connector.as_deref(),
        bounds,
    )?;
    if hash_json(&snapshot_json(&bindings)) != generation.snapshot_sha256 {
        return Ok(FinishResult::Rebuild);
    }
    let target = capture_source_membership(
        store,
        generation.context_id,
        &bindings,
        Some(generation.id),
        bounds,
    )?;
    let mut next = generation.clone();
    next.bindings = bindings;
    next.target = target;
    if next.target != generation.target {
        return Ok(FinishResult::Continue(next));
    }
    match finish_stable(store, &next, registration, profile, bounds, retry_count)? {
        Some(receipt) => Ok(FinishResult::Ready(Box::new(receipt))),
        None => Ok(FinishResult::Continue(next)),
    }
}

fn finish_stable(
    store: &Store,
    generation: &Generation,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    retry_count: u64,
) -> Result<Option<EventSearchBuildReceipt>> {
    let tx = store.begin_tx()?;
    let result = store.internal_db().execute_in_tx(
        tx,
        "SELECT context_id, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, completed_at FROM event_search_generations WHERE id = $id",
        &HashMap::from([("id".into(), Value::Uuid(generation.id.as_uuid()))]),
    ).map_err(map_engine_err)?;
    let row = result.rows.first().ok_or(CgError::NotFound {
        id: Some(generation.id.to_string()),
        missing: Some("event search generation"),
    })?;
    let eligible = get_u64(row, &result.columns, "eligible_event_count")?;
    let excluded = get_u64(row, &result.columns, "excluded_event_count")?;
    let state = get_text(row, &result.columns, "state")?;
    if get_uuid(row, &result.columns, "context_id")? != generation.context_id.as_uuid()
        || get_text(row, &result.columns, "interpretation_snapshot_sha256")?
            != generation.snapshot_sha256
        || get_u64(row, &result.columns, "event_count")? != generation.event_count
        || get_u64(row, &result.columns, "capture_ordinal")? != generation.target.event_count
        || get_text(row, &result.columns, "ordered_source_set_sha256")?
            != generation.ordered_source_set_sha256
        || get_text(row, &result.columns, "source_content_sha256")?
            != generation.source_content_sha256
        || get_text(row, &result.columns, "renderer_id")? != EVENT_SEARCH_RENDERER_ID
        || get_text(row, &result.columns, "renderer_version")? != EVENT_SEARCH_RENDERER_VERSION
        || get_json(row, &result.columns, "renderer_profile")?
            != serde_json::to_value(profile).map_err(json_error)?
        || get_text(row, &result.columns, "embedding_space_id")?
            != registration.space.embedding_space_id
        || get_text(row, &result.columns, "encoding_profile")?
            != registration.embedder.encoding_profile_id()
        || state != generation.expected_state()
        || eligible.checked_add(excluded) != Some(generation.target.event_count)
    {
        let _ = store.internal_db().rollback(tx);
        return Err(stale_claim(
            "event-search generation seal or counters changed before completion",
        ));
    }
    let finish_guard = exact_conditions(
        row,
        &result.columns,
        &[
            "context_id",
            "interpretation_snapshot_sha256",
            "event_count",
            "ordered_source_set_sha256",
            "source_content_sha256",
            "capture_ordinal",
            "renderer_id",
            "renderer_version",
            "renderer_profile",
            "embedding_space_id",
            "encoding_profile",
            "state",
            "eligible_event_count",
            "rendered_event_count",
            "embedded_event_count",
            "excluded_event_count",
            "failed_event_count",
            "passage_count",
            "completed_at",
        ],
    )?;
    if !store
        .internal_db()
        .guard_row_conditions_in_tx(
            tx,
            "event_search_generations",
            "id",
            &Value::Uuid(generation.id.as_uuid()),
            &finish_guard,
        )
        .map_err(map_engine_err)?
    {
        let _ = store.internal_db().rollback(tx);
        return Err(stale_claim(
            "event-search generation changed during completion",
        ));
    }
    let event_revision = match guard_event_context_revision_in_tx(store, tx, generation.context_id)
    {
        Ok(Some(revision)) => revision,
        Ok(None) => {
            let _ = store.internal_db().rollback(tx);
            return Ok(None);
        }
        Err(error) => {
            let _ = store.internal_db().rollback(tx);
            return Err(error);
        }
    };
    let live_count = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT COUNT(*) AS event_count FROM events WHERE context_id = $context_id",
            &HashMap::from([(
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            )]),
        )
        .map_err(map_engine_err)?;
    let live_row = live_count
        .rows
        .first()
        .ok_or_else(|| stale_claim("event-search final source count returned no row"))?;
    if live_count.rows.len() != 1
        || get_u64(live_row, &live_count.columns, "event_count")? != generation.target.event_count
    {
        let _ = store.internal_db().rollback(tx);
        return Ok(None);
    }
    if let Err(error) = guard_final_interpretations_in_tx(store, tx, generation) {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    #[cfg(context_graph_internal_test_support)]
    if let Some(pause) = FINALIZATION_PAUSE.with(|slot| slot.borrow_mut().take()) {
        pause(store);
    }
    let unchanged_ready = generation.mode == GenerationMode::ExtendReady
        && generation.target.event_count == generation.event_count
        && generation.target.ordered_source_set_sha256 == generation.ordered_source_set_sha256
        && generation.target.source_content_sha256 == generation.source_content_sha256;
    let completed_at = if unchanged_ready {
        get_optional_txid(row, &result.columns, "completed_at")?.ok_or_else(|| {
            stale_claim("event-search ready generation has no completion transaction")
        })?
    } else {
        store.internal_db().execute_in_tx(
            tx,
            "UPDATE event_search_generations SET event_count = $event_count, ordered_source_set_sha256 = $ordered_sha256, source_content_sha256 = $source_sha256, state = $state, completed_at = $completed_at WHERE id = $id",
            &HashMap::from([
                (
                    "event_count".into(),
                    integer(generation.target.event_count, "event count")?,
                ),
                (
                    "ordered_sha256".into(),
                    Value::Text(generation.target.ordered_source_set_sha256.clone()),
                ),
                (
                    "source_sha256".into(),
                    Value::Text(generation.target.source_content_sha256.clone()),
                ),
                ("state".into(), Value::Text("ready".into())),
                ("completed_at".into(), Value::TxId(tx)),
                ("id".into(), Value::Uuid(generation.id.as_uuid())),
            ]),
        ).map_err(map_engine_err)?;
        tx
    };
    let finalized = store.internal_db().execute_in_tx(
        tx,
        "UPDATE event_search_generations SET finalized_event_revision_tx = $revision_tx WHERE id = $id",
        &HashMap::from([
            ("revision_tx".into(), Value::TxId(event_revision)),
            ("id".into(), Value::Uuid(generation.id.as_uuid())),
        ]),
    ).map_err(map_engine_err)?;
    if finalized.rows_affected != 1 {
        let _ = store.internal_db().rollback(tx);
        return Err(stale_claim(
            "event-search generation disappeared before recording its event revision",
        ));
    }
    let changed = if unchanged_ready
        || generation.mode == GenerationMode::ExtendReady
        || generation.pending_dual_head_activation
    {
        false
    } else {
        activate_search_head_in_tx(store, tx, generation.context_id, generation.id)?
    };
    let rendered = get_u64(row, &result.columns, "rendered_event_count")?;
    let embedded = get_u64(row, &result.columns, "embedded_event_count")?;
    let failed = get_u64(row, &result.columns, "failed_event_count")?;
    match store.internal_db().commit(tx) {
        Ok(()) => {}
        Err(EngineError::ConditionalUpdateConflict { .. }) => {
            let _ = store.internal_db().rollback(tx);
            return Ok(None);
        }
        Err(error) => return Err(map_engine_err(error)),
    }
    let (generation_content_sha256, active_passage_count) =
        fold_current_generation_passages(store, generation.id, bounds)?;
    Ok(Some(EventSearchBuildReceipt {
        context_id: generation.context_id,
        generation_id: generation.id,
        interpretation_snapshot_sha256: generation.snapshot_sha256.clone(),
        interpretation_generation_ids: generation
            .bindings
            .values()
            .filter_map(InterpretationBinding::generation_id)
            .collect(),
        source_watermark: EventWatermark {
            event_count: generation.target.event_count,
            ordered_source_set_sha256: generation.target.ordered_source_set_sha256.clone(),
        },
        source_content_sha256: generation.target.source_content_sha256.clone(),
        generation_content_sha256,
        renderer_id: EVENT_SEARCH_RENDERER_ID.into(),
        renderer_version: EVENT_SEARCH_RENDERER_VERSION.into(),
        profile: profile.clone(),
        embedding_space_id: registration.space.embedding_space_id.clone(),
        encoding_profile_id: registration.embedder.encoding_profile_id(),
        eligible_event_count: eligible,
        rendered_event_count: rendered,
        embedded_event_count: embedded,
        excluded_event_count: excluded,
        failed_event_count: failed,
        passage_count: active_passage_count,
        retry_count,
        failure_count: 0,
        state: EventSearchGenerationState::Ready,
        active_generation_changed: changed,
        started_at: generation.started_at,
        completed_at: Some(completed_at),
        timings: EventSearchBuildTimings::default(),
        sanitized_failure: None,
    }))
}

fn guard_event_context_revision_in_tx(
    store: &Store,
    tx: TxId,
    context_id: crate::ContextId,
) -> Result<Option<TxId>> {
    let result = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT revision_tx FROM event_context_revisions WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    let Some(row) = result.rows.first() else {
        return Err(stale_claim(
            "event Context revision disappeared before completion",
        ));
    };
    if result.rows.len() != 1 {
        return Err(stale_claim(
            "event Context revision primary key returned more than one row",
        ));
    }
    let revision = get_txid(row, &result.columns, "revision_tx")?;
    let guarded = store
        .internal_db()
        .guard_row_conditions_in_tx(
            tx,
            "event_context_revisions",
            "context_id",
            &Value::Uuid(context_id.as_uuid()),
            &[("revision_tx".into(), Value::TxId(revision))],
        )
        .map_err(map_engine_err)?;
    Ok(guarded.then_some(revision))
}

fn activate_search_head_in_tx(
    store: &Store,
    tx: TxId,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
) -> Result<bool> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
        ("activated_at".into(), Value::TxId(tx)),
    ]);
    let existing = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
            &params,
        )
        .map_err(map_engine_err)?;
    if existing.rows.first().is_some_and(|row| {
        get_uuid(row, &existing.columns, "generation_id").ok() == Some(generation_id.as_uuid())
    }) {
        return Ok(false);
    }
    let sql = if existing.rows.is_empty() {
        "INSERT INTO event_search_heads (context_id, generation_id, activated_at) VALUES ($context_id, $generation_id, $activated_at)"
    } else {
        "UPDATE event_search_heads SET generation_id = $generation_id, activated_at = $activated_at WHERE context_id = $context_id"
    };
    store
        .internal_db()
        .execute_in_tx(tx, sql, &params)
        .map_err(map_engine_err)?;
    Ok(true)
}

/// A replacement search generation that passed every activation check inside the activation
/// transaction. Its commit-time guards are registered, so a concurrent source append, head
/// change, or generation change makes the whole activation transaction fail at commit.
pub(super) struct StagedSearchActivation {
    pub(super) previous_generation_id: Option<EventSearchGenerationId>,
    pub(super) passage_count: u64,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    promoted_snapshot: Option<(JsonValue, String)>,
}

pub(super) enum StagedSearchCheck {
    Valid(StagedSearchActivation),
    Refused(InterpretationActivationRefusalReason),
    Changed,
}

/// Validates, inside `tx`, that `search_generation_id` is a complete ready generation whose
/// sealed interpretation snapshot selects `interpretation_generation_id` for `connector_id`, whose
/// other connector bindings still equal the current interpretation heads, and whose finalized
/// Context-wide source revision is still the live one.
pub(super) fn check_staged_search_in_tx(
    store: &Store,
    tx: TxId,
    context_id: crate::ContextId,
    connector_id: &str,
    interpretation_generation_id: InterpretationGenerationId,
    search_generation_id: EventSearchGenerationId,
) -> Result<StagedSearchCheck> {
    use InterpretationActivationRefusalReason as Refusal;

    let result = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT context_id, interpretation_generation_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, state, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, completed_at, finalized_event_revision_tx, sanitized_failure FROM event_search_generations WHERE id = $id",
            &HashMap::from([("id".into(), Value::Uuid(search_generation_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    let row = match result.rows.as_slice() {
        [row] if get_uuid(row, &result.columns, "context_id")? == context_id.as_uuid() => row,
        _ => {
            return Err(CgError::NotFound {
                id: Some(search_generation_id.to_string()),
                missing: Some("event search generation"),
            });
        }
    };
    let columns = &result.columns;
    let ready = get_text(row, columns, "state")? == "ready"
        && get_optional_txid(row, columns, "completed_at")?.is_some()
        && get_optional_text(row, columns, "sanitized_failure")?.is_none();
    let Some(finalized_revision) =
        get_optional_txid(row, columns, "finalized_event_revision_tx")?.filter(|_| ready)
    else {
        return Ok(StagedSearchCheck::Refused(Refusal::GenerationNotReady));
    };
    let event_count = get_u64(row, columns, "event_count")?;
    let eligible = get_u64(row, columns, "eligible_event_count")?;
    if get_u64(row, columns, "failed_event_count")? != 0
        || get_u64(row, columns, "capture_ordinal")? != event_count
        || eligible.checked_add(get_u64(row, columns, "excluded_event_count")?) != Some(event_count)
        || get_u64(row, columns, "rendered_event_count")? != eligible
        || get_u64(row, columns, "embedded_event_count")? != eligible
    {
        return Ok(StagedSearchCheck::Refused(Refusal::PartialGeneration));
    }

    let snapshot = get_json(row, columns, "interpretation_snapshot")?;
    if hash_json(&snapshot) != get_text(row, columns, "interpretation_snapshot_sha256")? {
        return Ok(StagedSearchCheck::Refused(
            Refusal::SearchInterpretationMismatch,
        ));
    }
    let Ok(mut bindings) = bindings_from_snapshot(&snapshot) else {
        return Ok(StagedSearchCheck::Refused(
            Refusal::SearchInterpretationMismatch,
        ));
    };
    let staged = match bindings.get(connector_id).map(|binding| binding.provenance) {
        Some(InterpretationProvenance::ExplicitStaged(id))
            if id == interpretation_generation_id =>
        {
            true
        }
        Some(InterpretationProvenance::Active(id)) if id == interpretation_generation_id => false,
        _ => {
            return Ok(StagedSearchCheck::Refused(
                Refusal::SearchInterpretationMismatch,
            ));
        }
    };
    for (connector, binding) in &bindings {
        if connector == connector_id {
            continue;
        }
        let head = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("connector_id".into(), Value::Text(connector.clone())),
                ]),
            )
            .map_err(map_engine_err)?;
        let current = head
            .rows
            .first()
            .map(|row| get_uuid(row, &head.columns, "generation_id"))
            .transpose()?;
        let unchanged = match binding.provenance {
            InterpretationProvenance::CapturedBase => current.is_none(),
            InterpretationProvenance::Active(id) => current == Some(id.as_uuid()),
            InterpretationProvenance::ExplicitStaged(_) => false,
        };
        if !unchanged {
            return Ok(StagedSearchCheck::Refused(
                Refusal::SearchInterpretationMismatch,
            ));
        }
    }
    let selected = bindings
        .values()
        .filter_map(InterpretationBinding::generation_id)
        .collect::<Vec<_>>();
    let sole_selected = (selected.len() == 1).then(|| selected[0]);
    if get_optional_interpretation_id(row, columns, "interpretation_generation_id")?
        != sole_selected
    {
        return Ok(StagedSearchCheck::Refused(
            Refusal::SearchInterpretationMismatch,
        ));
    }

    let revision = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT revision_tx FROM event_context_revisions WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    let live_revision = revision
        .rows
        .first()
        .map(|row| get_txid(row, &revision.columns, "revision_tx"))
        .transpose()?;
    if revision.rows.len() != 1 || live_revision != Some(finalized_revision) {
        return Ok(StagedSearchCheck::Refused(Refusal::SourceWatermarkMismatch));
    }
    let live_count = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT COUNT(*) AS event_count FROM events WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    match live_count.rows.as_slice() {
        [count] if get_u64(count, &live_count.columns, "event_count")? == event_count => {}
        _ => return Ok(StagedSearchCheck::Refused(Refusal::SourceWatermarkMismatch)),
    }

    let passages = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT COUNT(*) AS passage_count FROM event_search_passages WHERE generation_id = $generation_id AND is_current = $current",
            &HashMap::from([
                (
                    "generation_id".into(),
                    Value::Uuid(search_generation_id.as_uuid()),
                ),
                ("current".into(), Value::Bool(true)),
            ]),
        )
        .map_err(map_engine_err)?;
    let passage_count = passages
        .rows
        .first()
        .map(|row| get_u64(row, &passages.columns, "passage_count"))
        .transpose()?
        .ok_or_else(|| stale_claim("event-search passage count returned no row"))?;

    let head = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT generation_id, activated_at FROM event_search_heads WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    let previous_head = head
        .rows
        .first()
        .map(|row| {
            Ok::<_, CgError>((
                EventSearchGenerationId::from(get_uuid(row, &head.columns, "generation_id")?),
                get_txid(row, &head.columns, "activated_at")?,
            ))
        })
        .transpose()?;

    let db = store.internal_db();
    let revision_guarded = db
        .guard_row_conditions_in_tx(
            tx,
            "event_context_revisions",
            "context_id",
            &Value::Uuid(context_id.as_uuid()),
            &[("revision_tx".into(), Value::TxId(finalized_revision))],
        )
        .map_err(map_engine_err)?;
    let generation_guarded = db
        .guard_row_conditions_in_tx(
            tx,
            "event_search_generations",
            "id",
            &Value::Uuid(search_generation_id.as_uuid()),
            &exact_conditions(
                row,
                columns,
                &[
                    "context_id",
                    "interpretation_generation_id",
                    "interpretation_snapshot_sha256",
                    "event_count",
                    "ordered_source_set_sha256",
                    "source_content_sha256",
                    "capture_ordinal",
                    "state",
                    "eligible_event_count",
                    "excluded_event_count",
                    "failed_event_count",
                    "passage_count",
                    "completed_at",
                    "finalized_event_revision_tx",
                ],
            )?,
        )
        .map_err(map_engine_err)?;
    let head_guarded = match previous_head {
        Some((generation_id, activated_at)) => db
            .guard_row_conditions_in_tx(
                tx,
                "event_search_heads",
                "context_id",
                &Value::Uuid(context_id.as_uuid()),
                &[
                    ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                    ("activated_at".into(), Value::TxId(activated_at)),
                ],
            )
            .map_err(map_engine_err)?,
        // An absent head cannot be row-guarded; a concurrent first head makes this activation's
        // INSERT conflict on the Context primary key instead.
        None => true,
    };
    if !(revision_guarded && generation_guarded && head_guarded) {
        return Ok(StagedSearchCheck::Changed);
    }

    let promoted_snapshot = staged.then(|| {
        bindings.insert(
            connector_id.to_owned(),
            InterpretationBinding {
                provenance: InterpretationProvenance::Active(interpretation_generation_id),
            },
        );
        let promoted = snapshot_json(&bindings);
        let promoted_sha256 = hash_json(&promoted);
        (promoted, promoted_sha256)
    });
    Ok(StagedSearchCheck::Valid(StagedSearchActivation {
        previous_generation_id: previous_head.map(|(generation_id, _)| generation_id),
        passage_count,
        context_id,
        generation_id: search_generation_id,
        promoted_snapshot,
    }))
}

/// Makes a checked replacement the Context's search head at `tx`. A staged snapshot entry is
/// rewritten to the active provenance it now has, so ordinary catch-up extends this head in place
/// instead of re-embedding the whole history under a new generation.
pub(super) fn activate_staged_search_in_tx(
    store: &Store,
    tx: TxId,
    activation: &StagedSearchActivation,
) -> Result<()> {
    if let Some((snapshot, snapshot_sha256)) = &activation.promoted_snapshot {
        let promoted = store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_generations SET interpretation_snapshot = $snapshot, interpretation_snapshot_sha256 = $snapshot_sha256 WHERE id = $id",
                &HashMap::from([
                    ("snapshot".into(), Value::Json(snapshot.clone())),
                    (
                        "snapshot_sha256".into(),
                        Value::Text(snapshot_sha256.clone()),
                    ),
                    ("id".into(), Value::Uuid(activation.generation_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        if promoted.rows_affected != 1 {
            return Err(stale_claim(
                "activated event-search generation disappeared before its snapshot was promoted",
            ));
        }
    }
    if !activate_search_head_in_tx(store, tx, activation.context_id, activation.generation_id)? {
        let restamped = store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_heads SET activated_at = $activated_at WHERE context_id = $context_id",
                &HashMap::from([
                    ("activated_at".into(), Value::TxId(tx)),
                    (
                        "context_id".into(),
                        Value::Uuid(activation.context_id.as_uuid()),
                    ),
                ]),
            )
            .map_err(map_engine_err)?;
        if restamped.rows_affected != 1 {
            return Err(stale_claim(
                "event-search head disappeared before activation restamped it",
            ));
        }
    }
    Ok(())
}

fn guard_final_interpretations_in_tx(
    store: &Store,
    tx: TxId,
    generation: &Generation,
) -> Result<()> {
    if generation.mode == GenerationMode::ExtendReady {
        let head = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
                &HashMap::from([(
                    "context_id".into(),
                    Value::Uuid(generation.context_id.as_uuid()),
                )]),
            )
            .map_err(map_engine_err)?;
        if head.rows.len() != 1
            || get_uuid(&head.rows[0], &head.columns, "generation_id")? != generation.id.as_uuid()
        {
            return Err(stale_claim(
                "event-search ready generation stopped being the current head before completion",
            ));
        }
    }
    for (connector, binding) in &generation.bindings {
        match binding.provenance {
            InterpretationProvenance::CapturedBase | InterpretationProvenance::Active(_) => {
                let head = store
                    .internal_db()
                    .execute_in_tx(
                        tx,
                        "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
                        &HashMap::from([
                            (
                                "context_id".into(),
                                Value::Uuid(generation.context_id.as_uuid()),
                            ),
                            ("connector_id".into(), Value::Text(connector.clone())),
                        ]),
                    )
                    .map_err(map_engine_err)?;
                match binding.provenance {
                    InterpretationProvenance::CapturedBase if head.rows.is_empty() => {}
                    InterpretationProvenance::Active(id)
                        if head.rows.len() == 1
                            && get_uuid(&head.rows[0], &head.columns, "generation_id")?
                                == id.as_uuid() => {}
                    _ => {
                        return Err(stale_claim(
                            "event-search interpretation head changed before completion",
                        ));
                    }
                }
            }
            InterpretationProvenance::ExplicitStaged(id) => {
                let selected = store
                    .internal_db()
                    .execute_in_tx(
                        tx,
                        "SELECT event_count, capture_ordinal, build_ordinal, state FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id AND id = $id",
                        &HashMap::from([
                            (
                                "context_id".into(),
                                Value::Uuid(generation.context_id.as_uuid()),
                            ),
                            ("connector_id".into(), Value::Text(connector.clone())),
                            ("id".into(), Value::Uuid(id.as_uuid())),
                        ]),
                    )
                    .map_err(map_engine_err)?;
                let row = selected.rows.first().ok_or_else(|| {
                    stale_claim("explicit interpretation input disappeared before completion")
                })?;
                if selected.rows.len() != 1 {
                    return Err(stale_claim(
                        "explicit interpretation input changed before completion",
                    ));
                }
                validate_ready_interpretation_row(row, &selected.columns, id)?;
            }
        }
    }
    Ok(())
}

fn fold_current_generation_passages(
    store: &Store,
    id: EventSearchGenerationId,
    bounds: Bounds,
) -> Result<(String, u64)> {
    let mut digest = Sha256::new();
    digest.update(b"context-graph/event-search-generation/v1");
    let mut count = 0_u64;
    let mut previous = None;
    stream_rows(
        store,
        bounds,
        "SELECT passage_ordinal, content_sha256 FROM event_search_passages WHERE generation_id = $generation_id AND is_current = $current ORDER BY passage_ordinal ASC",
        &HashMap::from([
            ("generation_id".into(), Value::Uuid(id.as_uuid())),
            ("current".into(), Value::Bool(true)),
        ]),
        |columns, row| {
            let ordinal = get_u64(row, columns, "passage_ordinal")?;
            if previous.is_some_and(|prior| ordinal <= prior) {
                return Err(stale_claim(
                    "current event-search passage ordinals are not strictly increasing",
                ));
            }
            previous = Some(ordinal);
            hash_part(&mut digest, &ordinal.to_be_bytes());
            hash_part(
                &mut digest,
                get_text(row, columns, "content_sha256")?.as_bytes(),
            );
            count = count.checked_add(1).ok_or_else(|| {
                CgError::Engine("active event-search passage count exhausted u64".into())
            })?;
            Ok(())
        },
    )?;
    Ok((finish_hash(digest), count))
}

fn event_body(event: &ConversationEvent) -> Option<String> {
    if let Some(text) = event.text.as_ref().filter(|text| !text.is_empty()) {
        return Some(text.clone());
    }
    if matches!(
        event.render_role,
        EventRenderRole::Metadata | EventRenderRole::Opaque
    ) || !json_has_content(&event.payload)
    {
        return None;
    }
    Some(String::from_utf8(canonical_json_bytes(&event.payload)).expect("canonical JSON is UTF-8"))
}

fn event_prefix(event: &ConversationEvent) -> String {
    let label = match event.render_role {
        EventRenderRole::HumanMessage => "User",
        EventRenderRole::AgentMessage => "Assistant",
        EventRenderRole::ToolRequest => "Tool call",
        EventRenderRole::ToolResult => "Tool result",
        EventRenderRole::System => "System",
        EventRenderRole::Attachment => "Attachment",
        EventRenderRole::Metadata => "Metadata",
        EventRenderRole::Opaque => "Opaque",
    };
    let author = String::from_utf8(canonical_json_bytes(&JsonValue::String(
        event.author.clone(),
    )))
    .expect("JSON strings are UTF-8");
    format!("{label} [author={author}]: ")
}

fn json_has_content(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null => false,
        JsonValue::String(value) => !value.is_empty(),
        JsonValue::Array(values) => values.iter().any(json_has_content),
        JsonValue::Object(values) => values.values().any(json_has_content),
        JsonValue::Bool(_) | JsonValue::Number(_) => true,
    }
}

fn decode_event(columns: &[String], row: &[Value]) -> Result<ConversationEvent> {
    Ok(ConversationEvent {
        id: crate::EventId::from(get_uuid(row, columns, "id")?),
        context_id: crate::ContextId::from(get_uuid(row, columns, "context_id")?),
        connector_id: get_text(row, columns, "connector_id")?,
        adapter_version: get_text(row, columns, "adapter_version")?,
        anchor_declaration_version: get_text(row, columns, "anchor_declaration_version")?,
        source_instance_ref: get_text(row, columns, "source_instance_ref")?,
        session_ref: get_text(row, columns, "session_ref")?,
        source_event_key: get_text(row, columns, "source_event_key")?,
        source_position: get_json(row, columns, "source_position")?,
        seq: get_u64(row, columns, "seq")?,
        content_hash: get_text(row, columns, "content_hash")?,
        anchor_count: get_u64(row, columns, "anchor_count")?,
        anchor_set_hash: get_text(row, columns, "anchor_set_hash")?,
        occurred_at: get_optional_timestamp(row, columns, "occurred_at")?,
        author: get_text(row, columns, "author")?,
        kind: get_text(row, columns, "kind")?,
        render_role: parse_role(&get_text(row, columns, "render_role")?)?,
        interaction_ref: get_optional_text(row, columns, "interaction_ref")?,
        text: get_optional_text(row, columns, "text")?,
        payload: get_json(row, columns, "payload")?,
        anchors: get_json(row, columns, "anchors")?,
        source_remainder: get_json(row, columns, "source_remainder")?,
        coverage: parse_coverage(&get_text(row, columns, "coverage")?)?,
    })
}

fn parse_role(value: &str) -> Result<EventRenderRole> {
    match value {
        "human_message" => Ok(EventRenderRole::HumanMessage),
        "agent_message" => Ok(EventRenderRole::AgentMessage),
        "tool_request" => Ok(EventRenderRole::ToolRequest),
        "tool_result" => Ok(EventRenderRole::ToolResult),
        "system" => Ok(EventRenderRole::System),
        "attachment" => Ok(EventRenderRole::Attachment),
        "metadata" => Ok(EventRenderRole::Metadata),
        "opaque" => Ok(EventRenderRole::Opaque),
        other => Err(CgError::Engine(format!(
            "unknown event render role: {other}"
        ))),
    }
}

fn parse_coverage(value: &str) -> Result<EventCoverage> {
    match value {
        "complete" => Ok(EventCoverage::Complete),
        "opaque" => Ok(EventCoverage::Opaque),
        "payload_refused" => Ok(EventCoverage::PayloadRefused),
        "source_declared_missing" => Ok(EventCoverage::SourceDeclaredMissing),
        other => Err(CgError::Engine(format!("unknown event coverage: {other}"))),
    }
}

fn stream_rows(
    store: &Store,
    bounds: Bounds,
    sql: &str,
    params: &HashMap<String, Value>,
    mut consume: impl FnMut(&[String], &[Value]) -> Result<()>,
) -> Result<()> {
    let session = store
        .internal_db()
        .read_session(read_limits(bounds))
        .map_err(map_engine_err)?;
    let mut cursor = session.open_cursor(sql, params).map_err(map_engine_err)?;
    let read = (|| -> Result<()> {
        let first = cursor.first_page();
        let mut more = first.has_more;
        for row in &first.rows {
            consume(&first.columns, row)?;
        }
        while more {
            let page = cursor
                .fetch(NonZeroUsize::new(bounds.events.max(1)))
                .map_err(map_engine_err)?;
            more = page.has_more;
            for row in &page.rows {
                consume(&page.columns, row)?;
            }
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    read?;
    closed
}

fn query_rows(
    store: &Store,
    bounds: Bounds,
    sql: &str,
    params: &HashMap<String, Value>,
) -> Result<contextdb_engine::QueryResult> {
    let session = store
        .internal_db()
        .read_session(read_limits(bounds))
        .map_err(map_engine_err)?;
    session.execute(sql, params).map_err(map_engine_err)
}

fn read_limits(bounds: Bounds) -> ReadLimits {
    let mut limits = crate::owner_control::context_graph_read_limits();
    limits.result_rows = limits
        .result_rows
        .min(u64::try_from(bounds.events).unwrap_or(u64::MAX))
        .max(1);
    limits.cursor_page_rows = limits
        .cursor_page_rows
        .min(u64::try_from(bounds.events).unwrap_or(u64::MAX))
        .max(1);
    limits
}

fn exact_conditions(
    row: &[Value],
    columns: &[String],
    names: &[&str],
) -> Result<Vec<(String, Value)>> {
    names
        .iter()
        .map(|name| {
            let index = column_index(columns, name)?;
            Ok((
                (*name).into(),
                row.get(index).cloned().ok_or_else(|| {
                    CgError::Engine(format!("column {name}: missing guard value"))
                })?,
            ))
        })
        .collect()
}

fn get_i64(row: &[Value], columns: &[String], name: &str) -> Result<i64> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Int64(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer, got {other:?}"
        ))),
    }
}

fn get_u64(row: &[Value], columns: &[String], name: &str) -> Result<u64> {
    u64::try_from(get_i64(row, columns, name)?)
        .map_err(|_| CgError::Engine(format!("column {name}: expected non-negative integer")))
}

fn get_optional_u64(row: &[Value], columns: &[String], name: &str) -> Result<Option<u64>> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Int64(value)) => u64::try_from(*value)
            .map(Some)
            .map_err(|_| CgError::Engine(format!("column {name}: expected non-negative integer"))),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer|null, got {other:?}"
        ))),
    }
}

fn get_optional_uuid(row: &[Value], columns: &[String], name: &str) -> Result<Option<uuid::Uuid>> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Uuid(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid|null, got {other:?}"
        ))),
    }
}

fn get_optional_interpretation_id(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<InterpretationGenerationId>> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Uuid(value)) => Ok(Some(InterpretationGenerationId::from(*value))),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid|null, got {other:?}"
        ))),
    }
}

fn get_timestamp(row: &[Value], columns: &[String], name: &str) -> Result<i64> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Timestamp(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected timestamp, got {other:?}"
        ))),
    }
}

fn get_optional_timestamp(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<DateTime<Utc>>> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Timestamp(value)) => DateTime::from_timestamp_millis(*value)
            .map(Some)
            .ok_or_else(|| CgError::Engine(format!("column {name}: invalid timestamp"))),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected timestamp|null, got {other:?}"
        ))),
    }
}

fn get_txid(row: &[Value], columns: &[String], name: &str) -> Result<TxId> {
    match row.get(column_index(columns, name)?) {
        Some(Value::TxId(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid, got {other:?}"
        ))),
    }
}

fn get_optional_txid(row: &[Value], columns: &[String], name: &str) -> Result<Option<TxId>> {
    match row.get(column_index(columns, name)?) {
        Some(Value::TxId(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid|null, got {other:?}"
        ))),
    }
}

fn integer(value: u64, name: &str) -> Result<Value> {
    i64::try_from(value)
        .map(Value::Int64)
        .map_err(|_| CgError::Engine(format!("{name} exceeds engine INTEGER")))
}

fn optional_integer(value: Option<u64>, name: &str) -> Result<Value> {
    value
        .map(|value| integer(value, name))
        .transpose()
        .map(|value| value.unwrap_or(Value::Null))
}

fn now_millis() -> Result<i64> {
    i64::try_from(Wallclock::now().0)
        .map_err(|_| CgError::Engine("system timestamp exceeds engine TIMESTAMP".into()))
}

fn claim_hash<'a>(events: impl IntoIterator<Item = &'a ClaimedEvent>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"context-graph/event-search-claim/v1");
    for event in events {
        hash_part(&mut digest, &event.ordinal.to_be_bytes());
        hash_part(&mut digest, event.event.id.as_uuid().as_bytes());
        hash_part(&mut digest, &event.event.seq.to_be_bytes());
        hash_part(&mut digest, event.captured_source_sha256.as_bytes());
        hash_part(&mut digest, event.effective_source_sha256.as_bytes());
    }
    finish_hash(digest)
}

fn stale_claim(reason: &str) -> CgError {
    CgError::Engine(format!("event-search claim refused: {reason}"))
}
fn is_stale_claim(error: &CgError) -> bool {
    matches!(error, CgError::Engine(message) if message.starts_with("event-search claim refused:"))
}
fn json_error(error: serde_json::Error) -> CgError {
    CgError::Engine(format!("event-search JSON encoding failed: {error}"))
}
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hash_json(value: &JsonValue) -> String {
    sha256(&canonical_json_bytes(value))
}
fn hash_part(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}
fn finish_hash(digest: Sha256) -> String {
    format!("{:x}", digest.finalize())
}

fn canonical_json_bytes(value: &JsonValue) -> Vec<u8> {
    serde_json::to_vec(&canonicalize_json(value)).expect("JSON values always serialize")
}

fn canonicalize_json(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(values) => {
            JsonValue::Array(values.iter().map(canonicalize_json).collect())
        }
        JsonValue::Object(values) => JsonValue::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        value => value.clone(),
    }
}
