//! Bounded construction of the derived event-history search projection.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::sync::mpsc::{RecvTimeoutError, sync_channel};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use contextdb_core::{Error as EngineError, Value, Wallclock};
use contextdb_engine::{ReadLimits, ReadSession};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use sha2::{Digest, Sha256};

use crate::{
    CgError, ContextSelector, ConversationEvent, EVENT_SEARCH_RENDERER_ID,
    EVENT_SEARCH_RENDERER_VERSION, EmbeddingId, EmbeddingTarget, EventCoverage, EventRenderRole,
    EventSearchBatchWriteLimit, EventSearchBuildReceipt, EventSearchBuildRequest,
    EventSearchBuildTimings, EventSearchEmbedder, EventSearchFailureDisposition,
    EventSearchGenerationId, EventSearchGenerationState, EventSearchOrder, EventSearchPassageId,
    EventSearchPendingBatch, EventSearchProfile, EventSearchRebuildRequest, EventSearchRunOutcome,
    EventSearchRunRequest, EventSearchSelectedStream, EventSearchSelectedWork,
    EventSearchSessionScope, EventSearchWorkAssessment, EventSearchWorkPolicy, EventSearchWorkSpec,
    EventWatermark, InterpretationActivationRefusalReason, InterpretationGenerationId,
    PassageEventSource, Result, Store, TxId,
};

use super::convert::map_engine_err;
use super::encode::{column_index, get_bool, get_json, get_optional_text, get_text, get_uuid};
use super::event_interpretation::{
    CAPTURED_ANCHOR_COLUMNS, CAPTURED_EVENT_COLUMNS, captured_source_sha256,
};
use super::event_search_read::{EventSearchRows, captured_event_counts};
use super::evidence::producer_value;

const MAX_STALE_RETRIES: usize = 8;

const HELD_BY_READERS_FAILURE: &str =
    "the search index could not be built: other processes were reading the store throughout";
const HELD_BY_WRITER_FAILURE: &str =
    "the search index could not be built: another process was writing the store throughout";
const GENERIC_PROJECTION_FAILURE: &str = "the search projection stopped on a store or model error";

/// Retry this exact reopen while another writer finishes, retaining the claim and rendered batch.
/// The shared Store door independently handles genuine direct-reader hydration.
fn open_for_run(
    config: &crate::StoreConfig,
    embedder: &crate::EmbedderRegistration,
) -> Result<Store> {
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    STORE_OPENS.with(|opens| opens.set(opens.get() + 1));
    super::open_retry::retry(
        &config.db_path,
        super::open_retry::Contention::Writer,
        || Store::open_with_embedder_registrations(config.clone(), vec![embedder.clone()]),
    )
}

/// Admission shared by every projection door that has selected an event-search registration.
/// Capability support stays a typed embedder condition; a supported embedder's finite ceiling is
/// then checked without normalizing the caller's profile.
fn validate_profile_for_registration(
    profile: &EventSearchProfile,
    registration: &crate::EmbedderRegistration,
) -> Result<()> {
    let capability = registration
        .embedder
        .event_search_capability()
        .ok_or_else(|| CgError::EventSearchPreparationUnsupported {
            embedding_space_id: registration.space.embedding_space_id.clone(),
        })?;
    profile.validate_for(capability)
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
/// A ready head keeps its coverage while its own failed extension is reported separately.
pub(crate) fn record_refusal_standalone(
    request: EventSearchRunRequest,
    reason: &str,
) -> Result<()> {
    validate_profile_for_registration(&request.profile, &request.embedder)?;
    let context_ids = request.contexts.context_ids();
    if context_ids.len() != 1 {
        return Err(CgError::Engine(
            "one recorded event-search refusal describes exactly one Context".into(),
        ));
    }
    let context_id = context_ids[0];
    let store = open_for_run(&request.store, &request.embedder)?;
    admit_projection_embedding_space(&store, &request.embedder)?;
    let connectors = event_connectors(&store, context_id)?;
    let bindings = interpretation_bindings(
        &store,
        context_id,
        &connectors,
        InterpretationSelection::Ordinary,
        None,
    )?;
    let snapshot = snapshot_json(&bindings);
    let spec = EventSearchWorkSpec {
        profile: request.profile,
        embedding_space_id: request.embedder.space.embedding_space_id,
        encoding_profile_id: request.embedder.embedder.encoding_profile_id(),
    };
    let generation = select_generation(
        &store,
        context_id,
        &snapshot,
        &hash_json(&snapshot),
        &spec,
        true,
    )?
    .ok_or_else(|| {
        CgError::Engine("no existing event-search generation matches the refusal request".into())
    })?
    .0;
    record_refusal_attempt(&store, context_id, generation.id, reason)
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
    release_claim_in(&store, held);
}

/// [`release_claim`] on a Store the caller already holds.
fn release_claim_in(store: &Store, held: &HeldClaim) {
    let _ = try_release_claim_in(store, held);
}

/// Persist release of this run's exact token and epoch before a stale-claim retry.
///
/// A zero-row delete is success: it means the lease was already removed or replaced, and the
/// ownership predicates ensure a newer owner's row is never deleted. An engine refusal is not a
/// retryable stale claim; the caller must surface it instead of hiding failed cleanup behind the
/// next claim attempt.
fn try_release_claim_in(store: &Store, held: &HeldClaim) -> Result<()> {
    store
        .internal_db()
        .execute(
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
    )
        .map(|_| ())
        .map_err(map_engine_err)
}

/// The claim's current expiry and renewal, carried through preparation and every embed call.
struct LeaseRenewal<'a> {
    expires_at: i64,
    renew: &'a mut dyn FnMut(i64) -> Result<i64>,
}

impl LeaseRenewal<'_> {
    /// Renew if due, between preparation candidates as well as between embed calls.
    fn keep(&mut self) -> Result<()> {
        self.expires_at = (self.renew)(self.expires_at)?;
        Ok(())
    }
}

/// Whether more than a third of the lease has elapsed, so a live run renews its claim now.
fn renewal_due(expires_at: i64, bounds: Bounds) -> Result<bool> {
    let lease = i64::try_from(bounds.lease_ms)
        .map_err(|_| CgError::Engine("event-search lease exceeds TIMESTAMP".into()))?;
    Ok(expires_at.saturating_sub(now_millis()?) < lease.saturating_sub(lease / 3))
}

/// Extend this run's own claim by one lease from now, matched on its token, epoch and the expiry
/// it last wrote. A claim that is gone or was taken over is refused as stale, never re-taken.
fn renew_claim(store: &Store, held: &HeldClaim, expires_at: i64, bounds: Bounds) -> Result<i64> {
    let now = now_millis()?;
    if expires_at <= now {
        return Err(stale_claim("event-search claim expired while embedding"));
    }
    let renewed = now
        .checked_add(
            i64::try_from(bounds.lease_ms)
                .map_err(|_| CgError::Engine("event-search lease exceeds TIMESTAMP".into()))?,
        )
        .ok_or_else(|| CgError::Engine("event-search lease expiry exhausted TIMESTAMP".into()))?;
    let updated = store
        .internal_db()
        .execute(
            "UPDATE event_search_leases SET expires_at = $renewed WHERE context_id = $context_id AND generation_id = $generation_id AND owner_token = $owner_token AND lease_epoch = $lease_epoch AND expires_at = $expires_at",
            &HashMap::from([
                ("renewed".to_string(), Value::Timestamp(renewed)),
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
                ("expires_at".to_string(), Value::Timestamp(expires_at)),
            ]),
        )
        .map_err(map_engine_err)?;
    if updated.rows_affected != 1 {
        return Err(stale_claim(
            "event-search claim was taken over while embedding",
        ));
    }
    Ok(renewed)
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
type StoreHook = Box<dyn FnOnce(&Store)>;

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
thread_local! {
    static STORE_OPENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static MEMBERSHIP_CAPTURES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static NEXT_MEMBERSHIP_CAPTURE_FAILURE: std::cell::RefCell<Option<CgError>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_MEMBERSHIP_READ_PAGE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static PROJECTION_READ_ACTIVE_MS: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static BEFORE_FAILURE_RECORD: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static BEFORE_BATCH_COMMIT: std::cell::RefCell<Option<StoreHook>> = const { std::cell::RefCell::new(None) };
    static AFTER_BATCH_COMMIT: std::cell::RefCell<Option<StoreHook>> = const { std::cell::RefCell::new(None) };
    static BEFORE_ATTEMPT_COMMIT: std::cell::RefCell<Option<StoreHook>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_ATTEMPT_COMMIT_BEFORE_READBACK: std::cell::RefCell<Option<StoreHook>> =
        const { std::cell::RefCell::new(None) };
    static BEFORE_COMPLETION_COMMIT: std::cell::RefCell<Option<StoreHook>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_COMPLETION_COMMIT_BEFORE_READBACK: std::cell::RefCell<Option<StoreHook>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
type BatchAdmissionHook = Box<dyn Fn(&str, &str, u64, Option<u64>, usize)>;

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
thread_local! {
    static BATCH_ADMISSION_HOOK: std::cell::RefCell<Option<BatchAdmissionHook>> = const { std::cell::RefCell::new(None) };
}

/// Observe the engine's actual write budget after opening and before taking each source claim.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn observe_event_search_batch_admission_for_test(hook: Option<BatchAdmissionHook>) {
    BATCH_ADMISSION_HOOK.with(|slot| *slot.borrow_mut() = hook);
}

/// How many Stores the event-search runner has opened on the calling thread so far.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn event_search_store_opens_for_test() -> usize {
    STORE_OPENS.with(std::cell::Cell::get)
}

/// Fail the next source-membership preparation on this thread with one deterministic core error.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn fail_next_event_search_membership_capture_for_test(reason: impl Into<String>) {
    fail_next_event_search_membership_capture_with_error_for_test(CgError::Engine(reason.into()));
}

/// Fail the next source-membership preparation with a typed condition.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn fail_next_event_search_membership_capture_with_error_for_test(error: CgError) {
    NEXT_MEMBERSHIP_CAPTURE_FAILURE.with(|slot| *slot.borrow_mut() = Some(error));
}

/// Observe the first fully read membership page before its rows are processed.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn after_next_event_search_membership_read_page_for_test(hook: impl FnOnce() + 'static) {
    AFTER_MEMBERSHIP_READ_PAGE.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// Use shorter statement and cursor deadlines for projection reads on this thread.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn set_event_search_projection_read_deadline_ms_for_test(active_ms: Option<u64>) {
    PROJECTION_READ_ACTIVE_MS.with(|slot| slot.set(active_ms));
}

/// Return and reset this thread's source-membership preparation count.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn take_event_search_membership_captures_for_test() -> usize {
    MEMBERSHIP_CAPTURES.with(|count| count.replace(0))
}

/// Move the connector's active, ready interpretation generation forward by one ordinal while it
/// stays a completely sealed ready input (`event_count`, `capture_ordinal` and `build_ordinal`
/// advance together) and no event is appended: the stamp a search generation was certified
/// against no longer describes it, while the Context's event revision and the interpretation
/// binding are unchanged. The product's own builds seal a new generation instead; this fixture
/// isolates the stamp half of completion's reuse check.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn advance_bound_interpretation_stamp_for_test(
    store: &Store,
    context_id: crate::ContextId,
    connector_id: &str,
) -> Result<()> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("connector_id".into(), Value::Text(connector_id.into())),
    ]);
    let head = store
        .internal_db()
        .execute(
            "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
            &params,
        )
        .map_err(map_engine_err)?;
    let row = head
        .rows
        .first()
        .ok_or_else(|| CgError::Engine("the connector has no active interpretation".into()))?;
    let id = get_uuid(row, &head.columns, "generation_id")?;
    let mut params = params;
    params.insert("id".into(), Value::Uuid(id));
    store
        .internal_db()
        .execute(
            "UPDATE event_interpretation_generations SET event_count = event_count + 1, capture_ordinal = capture_ordinal + 1, build_ordinal = build_ordinal + 1 WHERE context_id = $context_id AND connector_id = $connector_id AND id = $id AND state = 'ready'",
            &params,
        )
        .map_err(map_engine_err)?;
    Ok(())
}

/// Run one rendezvous immediately before the next matching-attempt failure write.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn before_next_event_search_failure_record_for_test(hook: impl FnOnce() + 'static) {
    BEFORE_FAILURE_RECORD.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// A process-crash cut after staging the cursor and pending marker, before their commit.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn before_next_event_search_batch_commit_for_test(hook: impl FnOnce(&Store) + 'static) {
    BEFORE_BATCH_COMMIT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// A process-crash cut immediately after publishing the cursor and pending marker together.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn after_next_event_search_batch_commit_for_test(hook: impl FnOnce(&Store) + 'static) {
    AFTER_BATCH_COMMIT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// Run one deterministic interleaving immediately before the next attempt transaction commits.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn before_next_event_search_attempt_commit_for_test(hook: impl FnOnce(&Store) + 'static) {
    BEFORE_ATTEMPT_COMMIT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// Run one deterministic competing-owner interleaving after the next attempt commit, before its
/// ownership-filtered committed timestamp readback. Both new and resumed attempts use this seam.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn after_next_event_search_attempt_commit_before_readback_for_test(
    hook: impl FnOnce(&Store) + 'static,
) {
    AFTER_ATTEMPT_COMMIT_BEFORE_READBACK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// Run one deterministic interleaving immediately before the next completion transaction commits.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn before_next_event_search_completion_commit_for_test(hook: impl FnOnce(&Store) + 'static) {
    BEFORE_COMPLETION_COMMIT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// Run one deterministic competing-owner interleaving after the next completion commit, before
/// its ownership-filtered committed timestamp readback.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn after_next_event_search_completion_commit_before_readback_for_test(
    hook: impl FnOnce(&Store) + 'static,
) {
    AFTER_COMPLETION_COMMIT_BEFORE_READBACK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// Consumer-side work performed by one pending-stream discovery pass.
///
/// This is compiled only for the internal regression suite. Engine examined-entry accounting is
/// owned and proved by ContextDB separately.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct EventSearchDiscoveryObservation {
    pub stream_rows_examined: usize,
    pub cursor_lookups: usize,
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
thread_local! {
    static DISCOVERY_OBSERVATIONS: std::cell::RefCell<Vec<EventSearchDiscoveryObservation>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn clear_event_search_discovery_observations_for_test() {
    DISCOVERY_OBSERVATIONS.with(|observations| observations.borrow_mut().clear());
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn event_search_discovery_observations_for_test() -> Vec<EventSearchDiscoveryObservation> {
    DISCOVERY_OBSERVATIONS.with(|observations| observations.borrow().clone())
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
fn observe_event_search_discovery_for_test(stream_rows_examined: usize, cursor_lookups: usize) {
    DISCOVERY_OBSERVATIONS.with(|observations| {
        observations
            .borrow_mut()
            .push(EventSearchDiscoveryObservation {
                stream_rows_examined,
                cursor_lookups,
            });
    });
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
type FinalizationPause = Box<dyn FnOnce(&Store)>;

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
thread_local! {
    static FINALIZATION_PAUSE: std::cell::RefCell<Option<FinalizationPause>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `pause` once inside the next event-search head-switch transaction on this thread, after
/// the Context revision is read and guarded and the live event count is checked, before commit.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn pause_next_event_search_finalization_for_test(pause: impl FnOnce(&Store) + 'static) {
    FINALIZATION_PAUSE.with(|slot| *slot.borrow_mut() = Some(Box::new(pause)));
}
#[cfg(feature = "test-support")]
type WaveObserver = Box<dyn FnMut(usize)>;
#[cfg(feature = "test-support")]
thread_local! {
    static WAVE_OBSERVER: std::cell::RefCell<Option<WaveObserver>> = const { std::cell::RefCell::new(None) };
    static LOADED_EVENT_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CLAIM_LOADS: std::cell::RefCell<Vec<(usize, usize, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}
#[cfg(feature = "test-support")]
pub fn observe_embedding_waves_for_test(observer: impl FnMut(usize) + 'static) {
    WAVE_OBSERVER.with(|slot| *slot.borrow_mut() = Some(Box::new(observer)));
}
#[cfg(feature = "test-support")]
pub fn take_claim_loads_for_test() -> Vec<(usize, usize, usize)> {
    CLAIM_LOADS.with(|loads| std::mem::take(&mut *loads.borrow_mut()))
}

const FIRST_PENDING_SOURCE_STATEMENT: &str = "SELECT ordinal FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session AND source_sequence > $committed ORDER BY source_sequence ASC LIMIT 1";
const HAS_LATER_SESSION_STATEMENT: &str = "SELECT source_sequence FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session AND source_sequence > $after ORDER BY source_sequence ASC LIMIT 1";
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
const SELECTED_STREAM_COUNT_STATEMENT: &str = "SELECT COUNT(*) AS event_count FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND session_ref = $session AND source_instance_ref = $source";
const SELECTED_TAIL_COUNT_STATEMENT: &str = "SELECT COUNT(*) AS event_count FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND session_ref = $session AND source_instance_ref = $source AND source_sequence > $frontier";

/// Events of one selected stream at or below `$frontier`: the stream's maintained exact-key total
/// less the tail above the frontier. The tail is pending work, so the count never walks the
/// completed history below it (ruling 164).
fn selected_events_through(mut count: impl FnMut(&str) -> Result<u64>) -> Result<u64> {
    let total = count(SELECTED_STREAM_COUNT_STATEMENT)?;
    let tail = count(SELECTED_TAIL_COUNT_STATEMENT)?;
    total
        .checked_sub(tail)
        .ok_or_else(|| stale_claim("selected stream tail exceeds its maintained total"))
}
const SOURCE_POSITION_SESSION_STATEMENT: &str = "SELECT session_ref FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND source_sequence = $start";
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
    certified: Option<CaptureCertificate>,
    started_at: TxId,
    attempt_tx: Option<TxId>,
    attempt_token: Option<uuid::Uuid>,
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

/// The Context revision and interpretation stamp a membership capture was certified against.
/// Completion reuses the capture while both are unchanged, and refuses to finalize past a
/// revision the capture never saw.
#[derive(Clone, Debug, PartialEq)]
struct CaptureCertificate {
    revision: TxId,
    stamp: JsonValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationMode {
    Building,
    ExtendReady,
}

struct Claim {
    publication_event_count: Option<u64>,
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
    prior_open_tail_members: Vec<uuid::Uuid>,
    /// The cursor's recorded open-tail passage exactly as read, for the commit-time cursor check.
    prior_open_tail_passage_id: Option<uuid::Uuid>,
    /// That passage when this claim reopens its tail: the only passage the claim may retire.
    reopened_tail_passage_id: Option<uuid::Uuid>,
    open_tail_events: Vec<ClaimedEvent>,
    superseded_tail: Vec<SupersededPassage>,
    session_ended: bool,
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
    next_open_tail_members: Vec<uuid::Uuid>,
}

struct PendingStream {
    publication_event_count: Option<u64>,
    through_sequence: Option<u64>,
    start_ordinal: u64,
    connector_id: String,
    source_instance_ref: String,
    session_ref: String,
    committed_sequence: u64,
    open_tail_start_sequence: Option<u64>,
    open_tail_members: Vec<uuid::Uuid>,
    open_tail_passage_id: Option<uuid::Uuid>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SessionKey {
    connector_id: String,
    session_ref: String,
}

pub(crate) struct SessionWorkSet {
    frontiers: Option<BTreeMap<(SessionKey, String), u64>>,
    rank: BTreeMap<SessionKey, usize>,
    events: BTreeMap<SessionKey, u64>,
    selected_events: u64,
}

impl SessionWorkSet {
    pub(crate) fn selected_events(&self) -> u64 {
        self.selected_events
    }

    fn rank(&self, connector_id: &str, session_ref: &str) -> Option<usize> {
        self.rank
            .get(&SessionKey {
                connector_id: connector_id.into(),
                session_ref: session_ref.into(),
            })
            .copied()
    }
}

enum ClaimResult {
    Work(Box<Claim>),
    Complete,
    ScopeComplete,
}

enum FinishResult {
    Continue(Box<Generation>),
    Rebuild,
    Ready(Box<EventSearchBuildReceipt>),
}

#[derive(Clone, Copy)]
struct ProjectionAttempt {
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    tx: TxId,
    token: uuid::Uuid,
}

#[derive(Default)]
struct RunObservations {
    attempt: Option<ProjectionAttempt>,
    embedding_invocations: u64,
    event_read: Duration,
    render_and_tokenize: Duration,
    embed: Duration,
    vector_insert: Duration,
    commit: Duration,
}

impl RunObservations {
    fn timings(&self) -> EventSearchBuildTimings {
        let ms = |duration: Duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
        EventSearchBuildTimings {
            event_read_ms: ms(self.event_read),
            render_and_tokenize_ms: ms(self.render_and_tokenize),
            embed_ms: ms(self.embed),
            vector_insert_ms: ms(self.vector_insert),
            commit_ms: ms(self.commit),
        }
    }
}

fn measured<T>(duration: &mut Duration, operation: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = operation();
    *duration += started.elapsed();
    result
}

pub(crate) fn catch_up_standalone(request: EventSearchRunRequest) -> EventSearchRunOutcome {
    let embedding_workers = request.embedding_workers;
    catch_up_standalone_with_policy_and_embedding_workers(
        request,
        EventSearchWorkPolicy::default(),
        move || embedding_workers,
    )
}

pub(crate) fn catch_up_standalone_with_policy(
    request: EventSearchRunRequest,
    policy: EventSearchWorkPolicy,
) -> EventSearchRunOutcome {
    let embedding_workers = request.embedding_workers;
    catch_up_standalone_with_policy_and_embedding_workers(request, policy, move || {
        embedding_workers
    })
}

pub(crate) fn catch_up_standalone_with_embedding_workers(
    request: EventSearchRunRequest,
    resolve_embedding_workers: impl FnMut() -> NonZeroUsize,
) -> EventSearchRunOutcome {
    catch_up_standalone_with_policy_and_embedding_workers(
        request,
        EventSearchWorkPolicy::default(),
        resolve_embedding_workers,
    )
}

pub(crate) fn catch_up_standalone_with_policy_and_embedding_workers(
    request: EventSearchRunRequest,
    policy: EventSearchWorkPolicy,
    resolve_embedding_workers: impl FnMut() -> NonZeroUsize,
) -> EventSearchRunOutcome {
    let limit = request.store.write_memory_limit_bytes;
    catch_up_standalone_with_batch_limits(request, policy, resolve_embedding_workers, |_| Ok(limit))
}

pub(crate) fn catch_up_standalone_with_batch_limits(
    mut request: EventSearchRunRequest,
    policy: EventSearchWorkPolicy,
    mut resolve_embedding_workers: impl FnMut() -> NonZeroUsize,
    mut resolve_write_memory_limit: impl FnMut(
        &dyn Fn(EventSearchWorkPolicy) -> Result<bool>,
    ) -> Result<NonZeroUsize>,
) -> EventSearchRunOutcome {
    let mut observations = RunObservations::default();
    let mut result = validate_profile_for_registration(&request.profile, &request.embedder)
        .and_then(|()| {
            catch_up_standalone_run(
                &mut request,
                policy,
                &mut resolve_embedding_workers,
                &mut resolve_write_memory_limit,
                None,
                None,
                &mut observations,
            )
        });
    let failure_disposition = match &result {
        Ok(_) => EventSearchFailureDisposition::NoFailure,
        Err(error) => persist_run_failure(
            &request.store,
            &request.embedder,
            observations.attempt,
            error,
        ),
    };
    let timings = observations.timings();
    if let Ok(receipt) = &mut result {
        receipt.timings = timings.clone();
    }
    EventSearchRunOutcome {
        result,
        failure_disposition,
        embedding_invocations: observations.embedding_invocations,
        timings,
    }
}

pub(crate) fn catch_up_standalone_with_selected_work(
    mut request: EventSearchRunRequest,
    policy: EventSearchWorkPolicy,
    selected_work: EventSearchSelectedWork,
    mut resolve_embedding_workers: impl FnMut() -> NonZeroUsize,
    mut resolve_selected_limit: impl FnMut(
        Option<EventSearchPendingBatch<'_>>,
    ) -> Result<EventSearchBatchWriteLimit>,
) -> EventSearchRunOutcome {
    let request_limit = request.store.write_memory_limit_bytes;
    let mut observations = RunObservations::default();
    let mut result = validate_profile_for_registration(&request.profile, &request.embedder)
        .and_then(|()| {
            catch_up_standalone_run(
                &mut request,
                policy,
                &mut resolve_embedding_workers,
                &mut |_| Ok(request_limit),
                Some(&selected_work),
                Some(&mut resolve_selected_limit),
                &mut observations,
            )
        });
    let failure_disposition = match &result {
        Ok(_) => EventSearchFailureDisposition::NoFailure,
        Err(error) => persist_run_failure(
            &request.store,
            &request.embedder,
            observations.attempt,
            error,
        ),
    };
    let timings = observations.timings();
    if let Ok(receipt) = &mut result {
        receipt.timings = timings.clone();
    }
    EventSearchRunOutcome {
        result,
        failure_disposition,
        embedding_invocations: observations.embedding_invocations,
        timings,
    }
}

/// The reason status should show for a run that stopped. The text names a condition only;
/// contention and general refusals use fixed phrases, while core engine errors retain their
/// already-sanitized reason.
fn sanitized_failure(error: &CgError) -> String {
    match error {
        CgError::StoreHeldByReaders { .. } => HELD_BY_READERS_FAILURE.into(),
        CgError::StoreLocked { .. } => HELD_BY_WRITER_FAILURE.into(),
        CgError::Engine(message) => message.clone(),
        CgError::WriteDidNotFitTheMemoryLimit { .. } => error.to_string(),
        CgError::ReadActiveTimeExceeded { .. } => format!(
            "{error}; committed search progress is retained; run the same import with --search wait again"
        ),
        _ => GENERIC_PROJECTION_FAILURE.into(),
    }
}

fn persist_run_failure(
    config: &crate::StoreConfig,
    embedder: &crate::EmbedderRegistration,
    attempt: Option<ProjectionAttempt>,
    error: &CgError,
) -> EventSearchFailureDisposition {
    if is_stale_claim(error) {
        return EventSearchFailureDisposition::Stale;
    }
    let Some(attempt) = attempt else {
        return EventSearchFailureDisposition::Unrecorded;
    };
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    BEFORE_FAILURE_RECORD.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
    let Ok(store) = open_for_run(config, embedder) else {
        return EventSearchFailureDisposition::Unrecorded;
    };
    match record_run_failure(&store, attempt, &sanitized_failure(error)) {
        Ok(disposition) => disposition,
        Err(_) => EventSearchFailureDisposition::Unrecorded,
    }
}

/// Only the attempt that still owns this marker may publish a failure. Coverage and generation
/// creation time do not change, and a late result cannot overwrite a newer attempt's outcome.
fn record_run_failure(
    store: &Store,
    attempt: ProjectionAttempt,
    reason: &str,
) -> Result<EventSearchFailureDisposition> {
    let updated = store
        .internal_db()
        .execute(
            "UPDATE event_search_generations SET sanitized_failure = $failure WHERE context_id = $context_id AND id = $id AND last_attempt_at = $attempt AND last_attempt_token = $attempt_token",
            &HashMap::from([
                ("failure".into(), Value::Text(reason.to_string())),
                ("context_id".into(), Value::Uuid(attempt.context_id.as_uuid())),
                ("id".into(), Value::Uuid(attempt.generation_id.as_uuid())),
                ("attempt".into(), Value::TxId(attempt.tx)),
                ("attempt_token".into(), Value::Uuid(attempt.token)),
            ]),
        )
        .map_err(map_engine_err)?;
    match updated.rows_affected {
        1 => Ok(EventSearchFailureDisposition::Recorded),
        0 => Ok(EventSearchFailureDisposition::Stale),
        count => Err(CgError::Engine(format!(
            "event-search failure recording changed {count} generation rows"
        ))),
    }
}

/// Claim an attempt before resuming metadata work. A live claim is somebody else's; an expired
/// claim and the old attempt marker are guarded in the same transaction that clears its failure.
fn begin_generation_attempt(
    store: &Store,
    context_id: crate::ContextId,
    id: EventSearchGenerationId,
) -> Result<ProjectionAttempt> {
    write_generation_attempt(store, context_id, id, None)
}

/// Record a caller-side refusal as one atomic metadata attempt. If admission or commit is denied,
/// the previous failure remains untouched.
fn record_refusal_attempt(
    store: &Store,
    context_id: crate::ContextId,
    id: EventSearchGenerationId,
    reason: &str,
) -> Result<()> {
    write_generation_attempt(store, context_id, id, Some(reason)).map(|_| ())
}

fn write_generation_attempt(
    store: &Store,
    context_id: crate::ContextId,
    id: EventSearchGenerationId,
    failure: Option<&str>,
) -> Result<ProjectionAttempt> {
    let attempt_token = uuid::Uuid::now_v7();
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        let params = HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("id".into(), Value::Uuid(id.as_uuid())),
        ]);
        let generation = store.internal_db().execute_in_tx(tx,
            "SELECT last_attempt_at, last_attempt_token FROM event_search_generations WHERE context_id = $context_id AND id = $id", &params).map_err(map_engine_err)?;
        let row = generation
            .rows
            .first()
            .ok_or_else(|| stale_claim("event-search generation disappeared before attempt"))?;
        let conditions = exact_conditions(
            row,
            &generation.columns,
            &["last_attempt_at", "last_attempt_token"],
        )?;
        if !store
            .internal_db()
            .guard_row_conditions_in_tx(
                tx,
                "event_search_generations",
                "id",
                &Value::Uuid(id.as_uuid()),
                &conditions,
            )
            .map_err(map_engine_err)?
        {
            return Err(stale_claim(
                "event-search attempt changed before resumption",
            ));
        }
        let lease = store.internal_db().execute_in_tx(tx,
            "SELECT owner_token, lease_epoch, expires_at FROM event_search_leases WHERE context_id = $context_id AND generation_id = $id", &params).map_err(map_engine_err)?;
        if let Some(row) = lease.rows.first() {
            if get_timestamp(row, &lease.columns, "expires_at")? > now_millis()? {
                return Err(stale_claim(
                    "another event-search worker owns the unexpired claim",
                ));
            }
            let conditions = exact_conditions(
                row,
                &lease.columns,
                &["owner_token", "lease_epoch", "expires_at"],
            )?;
            if !store
                .internal_db()
                .guard_row_conditions_in_tx(
                    tx,
                    "event_search_leases",
                    "generation_id",
                    &Value::Uuid(id.as_uuid()),
                    &conditions,
                )
                .map_err(map_engine_err)?
            {
                return Err(stale_claim("event-search claim changed before resumption"));
            }
        }
        let mut params = params;
        params.insert("attempt".into(), Value::TxId(tx));
        params.insert("attempt_token".into(), Value::Uuid(attempt_token));
        params.insert(
            "failure".into(),
            failure.map_or(Value::Null, |reason| Value::Text(reason.to_string())),
        );
        let updated = store.internal_db().execute_in_tx(tx,
            "UPDATE event_search_generations SET last_attempt_at = $attempt, last_attempt_token = $attempt_token, sanitized_failure = $failure WHERE context_id = $context_id AND id = $id", &params).map_err(map_engine_err)?;
        if updated.rows_affected != 1 {
            return Err(stale_claim(
                "event-search generation disappeared while its attempt was recorded",
            ));
        }
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) = BEFORE_ATTEMPT_COMMIT.with(|slot| slot.borrow_mut().take()) {
        hook(store);
    }
    store
        .internal_db()
        .commit(tx)
        .map_err(|error| match error {
            EngineError::ConditionalUpdateConflict { .. } => {
                stale_claim("event-search attempt changed before resumption committed")
            }
            other => map_engine_err(other),
        })?;
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) = AFTER_ATTEMPT_COMMIT_BEFORE_READBACK.with(|slot| slot.borrow_mut().take()) {
        hook(store);
    }
    read_committed_attempt(store, context_id, id, attempt_token).map(|(attempt, _)| attempt)
}

/// Read back only the attempt carrying this caller's unrewritable token. A racing owner makes
/// this attempt stale; it can never cause the caller to adopt that owner's timestamp.
fn read_committed_attempt(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    token: uuid::Uuid,
) -> Result<(ProjectionAttempt, TxId)> {
    let result = store
        .internal_db()
        .execute(
            "SELECT started_at, last_attempt_at FROM event_search_generations WHERE context_id = $context_id AND id = $id AND last_attempt_token = $attempt_token",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("id".into(), Value::Uuid(generation_id.as_uuid())),
                ("attempt_token".into(), Value::Uuid(token)),
            ]),
        )
        .map_err(map_engine_err)?;
    let row = result.rows.first().ok_or_else(|| {
        stale_claim("event-search attempt was superseded before committed timestamp readback")
    })?;
    if result.rows.len() != 1 {
        return Err(stale_claim(
            "event-search attempt ownership token was not unique",
        ));
    }
    Ok((
        ProjectionAttempt {
            context_id,
            generation_id,
            tx: get_txid(row, &result.columns, "last_attempt_at")?,
            token,
        },
        get_txid(row, &result.columns, "started_at")?,
    ))
}

fn read_committed_completion(store: &Store, generation: &Generation) -> Result<TxId> {
    let Some(attempt_tx) = generation.attempt_tx else {
        return Err(stale_claim(
            "event-search generation has no completed attempt",
        ));
    };
    let Some(attempt_token) = generation.attempt_token else {
        return Err(stale_claim(
            "event-search generation has no completed attempt ownership token",
        ));
    };
    let result = store
        .internal_db()
        .execute(
            "SELECT completed_at FROM event_search_generations WHERE context_id = $context_id AND id = $id AND last_attempt_at = $attempt AND last_attempt_token = $attempt_token",
            &HashMap::from([
                (
                    "context_id".into(),
                    Value::Uuid(generation.context_id.as_uuid()),
                ),
                ("id".into(), Value::Uuid(generation.id.as_uuid())),
                ("attempt".into(), Value::TxId(attempt_tx)),
                ("attempt_token".into(), Value::Uuid(attempt_token)),
            ]),
        )
        .map_err(map_engine_err)?;
    let row = result.rows.first().ok_or_else(|| {
        stale_claim("event-search attempt was superseded before completion timestamp readback")
    })?;
    if result.rows.len() != 1 {
        return Err(stale_claim(
            "event-search completion ownership token was not unique",
        ));
    }
    get_optional_txid(row, &result.columns, "completed_at")?.ok_or_else(|| {
        stale_claim("event-search ready generation has no committed completion transaction")
    })
}

type SelectedBatchLimitResolver<'a> =
    dyn FnMut(Option<EventSearchPendingBatch<'_>>) -> Result<EventSearchBatchWriteLimit> + 'a;

type BatchWriteMemoryLimitResolver<'a> =
    dyn FnMut(&dyn Fn(EventSearchWorkPolicy) -> Result<bool>) -> Result<NonZeroUsize> + 'a;

fn catch_up_standalone_run(
    request: &mut EventSearchRunRequest,
    work_policy: EventSearchWorkPolicy,
    resolve_embedding_workers: &mut dyn FnMut() -> NonZeroUsize,
    resolve_write_memory_limit: &mut BatchWriteMemoryLimitResolver<'_>,
    selected_work: Option<&EventSearchSelectedWork>,
    mut resolve_selected_limit: Option<&mut SelectedBatchLimitResolver<'_>>,
    observations: &mut RunObservations,
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
    // Every writable open costs memory in the engine today, so each open does as much as it can
    // before it is dropped: the open that prepares the generation takes the first claim, and the
    // open that commits or releases a claim takes the next. The Store is still never held while
    // rendering and embedding.
    request.store.write_memory_limit_bytes = match resolve_selected_limit.as_mut() {
        Some(resolve) => resolve(None)?.memory_limit_bytes,
        None => resolve_write_memory_limit(&|_| Ok(true))?,
    };
    let store = open_for_run(&request.store, &request.embedder)?;
    let mut generation = measured(&mut observations.event_read, || {
        ensure_generation(
            &store,
            context_id,
            &request.embedder,
            &request.profile,
            bounds,
            false,
            InterpretationSelection::Ordinary,
            &mut observations.attempt,
        )
    })?;
    persist_work_policy(&store, context_id, work_policy)?;
    let mut session_work = selected_session_work_for_run(
        &store,
        context_id,
        generation.id,
        work_policy,
        selected_work,
    )?;
    let mut open = Some(store);

    let mut retries = 0_u64;
    let mut event_limit = bounds.events;
    loop {
        let mut store = match open.take() {
            Some(store) => store,
            None => open_for_run(&request.store, &request.embedder)?,
        };
        let mut pending = measured(&mut observations.event_read, || {
            first_pending_stream(&store, &generation, &session_work, bounds)
        })?;
        let limit = if let Some(resolve) = resolve_selected_limit.as_mut() {
            let applies = |request_id| match &pending {
                Some(pending) => selected_frontier(&store, context_id, request_id, pending),
                None => Ok(None),
            };
            let admitted = resolve(pending.as_ref().map(|pending| EventSearchPendingBatch {
                connector_id: &pending.connector_id,
                session_ref: &pending.session_ref,
                source_instance_ref: &pending.source_instance_ref,
                committed_sequence: pending.committed_sequence,
                selected_frontier: &applies,
            }))?;
            if let Some(pending) = &mut pending {
                pending.through_sequence =
                    match (pending.through_sequence, admitted.through_sequence) {
                        (Some(selected), Some(admitted)) => Some(selected.min(admitted)),
                        (selected, admitted) => selected.or(admitted),
                    };
            }
            admitted.memory_limit_bytes
        } else if let Some(pending) = &pending {
            resolve_write_memory_limit(&|policy| {
                let primary = selected_captured_session_work(&store, context_id, work_policy)?;
                let inherited = primary
                    .rank(&pending.connector_id, &pending.session_ref)
                    .is_none();
                Ok(inherited
                    || selected_captured_session_work(&store, context_id, policy)?
                        .rank(&pending.connector_id, &pending.session_ref)
                        .is_some())
            })?
        } else {
            resolve_write_memory_limit(&|_| Ok(true))?
        };
        if limit != request.store.write_memory_limit_bytes {
            drop(store);
            request.store.write_memory_limit_bytes = limit;
            store = open_for_run(&request.store, &request.embedder)?;
        }
        #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
        if let Some(pending) = &pending {
            BATCH_ADMISSION_HOOK.with(|slot| {
                if let Some(hook) = slot.borrow().as_ref() {
                    hook(
                        &pending.session_ref,
                        &pending.source_instance_ref,
                        pending.committed_sequence,
                        pending.through_sequence,
                        store
                            .internal_db()
                            .memory_limit()
                            .expect("projection writer has a finite budget"),
                    );
                }
            });
        }
        let claim_result = measured(&mut observations.event_read, || {
            claim_pending_batch(
                &store,
                &generation,
                bounds,
                event_limit,
                pending,
                selected_work.and_then(|work| work.request_id),
            )
        })?;
        let ClaimResult::Work(claim) = claim_result else {
            let outcome = finish_or_continue(
                &store,
                &generation,
                &request.embedder,
                &request.profile,
                bounds,
                InterpretationSelection::Ordinary,
                retries,
                observations,
            );
            let outcome = match outcome {
                Ok(FinishResult::Rebuild) => measured(&mut observations.event_read, || {
                    ensure_generation(
                        &store,
                        context_id,
                        &request.embedder,
                        &request.profile,
                        bounds,
                        true,
                        InterpretationSelection::Ordinary,
                        &mut observations.attempt,
                    )
                })
                .map(|generation| FinishResult::Continue(Box::new(generation))),
                Ok(FinishResult::Ready(receipt)) => {
                    maintain_passage_vectors(&store).map(|()| FinishResult::Ready(receipt))
                }
                other => other,
            };
            match outcome? {
                FinishResult::Continue(next) => {
                    generation = *next;
                    session_work = selected_session_work_for_run(
                        &store,
                        context_id,
                        generation.id,
                        work_policy,
                        selected_work,
                    )?;
                    open = Some(store);
                    continue;
                }
                FinishResult::Ready(receipt) => return Ok(*receipt),
                FinishResult::Rebuild => unreachable!("rebuild handled while Store was open"),
            }
        };
        drop(store);
        let embedding_workers = resolve_embedding_workers();

        // From here the run holds a claim, so every way out of it releases that claim first.
        let held = HeldClaim::of(&claim);
        // No Store exists in this scope: renderer/tokenizer/embedder execution is deliberately
        // between the claim close and the validated commit reopen.
        let mut renew = |expires_at| {
            if !renewal_due(expires_at, bounds)? {
                return Ok(expires_at);
            }
            let store = open_for_run(&request.store, &request.embedder)?;
            renew_claim(&store, &held, expires_at, bounds)
        };
        let rendered = render_and_embed(
            *claim,
            &request.embedder,
            &request.profile,
            bounds,
            embedding_workers,
            &mut renew,
            observations,
        );
        let built = match rendered {
            Ok(Rendered::Batch(built)) => *built,
            Ok(Rendered::OverPassageBound { fitting_events }) => {
                // Nothing was embedded: let the claim go and take the events that fit.
                let store = open_for_run(&request.store, &request.embedder)?;
                release_claim_in(&store, &held);
                event_limit = fitting_events;
                open = Some(store);
                continue;
            }
            Err(error) if retries < MAX_STALE_RETRIES as u64 && is_stale_claim(&error) => {
                // The claim was taken over while this run embedded; nothing is left to release.
                retries += 1;
                continue;
            }
            Err(error) => {
                release_claim(&request.store, &request.embedder, &held);
                return Err(error);
            }
        };
        let rendered_passages = built.passages.len();
        let store = match open_for_run(&request.store, &request.embedder) {
            Ok(store) => store,
            Err(error) => {
                release_claim(&request.store, &request.embedder, &held);
                return Err(error);
            }
        };
        if let Err(error) = commit_batch(
            &store,
            &request.embedder,
            &request.profile,
            bounds,
            built,
            observations,
        ) {
            // A conditional commit conflict can make this claim stale while its lease row is
            // still present. Release only this run's token and epoch before taking a fresh claim;
            // a real takeover remains untouched because the exact delete no longer matches.
            if retries < MAX_STALE_RETRIES as u64 && is_stale_claim(&error) {
                try_release_claim_in(&store, &held)?;
                retries += 1;
                open = Some(store);
                continue;
            }
            release_claim_in(&store, &held);
            return Err(error);
        }
        event_limit = next_event_limit(event_limit, rendered_passages, bounds);
        open = Some(store);
    }
}

/// The event limit for the claim after one that committed `rendered_passages` passages from
/// `event_limit` events. A narrowed limit is kept rather than reset to the event bound, since
/// resetting makes every claim overflow the passage bound again and pay a refused render and a
/// store reopen; it grows back only while claims render under half the passage bound.
fn next_event_limit(event_limit: usize, rendered_passages: usize, bounds: Bounds) -> usize {
    if rendered_passages.saturating_mul(2) <= bounds.passages {
        event_limit.saturating_mul(2).min(bounds.events).max(1)
    } else {
        event_limit
    }
}

pub(crate) fn catch_up_store(
    store: &Store,
    request: EventSearchBuildRequest,
) -> Result<EventSearchBuildReceipt> {
    catch_up_store_with_policy(store, request, EventSearchWorkPolicy::default())
}

pub(crate) fn catch_up_store_with_policy(
    store: &Store,
    request: EventSearchBuildRequest,
    work_policy: EventSearchWorkPolicy,
) -> Result<EventSearchBuildReceipt> {
    let embedding_workers = request.embedding_workers;
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
        embedding_workers,
        work_policy,
        false,
        InterpretationSelection::Ordinary,
    )
}

pub(crate) fn rebuild_store(
    store: &Store,
    request: EventSearchRebuildRequest,
) -> Result<EventSearchBuildReceipt> {
    rebuild_store_with_policy(store, request, EventSearchWorkPolicy::default())
}

pub(crate) fn rebuild_store_with_policy(
    store: &Store,
    request: EventSearchRebuildRequest,
    work_policy: EventSearchWorkPolicy,
) -> Result<EventSearchBuildReceipt> {
    let embedding_workers = request.embedding_workers;
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
        embedding_workers,
        work_policy,
        true,
        selection,
    )
}

/// Coordinates a held Store's event-search run from the caller's validated request fields.
///
/// Each argument is an independently validated part of the coordinator's request; grouping them
/// would obscure the standalone and rebuild call paths that supply the same run contract.
#[allow(clippy::too_many_arguments)]
fn run_with_store(
    store: &Store,
    context_id: crate::ContextId,
    embedder: crate::EmbedderRegistration,
    profile: EventSearchProfile,
    bounds: Bounds,
    embedding_workers: NonZeroUsize,
    work_policy: EventSearchWorkPolicy,
    force_rebuild: bool,
    selection: InterpretationSelection,
) -> Result<EventSearchBuildReceipt> {
    let mut observations = RunObservations::default();
    let result = (|| {
        validate_profile_for_registration(&profile, &embedder)?;
        let mut generation = measured(&mut observations.event_read, || {
            ensure_generation(
                store,
                context_id,
                &embedder,
                &profile,
                bounds,
                force_rebuild,
                selection,
                &mut observations.attempt,
            )
        })?;
        persist_work_policy(store, context_id, work_policy)?;
        let mut session_work =
            selected_session_work(store, context_id, generation.id, work_policy)?;
        let mut retries = 0_u64;
        let mut event_limit = bounds.events;
        loop {
            let claim_result = measured(&mut observations.event_read, || {
                claim_batch(store, &generation, &session_work, bounds, event_limit)
            })?;
            let ClaimResult::Work(claim) = claim_result else {
                match finish_or_continue(
                    store,
                    &generation,
                    &embedder,
                    &profile,
                    bounds,
                    selection,
                    retries,
                    &mut observations,
                )? {
                    FinishResult::Continue(next) => {
                        generation = *next;
                        session_work =
                            selected_session_work(store, context_id, generation.id, work_policy)?;
                        continue;
                    }
                    FinishResult::Rebuild => {
                        generation = measured(&mut observations.event_read, || {
                            ensure_generation(
                                store,
                                context_id,
                                &embedder,
                                &profile,
                                bounds,
                                true,
                                selection,
                                &mut observations.attempt,
                            )
                        })?;
                        session_work =
                            selected_session_work(store, context_id, generation.id, work_policy)?;
                        continue;
                    }
                    FinishResult::Ready(receipt) => {
                        maintain_passage_vectors(store)?;
                        return Ok(*receipt);
                    }
                }
            };
            // The caller necessarily retains `store`; this wrapper makes no Store-release claim.
            let held = HeldClaim::of(&claim);
            let mut renew = |expires_at| {
                if !renewal_due(expires_at, bounds)? {
                    return Ok(expires_at);
                }
                renew_claim(store, &held, expires_at, bounds)
            };
            let built = match render_and_embed(
                *claim,
                &embedder,
                &profile,
                bounds,
                embedding_workers,
                &mut renew,
                &mut observations,
            ) {
                Ok(Rendered::Batch(built)) => *built,
                Ok(Rendered::OverPassageBound { fitting_events }) => {
                    release_claim_in(store, &held);
                    event_limit = fitting_events;
                    continue;
                }
                Err(error) if retries < MAX_STALE_RETRIES as u64 && is_stale_claim(&error) => {
                    retries += 1;
                    continue;
                }
                Err(error) => {
                    release_claim_in(store, &held);
                    return Err(error);
                }
            };
            let rendered_passages = built.passages.len();
            match commit_batch(store, &embedder, &profile, bounds, built, &mut observations) {
                Ok(()) => event_limit = next_event_limit(event_limit, rendered_passages, bounds),
                Err(error) if retries < MAX_STALE_RETRIES as u64 && is_stale_claim(&error) => {
                    try_release_claim_in(store, &held)?;
                    retries += 1;
                }
                Err(error) => {
                    release_claim_in(store, &held);
                    return Err(error);
                }
            }
        }
    })();
    if let (Err(error), Some(attempt)) = (&result, observations.attempt)
        && !is_stale_claim(error)
    {
        let _ = record_run_failure(store, attempt, &sanitized_failure(error));
    }
    result.map(|mut receipt| {
        receipt.timings = observations.timings();
        receipt
    })
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
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
impl Store {
    #[doc(hidden)]
    pub fn maintain_event_search_vectors_for_test(&self) -> Result<()> {
        maintain_passage_vectors(self)
    }
}

/// The full membership verification of one event-search generation.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventSearchGenerationVerification {
    pub member_count: u64,
    pub ordered_source_set_sha256: String,
    pub source_content_sha256: String,
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
impl Store {
    /// Recompute a generation's membership chain from every persisted member's fields, refuse any
    /// member whose stored chain value differs, and refuse a generation whose carried digests or
    /// capture ordinal differ from the recomputed chain. Ordinary capture never runs this full walk;
    /// every full capture runs the same fold. Test support only: the Store's public surface is
    /// pinned (tests/acceptance/nomic_loads.rs).
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    pub fn verify_event_search_generation(
        &self,
        context_id: crate::ContextId,
        generation_id: uuid::Uuid,
    ) -> Result<EventSearchGenerationVerification> {
        let generation_id = EventSearchGenerationId::from(generation_id);
        let bounds = bounds(1024, 1024, usize::MAX, 60_000);
        let folded = fold_source_membership(self, context_id, generation_id, bounds)?;
        let generation = query_rows(
            self,
            bounds,
            "SELECT capture_ordinal, capture_ordered_sha256, capture_content_sha256 FROM event_search_generations WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("id".into(), Value::Uuid(generation_id.as_uuid())),
            ]),
        )?;
        let row = generation.rows.first().ok_or(CgError::NotFound {
            id: Some(generation_id.as_uuid().to_string()),
            missing: Some("event-search generation"),
        })?;
        let capture_ordinal = get_u64(row, &generation.columns, "capture_ordinal")?;
        let carried = MembershipChain::carried(
            get_text(row, &generation.columns, "capture_ordered_sha256")?,
            get_text(row, &generation.columns, "capture_content_sha256")?,
            capture_ordinal,
        );
        let recomputed = MembershipChain {
            ordered: folded.ordered_source_set_sha256.clone(),
            content: folded.source_content_sha256.clone(),
        };
        if capture_ordinal != folded.event_count || carried.as_ref() != Some(&recomputed) {
            return Err(CgError::Engine(format!(
                "{MEMBERSHIP_CHAIN_CORRUPT}generation {} carries digests or a capture ordinal that differ from its members",
                generation_id.as_uuid()
            )));
        }
        Ok(EventSearchGenerationVerification {
            member_count: folded.event_count,
            ordered_source_set_sha256: folded.ordered_source_set_sha256,
            source_content_sha256: folded.source_content_sha256,
        })
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

/// The one metadata selection shared by work assessment and generation preparation.
fn select_generation(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    snapshot: &JsonValue,
    snapshot_sha256: &str,
    spec: &EventSearchWorkSpec,
    allow_active: bool,
) -> Result<Option<(Generation, bool)>> {
    if allow_active
        && let Some(active) =
            current_ready_head(source, context_id, snapshot, snapshot_sha256, spec)?
    {
        return Ok(Some((active, false)));
    }
    incomplete_generation(source, context_id, snapshot, snapshot_sha256, spec)
}

pub(crate) fn persist_selected_work(
    store: &Store,
    context_id: crate::ContextId,
    selected: &EventSearchSelectedWork,
) -> Result<uuid::Uuid> {
    let id = uuid::Uuid::new_v4();
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        store.internal_db().execute_in_tx(tx, "INSERT INTO event_search_request_headers (context_id, request_id) VALUES ($context_id, $request_id)",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("request_id".into(), Value::Uuid(id))])).map_err(map_engine_err)?;
        for (ordinal, stream) in selected.streams.iter().enumerate() {
            store.internal_db().execute_in_tx(tx,
                "INSERT INTO event_search_request_selections (context_id, request_id, ordinal, connector_id, session_ref, source_instance_ref, frontier) VALUES ($context_id, $request_id, $ordinal, $connector, $session, $source, $frontier)",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("request_id".into(), Value::Uuid(id)),
                    ("ordinal".into(), integer(ordinal as u64, "selection ordinal")?),
                    ("connector".into(), Value::Text(stream.connector_id.clone())),
                    ("session".into(), Value::Text(stream.session_ref.clone())),
                    ("source".into(), Value::Text(stream.source_instance_ref.clone())),
                    ("frontier".into(), integer(stream.frontier, "selection frontier")?),
                ]),
            ).map_err(map_engine_err)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)?;
    Ok(id)
}

/// Freeze saved frontiers only when a differing in-flight profile can publish them later.
/// Same-profile queued work keeps its own selection and ceiling without broadening this request.
/// The accepting caller supplies identities from its bounded queue after writable admission.
pub(crate) fn persist_selected_work_with_retained_requests(
    store: &Store,
    context_id: crate::ContextId,
    selected: &EventSearchSelectedWork,
    retained_requests: &[uuid::Uuid],
    desired_spec: &EventSearchWorkSpec,
) -> Result<uuid::Uuid> {
    desired_spec.profile.validate()?;
    let desired_profile = serde_json::to_value(&desired_spec.profile).map_err(json_error)?;
    let mut running_requests = BTreeSet::new();
    // The lease and header association identify the exact saved run, including streams it
    // has not claimed yet. Both are committed in the same batch-claim transaction.
    store.stream_rows(
        "SELECT generation_id, owner_token FROM event_search_leases WHERE context_id = $context_id",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        &mut |columns, row| {
            let generation_id = get_uuid(row, columns, "generation_id")?;
            let owner_token = get_text(row, columns, "owner_token")?;
            let generation = store.query_rows(
                "SELECT renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile FROM event_search_generations WHERE context_id = $context_id AND id = $id",
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("id".into(), Value::Uuid(generation_id))]),
            )?;
            let Some(generation_row) = generation.rows.first() else {
                return Err(stale_claim("queued retention generation disappeared"));
            };
            if get_text(generation_row, &generation.columns, "renderer_id")? == EVENT_SEARCH_RENDERER_ID
                && get_text(generation_row, &generation.columns, "renderer_version")? == EVENT_SEARCH_RENDERER_VERSION
                && get_json(generation_row, &generation.columns, "renderer_profile")? == desired_profile
                && get_text(generation_row, &generation.columns, "embedding_space_id")? == desired_spec.embedding_space_id
                && get_text(generation_row, &generation.columns, "encoding_profile")? == desired_spec.encoding_profile_id
            {
                return Ok(());
            }
            for request_id in retained_requests {
                let header = store.query_rows(
                    "SELECT request_id FROM event_search_request_headers WHERE context_id = $context_id AND request_id = $request_id AND running_generation_id = $generation_id AND running_owner_token = $owner_token",
                    &HashMap::from([
                        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                        ("request_id".into(), Value::Uuid(*request_id)),
                        ("generation_id".into(), Value::Uuid(generation_id)),
                        ("owner_token".into(), Value::Text(owner_token.clone())),
                    ]),
                )?;
                if !header.rows.is_empty() {
                    running_requests.insert(*request_id);
                }
            }
            Ok(())
        },
    )?;
    let mut merged = selected.clone();
    for request_id in running_requests {
        for stream in load_selected_work(store, context_id, request_id)?.streams {
            if let Some(existing) = merged.streams.iter_mut().find(|existing| {
                existing.connector_id == stream.connector_id
                    && existing.session_ref == stream.session_ref
                    && existing.source_instance_ref == stream.source_instance_ref
            }) {
                existing.frontier = existing.frontier.max(stream.frontier);
            } else {
                merged.streams.push(stream);
            }
        }
    }
    persist_selected_work(store, context_id, &merged)
}

pub(crate) fn load_selected_work(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    request_id: uuid::Uuid,
) -> Result<EventSearchSelectedWork> {
    load_work_frontiers(source, context_id, request_id, false)
}

pub(crate) fn load_refusal_work(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    request_id: uuid::Uuid,
) -> Result<EventSearchSelectedWork> {
    load_work_frontiers(source, context_id, request_id, true)
}

fn load_work_frontiers(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    request_id: uuid::Uuid,
    refusal: bool,
) -> Result<EventSearchSelectedWork> {
    let (headers, frontiers) = if refusal {
        (
            "event_search_refusal_headers",
            "event_search_refusal_frontiers",
        )
    } else {
        (
            "event_search_request_headers",
            "event_search_request_selections",
        )
    };
    let header = source.query_rows(&format!("SELECT request_id FROM {headers} WHERE context_id = $context_id AND request_id = $request_id"),
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("request_id".into(), Value::Uuid(request_id))]))?;
    if header.rows.is_empty() {
        return Err(CgError::Engine(
            if refusal {
                "retired search refusal has no durable selected-work identity"
            } else {
                "pending search request has no durable selected-work identity"
            }
            .into(),
        ));
    }
    let mut streams = Vec::new();
    source.stream_rows(
        &format!("SELECT connector_id, session_ref, source_instance_ref, frontier FROM {frontiers} WHERE context_id = $context_id AND request_id = $request_id ORDER BY ordinal ASC"),
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("request_id".into(), Value::Uuid(request_id))]),
        &mut |columns, row| {
            streams.push(EventSearchSelectedStream { connector_id: get_text(row, columns, "connector_id")?, session_ref: get_text(row, columns, "session_ref")?, source_instance_ref: get_text(row, columns, "source_instance_ref")?, frontier: get_u64(row, columns, "frontier")? });
            Ok(())
        },
    )?;
    Ok(EventSearchSelectedWork {
        request_id: Some(request_id),
        streams,
    })
}

/// Copy completion evidence before runnable retirement. A crash between commits leaves the
/// original selection and/or its immutable evidence available for the next cleanup attempt.
pub(crate) fn preserve_refusal_work(
    store: &Store,
    context_id: crate::ContextId,
    request_id: uuid::Uuid,
) -> Result<()> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("request_id".into(), Value::Uuid(request_id)),
    ]);
    if !store.query_rows(
        "SELECT request_id FROM event_search_refusal_headers WHERE context_id = $context_id AND request_id = $request_id", &params,
    )?.rows.is_empty() {
        return Ok(());
    }
    let selected = load_selected_work(store, context_id, request_id)?;
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        store.internal_db().execute_in_tx(tx,
            "INSERT INTO event_search_refusal_headers (context_id, request_id) VALUES ($context_id, $request_id)", &params,
        ).map_err(map_engine_err)?;
        for (ordinal, stream) in selected.streams.iter().enumerate() {
            store.internal_db().execute_in_tx(tx,
                "INSERT INTO event_search_refusal_frontiers (context_id, request_id, ordinal, connector_id, session_ref, source_instance_ref, frontier) VALUES ($context_id, $request_id, $ordinal, $connector, $session, $source, $frontier)",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("request_id".into(), Value::Uuid(request_id)),
                    ("ordinal".into(), integer(ordinal as u64, "refusal ordinal")?),
                    ("connector".into(), Value::Text(stream.connector_id.clone())),
                    ("session".into(), Value::Text(stream.session_ref.clone())),
                    ("source".into(), Value::Text(stream.source_instance_ref.clone())),
                    ("frontier".into(), integer(stream.frontier, "refusal frontier")?),
                ]),
            ).map_err(map_engine_err)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)
}

pub(crate) fn release_selected_work(
    store: &Store,
    context_id: crate::ContextId,
    requests: &[uuid::Uuid],
) -> Result<()> {
    release_work_frontiers(store, context_id, requests, false)
}

fn release_work_frontiers(
    store: &Store,
    context_id: crate::ContextId,
    requests: &[uuid::Uuid],
    refusal: bool,
) -> Result<()> {
    let (headers, frontiers) = if refusal {
        (
            "event_search_refusal_headers",
            "event_search_refusal_frontiers",
        )
    } else {
        (
            "event_search_request_headers",
            "event_search_request_selections",
        )
    };
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        for id in requests {
            store.internal_db().execute_in_tx(tx,
                &format!("DELETE FROM {frontiers} WHERE context_id = $context_id AND request_id = $request_id"),
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("request_id".into(), Value::Uuid(*id))]),
            ).map_err(map_engine_err)?;
            store.internal_db().execute_in_tx(tx, &format!("DELETE FROM {headers} WHERE context_id = $context_id AND request_id = $request_id"),
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("request_id".into(), Value::Uuid(*id))])).map_err(map_engine_err)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)
}

/// Reclaim unpublished or acknowledged selections while the caller owns the writable Store.
/// The supplied identities come from the bounded external queue after writable admission.
pub(crate) fn reconcile_selected_work(store: &Store, retained: &[uuid::Uuid]) -> Result<()> {
    reconcile_work_frontiers(store, retained, false)
}

pub(crate) fn reconcile_refusal_work(store: &Store, retained: &[uuid::Uuid]) -> Result<()> {
    reconcile_work_frontiers(store, retained, true)
}

fn reconcile_work_frontiers(store: &Store, retained: &[uuid::Uuid], refusal: bool) -> Result<()> {
    let headers = if refusal {
        "event_search_refusal_headers"
    } else {
        "event_search_request_headers"
    };
    let params = retained
        .iter()
        .enumerate()
        .map(|(index, id)| (format!("keep_{index}"), Value::Uuid(*id)))
        .collect::<HashMap<_, _>>();
    let predicate = if retained.is_empty() {
        String::new()
    } else {
        format!(
            " WHERE request_id NOT IN ({})",
            (0..retained.len())
                .map(|index| format!("$keep_{index}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    loop {
        let rows = store.query_rows(
            &format!("SELECT context_id, request_id FROM {headers}{predicate} LIMIT 64"),
            &params,
        )?;
        if rows.rows.is_empty() {
            return Ok(());
        }
        let mut retired = BTreeMap::<crate::ContextId, Vec<uuid::Uuid>>::new();
        for row in &rows.rows {
            retired
                .entry(crate::ContextId::from(get_uuid(
                    row,
                    &rows.columns,
                    "context_id",
                )?))
                .or_default()
                .push(get_uuid(row, &rows.columns, "request_id")?);
        }
        for (context_id, requests) in retired {
            release_work_frontiers(store, context_id, &requests, refusal)?;
        }
    }
}

fn selected_frontier(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    request_id: uuid::Uuid,
    pending: &PendingStream,
) -> Result<Option<u64>> {
    let row = source.query_rows(
        "SELECT frontier FROM event_search_request_selections WHERE context_id = $context_id AND request_id = $request_id AND connector_id = $connector AND session_ref = $session AND source_instance_ref = $source",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())), ("request_id".into(), Value::Uuid(request_id)),
            ("connector".into(), Value::Text(pending.connector_id.clone())), ("session".into(), Value::Text(pending.session_ref.clone())), ("source".into(), Value::Text(pending.source_instance_ref.clone())),
        ]),
    )?;
    row.rows
        .first()
        .map(|value| get_u64(value, &row.columns, "frontier"))
        .transpose()
}

// The nil generation is captured work awaiting a target job. It stores no count/frontier;
// real generation markers are enrolled with their cursor in membership preparation.
const UNASSIGNED_SEARCH_GENERATION: uuid::Uuid = uuid::Uuid::nil();

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_stream_pending_in_tx(
    store: &Store,
    tx: TxId,
    context_id: crate::ContextId,
    generation_id: uuid::Uuid,
    connector: &str,
    source: &str,
    session: &str,
    pending: bool,
) -> Result<()> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id)),
        ("connector".into(), Value::Text(connector.into())),
        ("source".into(), Value::Text(source.into())),
        ("session".into(), Value::Text(session.into())),
        ("pending".into(), Value::Bool(pending)),
    ]);
    let existing = store.internal_db().execute_in_tx(tx,
        "SELECT pending FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session", &params).map_err(map_engine_err)?;
    let statement = if existing.rows.is_empty() {
        "INSERT INTO event_search_pending_streams (context_id, generation_id, connector_id, source_instance_ref, session_ref, pending) VALUES ($context_id, $generation_id, $connector, $source, $session, $pending)"
    } else {
        "UPDATE event_search_pending_streams SET pending = $pending WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session"
    };
    store
        .internal_db()
        .execute_in_tx(tx, statement, &params)
        .map_err(map_engine_err)?;
    Ok(())
}

pub(crate) fn mark_captured_stream_pending_in_tx(
    store: &Store,
    tx: TxId,
    event: &ConversationEvent,
) -> Result<()> {
    let params = HashMap::from([("context_id".into(), Value::Uuid(event.context_id.as_uuid()))]);
    let mut targets = BTreeSet::from([UNASSIGNED_SEARCH_GENERATION]);
    let head = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
            &params,
        )
        .map_err(map_engine_err)?;
    for row in &head.rows {
        targets.insert(get_uuid(row, &head.columns, "generation_id")?);
    }
    for state in ["capturing", "building"] {
        let mut params = params.clone();
        params.insert("state".into(), Value::Text(state.into()));
        let jobs = store.internal_db().execute_in_tx(tx,
            "SELECT id FROM event_search_generations WHERE context_id = $context_id AND state = $state", &params).map_err(map_engine_err)?;
        for row in &jobs.rows {
            targets.insert(get_uuid(row, &jobs.columns, "id")?);
        }
    }
    for target in targets {
        set_stream_pending_in_tx(
            store,
            tx,
            event.context_id,
            target,
            &event.connector_id,
            &event.source_instance_ref,
            &event.session_ref,
            true,
        )?;
        if target == UNASSIGNED_SEARCH_GENERATION {
            continue;
        }
        // The capture mark names this append, so a capture that read the stream before it commits
        // cannot clear the mark and leave these events uncaptured.
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_pending_streams SET capture_open = true, capture_pending_tx = $tx WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session",
                &HashMap::from([
                    ("tx".into(), Value::TxId(tx)),
                    ("context_id".into(), Value::Uuid(event.context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(target)),
                    ("connector".into(), Value::Text(event.connector_id.clone())),
                    ("source".into(), Value::Text(event.source_instance_ref.clone())),
                    ("session".into(), Value::Text(event.session_ref.clone())),
                ]),
            )
            .map_err(map_engine_err)?;
    }
    Ok(())
}

pub(crate) fn select_work(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    policy: EventSearchWorkPolicy,
) -> Result<EventSearchSelectedWork> {
    select_work_for_spec(source, context_id, policy, None)
}

pub(crate) fn select_work_for_spec(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    policy: EventSearchWorkPolicy,
    spec: Option<&EventSearchWorkSpec>,
) -> Result<EventSearchSelectedWork> {
    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    let head = source.query_rows(
        "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
        &params,
    )?;
    let mut target = head
        .rows
        .first()
        .map(|row| get_uuid(row, &head.columns, "generation_id"))
        .transpose()?;
    if let Some(spec) = spec {
        spec.profile.validate()?;
        let mut job_params = params.clone();
        job_params.extend([
            (
                "profile".into(),
                Value::Json(serde_json::to_value(&spec.profile).map_err(json_error)?),
            ),
            ("space".into(), Value::Text(spec.embedding_space_id.clone())),
            (
                "encoding".into(),
                Value::Text(spec.encoding_profile_id.clone()),
            ),
            ("ready".into(), Value::Text("ready".into())),
            ("failed".into(), Value::Text("failed".into())),
            ("retired".into(), Value::Text("retired".into())),
        ]);
        let job = source.query_rows(
            "SELECT id FROM event_search_generations WHERE context_id = $context_id AND renderer_profile = $profile AND embedding_space_id = $space AND encoding_profile = $encoding AND state != $ready AND state != $failed AND state != $retired ORDER BY started_at DESC, id DESC LIMIT 1", &job_params)?;
        if let Some(row) = job.rows.first() {
            target = Some(get_uuid(row, &job.columns, "id")?);
        }
    }
    if target.is_none() && spec.is_none() {
        let job = source.query_rows(
            "SELECT id FROM event_search_generations WHERE context_id = $context_id AND (state = $capturing OR state = $building) ORDER BY started_at DESC, id DESC LIMIT 1",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("capturing".into(), Value::Text("capturing".into())), ("building".into(), Value::Text("building".into()))]))?;
        if let Some(row) = job.rows.first() {
            target = Some(get_uuid(row, &job.columns, "id")?);
        }
    }
    let mut target_connectors = BTreeSet::new();
    let mut compatible = target.is_some() || spec.is_none();
    if let Some(target) = target {
        let row = source.query_rows(
            "SELECT renderer_profile, embedding_space_id, encoding_profile, interpretation_snapshot FROM event_search_generations WHERE id = $id",
            &HashMap::from([("id".into(), Value::Uuid(target))]))?;
        compatible = match row.rows.first() {
            Some(value) => {
                let saved = get_json(value, &row.columns, "interpretation_snapshot")?;
                let saved_bindings = bindings_from_snapshot(&saved)?;
                target_connectors.extend(saved_bindings.keys().cloned());
                let mut live_bindings = BTreeMap::new();
                for connector in saved_bindings.keys() {
                    live_bindings.insert(
                        connector.clone(),
                        InterpretationBinding {
                            provenance: resolve_search_binding(source, context_id, connector)?,
                        },
                    );
                }
                snapshot_json(&live_bindings) == saved
                    && spec
                        .map(|spec| -> Result<bool> {
                            Ok(get_json(value, &row.columns, "renderer_profile")?
                                == serde_json::to_value(&spec.profile).map_err(json_error)?
                                && get_text(value, &row.columns, "embedding_space_id")?
                                    == spec.embedding_space_id
                                && get_text(value, &row.columns, "encoding_profile")?
                                    == spec.encoding_profile_id)
                        })
                        .transpose()?
                        .unwrap_or(true)
            }
            None => false,
        };
    }
    // Recent scope ranks through grouped indexed top-N. All scope reads only marked streams.
    let recent = if matches!(policy.sessions, EventSearchSessionScope::Recent(_)) {
        Some(selected_captured_session_work(source, context_id, policy)?)
    } else {
        None
    };
    let mut candidates = BTreeSet::new();
    let mut complete = BTreeSet::new();
    if compatible {
        for generation_id in BTreeSet::from([
            target.unwrap_or(UNASSIGNED_SEARCH_GENERATION),
            UNASSIGNED_SEARCH_GENERATION,
        ]) {
            let mut params = params.clone();
            params.insert("generation_id".into(), Value::Uuid(generation_id));
            params.insert("pending".into(), Value::Bool(true));
            source.stream_rows(
                "SELECT connector_id, session_ref, source_instance_ref FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND pending = $pending ORDER BY connector_id ASC, session_ref ASC, source_instance_ref ASC",
                &params, &mut |columns, row| {
                    candidates.insert((get_text(row, columns, "connector_id")?, get_text(row, columns, "session_ref")?, get_text(row, columns, "source_instance_ref")?));
                    Ok(())
                })?;
        }
        // A request records every stream its policy selects for its target, complete ones
        // included: refusal acknowledgement and saved ceilings match the identities the request
        // observed. A completed stream keeps its cleared marker, so it is read from the marker
        // index; one outside the selected sessions is retained at its published frontier below.
        // Without a target spec this is the bounded pending selection: pending streams only.
        if let (Some(target), Some(_)) = (target, spec) {
            let mut params = params.clone();
            params.insert("generation_id".into(), Value::Uuid(target));
            params.insert("pending".into(), Value::Bool(false));
            source.stream_rows(
                "SELECT connector_id, session_ref, source_instance_ref FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND pending = $pending ORDER BY connector_id ASC, session_ref ASC, source_instance_ref ASC",
                &params, &mut |columns, row| {
                    let stream = (get_text(row, columns, "connector_id")?, get_text(row, columns, "session_ref")?, get_text(row, columns, "source_instance_ref")?);
                    candidates.insert(stream.clone());
                    complete.insert(stream);
                    Ok(())
                })?;
        }
    }
    if target.is_some()
        && candidates
            .iter()
            .any(|(connector, _, _)| !target_connectors.contains(connector))
    {
        compatible = false;
    }
    if !compatible || (target.is_none() && candidates.is_empty()) {
        candidates.clear();
        // A replacement target has no cursor: every captured stream is pending for that job.
        let groups = source.query_rows(
            "SELECT connector_id, source_instance_ref, COUNT(*) AS event_count FROM events WHERE context_id = $context_id GROUP BY connector_id, source_instance_ref", &params)?;
        for row in &groups.rows {
            let connector = get_text(row, &groups.columns, "connector_id")?;
            let stream = get_text(row, &groups.columns, "source_instance_ref")?;
            let newest = source.query_rows(
                "SELECT session_ref FROM events WHERE context_id = $context_id AND connector_id = $connector AND source_instance_ref = $source ORDER BY seq DESC LIMIT 1",
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("connector".into(), Value::Text(connector.clone())), ("source".into(), Value::Text(stream.clone()))]))?;
            let newest_row = newest
                .rows
                .first()
                .ok_or_else(|| CgError::Engine("captured group has no newest event".into()))?;
            candidates.insert((
                connector,
                get_text(newest_row, &newest.columns, "session_ref")?,
                stream,
            ));
        }
    }
    let mut streams = Vec::new();
    for (connector, session, stream) in candidates {
        let session_key = SessionKey {
            connector_id: connector.clone(),
            session_ref: session.clone(),
        };
        let params = HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector".into(), Value::Text(connector.clone())),
            ("session".into(), Value::Text(session.clone())),
            ("source".into(), Value::Text(stream.clone())),
        ]);
        let mut retained_frontier = None;
        if recent
            .as_ref()
            .is_some_and(|selected| !selected.rank.contains_key(&session_key))
        {
            if compatible
                && !complete.contains(&(connector.clone(), session.clone(), stream.clone()))
            {
                continue;
            }
            // Replacement rebuilds, and a request's completed streams outside its selected
            // sessions, keep previously published history at its committed frontier; newer
            // capture outside the requested scope remains deferred.
            let Some(head_row) = head.rows.first() else {
                continue;
            };
            let mut prior_params = params.clone();
            prior_params.insert(
                "generation_id".into(),
                Value::Uuid(get_uuid(head_row, &head.columns, "generation_id")?),
            );
            let progress = source.query_rows(
                "SELECT is_published, searchable_event_count FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND session_ref = $session", &prior_params)?;
            let Some(progress_row) = progress.rows.first() else {
                continue;
            };
            if !get_bool(progress_row, &progress.columns, "is_published")?
                || get_u64(progress_row, &progress.columns, "searchable_event_count")? == 0
            {
                continue;
            }
            let cursor = source.query_rows(
                "SELECT committed_sequence FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session", &prior_params)?;
            let Some(cursor_row) = cursor.rows.first() else {
                continue;
            };
            retained_frontier = Some(get_u64(cursor_row, &cursor.columns, "committed_sequence")?);
        }
        let session_row = source.query_rows(
            "SELECT latest_occurred_at FROM event_context_sessions WHERE context_id = $context_id AND connector_id = $connector AND session_ref = $session", &params)?;
        let occurred = session_row
            .rows
            .first()
            .map(|row| get_optional_timestamp(row, &session_row.columns, "latest_occurred_at"))
            .transpose()?
            .flatten();
        let newest = source.query_rows(
            "SELECT seq FROM events WHERE context_id = $context_id AND connector_id = $connector AND source_instance_ref = $source ORDER BY seq DESC LIMIT 1", &params)?;
        let row = newest
            .rows
            .first()
            .ok_or_else(|| CgError::Engine("pending stream has no newest event".into()))?;
        streams.push((
            occurred,
            EventSearchSelectedStream {
                connector_id: connector,
                session_ref: session,
                source_instance_ref: stream,
                frontier: retained_frontier.unwrap_or(get_u64(row, &newest.columns, "seq")?),
            },
        ));
    }
    streams.sort_by(|left, right| {
        let time = match (left.0, right.0) {
            (Some(left), Some(right)) => match policy.order {
                EventSearchOrder::Newest => right.cmp(&left),
                EventSearchOrder::Oldest => left.cmp(&right),
            },
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        time.then_with(|| {
            (
                &left.1.connector_id,
                &left.1.session_ref,
                &left.1.source_instance_ref,
            )
                .cmp(&(
                    &right.1.connector_id,
                    &right.1.session_ref,
                    &right.1.source_instance_ref,
                ))
        })
    });
    Ok(EventSearchSelectedWork {
        request_id: None,
        streams: streams.into_iter().map(|(_, stream)| stream).collect(),
    })
}

pub(crate) fn selected_work_complete(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    spec: &EventSearchWorkSpec,
    selected: &EventSearchSelectedWork,
) -> Result<bool> {
    spec.profile.validate()?;
    if selected.streams.is_empty() {
        return Ok(true);
    }
    let connectors = event_connectors(source, context_id)?;
    let bindings = interpretation_bindings(
        source,
        context_id,
        &connectors,
        InterpretationSelection::Ordinary,
        None,
    )?;
    let snapshot = snapshot_json(&bindings);
    let Some((generation, _)) = select_generation(
        source,
        context_id,
        &snapshot,
        &hash_json(&snapshot),
        spec,
        true,
    )?
    else {
        return Ok(selected.streams.is_empty());
    };
    if !generation_is_head(source, context_id, generation.id)? {
        return Ok(false);
    }
    for stream in &selected.streams {
        let rows = source.query_rows(
            "SELECT committed_sequence FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
                ("connector_id".into(), Value::Text(stream.connector_id.clone())),
                ("source".into(), Value::Text(stream.source_instance_ref.clone())),
                ("session".into(), Value::Text(stream.session_ref.clone())),
            ]),
        )?;
        if !rows
            .rows
            .first()
            .map(|row| {
                get_u64(row, &rows.columns, "committed_sequence")
                    .map(|committed| committed >= stream.frontier)
            })
            .transpose()?
            .unwrap_or(false)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn event_search_work(
    source: &dyn EventSearchRows,
    selector: ContextSelector,
    spec: &EventSearchWorkSpec,
    policy: EventSearchWorkPolicy,
) -> Result<Vec<EventSearchWorkAssessment>> {
    spec.profile.validate()?;
    let selected = selector.into_context_ids();
    super::event_history_status::context_names(source, &selected)?;
    let counts = captured_event_counts(source, &selected)?;
    selected
        .into_iter()
        .map(|context_id| {
            let captured_events = counts.get(&context_id).map_or(0, |count| count.captured);
            let connectors = event_connectors(source, context_id)?;
            let bindings = interpretation_bindings(
                source,
                context_id,
                &connectors,
                InterpretationSelection::Ordinary,
                None,
            )?;
            let snapshot = snapshot_json(&bindings);
            let selected_sessions = selected_captured_session_work(source, context_id, policy)?;
            let generation = select_generation(
                source,
                context_id,
                &snapshot,
                &hash_json(&snapshot),
                spec,
                true,
            )?;
            let (
                generation_id,
                covered_events,
                pending_events,
                outside_scope_events,
                outside_scope_indexed_events,
                no_searchable_text_events,
                active,
            ) = match generation {
                Some((generation, _)) => {
                    let coverage = policy_coverage(
                        source,
                        context_id,
                        generation.id,
                        &selected_sessions,
                        captured_events,
                    )?;
                    let active = generation_is_head(source, context_id, generation.id)?;
                    (
                        Some(generation.id),
                        coverage.covered_events,
                        coverage.pending_events,
                        coverage.outside_scope_events,
                        coverage.outside_scope_indexed_events,
                        coverage.no_searchable_text_events,
                        active,
                    )
                }
                None => {
                    let selected = selected_sessions.selected_events;
                    (
                        None,
                        0,
                        selected,
                        captured_events.saturating_sub(selected),
                        0,
                        0,
                        false,
                    )
                }
            };
            Ok(EventSearchWorkAssessment {
                context_id,
                captured_events,
                generation_id,
                covered_events,
                pending_events,
                outside_scope_events,
                outside_scope_indexed_events,
                no_searchable_text_events,
                work_required: captured_events > 0 && (!active || pending_events > 0),
                live_projection_lease: live_projection_lease(source, context_id, now_millis()?)?,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Coverage bookkeeping for a whole selected Context set
// ---------------------------------------------------------------------------------------------
//
// A find over 64 Contexts must cost what a find over 64 Contexts should cost: one search over the
// selected set, not sixty-four rounds of bookkeeping. Every read below names the WHOLE selected set
// in one statement and keys its answer by Context, so the statement count a find pays does not grow
// with the Context count. The per-Context entry points that follow are thin wrappers over these,
// so there is one shape of each read and the two can never answer differently.
//
// What this trades: the engine's SQL surface has no UNION ALL and no per-partition top-N, so a
// single statement cannot carry a per-Context `LIMIT`. The set-shaped session read therefore asks
// for the selected set's session rows and applies each Context's own order and session limit here.
// The ANSWER is identical; the rows the engine produces are every session of the selected
// Contexts rather than each Context's newest few. Closing that needs a per-partition limit at the
// engine, which is an upstream capability, not a consumer workaround.

/// The persisted work policy for every Context in `context_ids`, in one statement. A Context with
/// no persisted row takes the default, exactly as the single-Context read does.
pub(crate) fn persisted_work_policies(
    source: &dyn EventSearchRows,
    context_ids: &[crate::ContextId],
) -> Result<HashMap<crate::ContextId, EventSearchWorkPolicy>> {
    let mut policies = HashMap::new();
    if context_ids.is_empty() {
        return Ok(policies);
    }
    let mut params = HashMap::new();
    let filter = super::event_search_read::bind_contexts(
        &mut params,
        "policy_context",
        context_ids.iter().copied(),
    );
    let answer = source.query_rows(
        &format!(
            "SELECT context_id, search_order, search_sessions FROM event_search_work_policies \
             WHERE {filter}"
        ),
        &params,
    )?;
    for row in &answer.rows {
        let context_id = crate::ContextId::from(get_uuid(row, &answer.columns, "context_id")?);
        policies.insert(context_id, work_policy_from_row(row, &answer.columns)?);
    }
    for context_id in context_ids {
        policies
            .entry(*context_id)
            .or_insert_with(EventSearchWorkPolicy::default);
    }
    Ok(policies)
}

/// One session row as the selection reads it, before a Context's order and limit are applied.
struct CapturedSessionRow {
    key: SessionKey,
    events: u64,
    latest_occurred_at: Option<i64>,
}

/// Select policy-scoped sessions across the Context set. Recent scopes use indexed grouped
/// top-N reads; the explicit all-session scope retains the complete ordered archive.
pub(crate) fn selected_captured_session_work_for(
    source: &dyn EventSearchRows,
    context_ids: &[crate::ContextId],
    policies: &HashMap<crate::ContextId, EventSearchWorkPolicy>,
) -> Result<HashMap<crate::ContextId, SessionWorkSet>> {
    let mut work = HashMap::new();
    if context_ids.is_empty() {
        return Ok(work);
    }
    let mut recent = HashMap::new();
    let mut all_contexts = Vec::new();
    for context_id in context_ids {
        let policy = policies.get(context_id).copied().unwrap_or_default();
        match policy.sessions {
            EventSearchSessionScope::All => all_contexts.push(*context_id),
            EventSearchSessionScope::Recent(limit) => {
                recent.insert(*context_id, (policy.order, limit.get()));
            }
        }
    }
    let mut recent_rows: HashMap<crate::ContextId, Vec<CapturedSessionRow>> = HashMap::new();
    for order in [EventSearchOrder::Newest, EventSearchOrder::Oldest] {
        let limits = context_ids
            .iter()
            .filter_map(|context_id| {
                recent
                    .get(context_id)
                    .filter(|(selected_order, _)| *selected_order == order)
                    .map(|(_, limit)| (*context_id, *limit))
            })
            .collect::<Vec<_>>();
        read_recent_session_rows(source, &limits, Some(order), &mut recent_rows)?;
    }
    let remaining = context_ids
        .iter()
        .filter_map(|context_id| {
            let (_, limit) = recent.get(context_id)?;
            let retained = recent_rows.get(context_id).map_or(0, Vec::len);
            (retained < *limit).then_some((*context_id, limit.saturating_sub(retained)))
        })
        .collect::<Vec<_>>();
    read_recent_session_rows(source, &remaining, None, &mut recent_rows)?;
    for context_id in recent.keys() {
        work.insert(
            *context_id,
            session_work_set(recent_rows.remove(context_id).unwrap_or_default()),
        );
    }
    if all_contexts.is_empty() {
        return Ok(work);
    }
    let mut timestamped: HashMap<crate::ContextId, Vec<CapturedSessionRow>> = HashMap::new();
    let mut undated: HashMap<crate::ContextId, Vec<CapturedSessionRow>> = HashMap::new();
    for (timestamped_rows, predicate) in [
        (true, "latest_occurred_at IS NOT NULL"),
        (false, "latest_occurred_at IS NULL"),
    ] {
        let mut params = HashMap::new();
        let filter = super::event_search_read::bind_contexts(
            &mut params,
            "session_context",
            all_contexts.iter().copied(),
        );
        let into = if timestamped_rows {
            &mut timestamped
        } else {
            &mut undated
        };
        source.stream_rows(
            &format!(
                "SELECT context_id, connector_id, session_ref, event_count, latest_occurred_at \
                 FROM event_context_sessions WHERE {filter} AND {predicate}"
            ),
            &params,
            &mut |columns, row| {
                let context_id = crate::ContextId::from(get_uuid(row, columns, "context_id")?);
                let latest_occurred_at = match row.get(column_index(columns, "latest_occurred_at")?)
                {
                    Some(Value::Timestamp(value)) => Some(*value),
                    Some(Value::Int64(value)) => Some(*value),
                    _ => None,
                };
                into.entry(context_id)
                    .or_default()
                    .push(CapturedSessionRow {
                        key: SessionKey {
                            connector_id: get_text(row, columns, "connector_id")?,
                            session_ref: get_text(row, columns, "session_ref")?,
                        },
                        events: get_u64(row, columns, "event_count")?,
                        latest_occurred_at,
                    });
                Ok(())
            },
        )?;
    }
    for context_id in &all_contexts {
        let policy = policies
            .get(context_id)
            .copied()
            .unwrap_or_else(EventSearchWorkPolicy::default);
        let limit = match policy.sessions {
            EventSearchSessionScope::All => None,
            EventSearchSessionScope::Recent(limit) => Some(limit.get()),
        };
        let mut dated = timestamped.remove(context_id).unwrap_or_default();
        dated.sort_by(|left, right| {
            let by_time = left.latest_occurred_at.cmp(&right.latest_occurred_at);
            let by_time = match policy.order {
                EventSearchOrder::Newest => by_time.reverse(),
                EventSearchOrder::Oldest => by_time,
            };
            by_time
                .then_with(|| left.key.connector_id.cmp(&right.key.connector_id))
                .then_with(|| left.key.session_ref.cmp(&right.key.session_ref))
        });
        if let Some(limit) = limit {
            dated.truncate(limit);
        }
        let remaining = limit.map(|limit| limit.saturating_sub(dated.len()));
        let mut ordered = dated;
        if remaining != Some(0) {
            let mut undated_rows = undated.remove(context_id).unwrap_or_default();
            undated_rows.sort_by(|left, right| {
                left.key
                    .connector_id
                    .cmp(&right.key.connector_id)
                    .then_with(|| left.key.session_ref.cmp(&right.key.session_ref))
            });
            if let Some(remaining) = remaining {
                undated_rows.truncate(remaining);
            }
            ordered.extend(undated_rows);
        }
        work.insert(*context_id, session_work_set(ordered));
    }
    Ok(work)
}

/// SQL window ranking expresses a separate bounded prefix for each Context in one read.
/// Dated sessions take precedence; a second read fills only the remaining slots with undated
/// sessions. Tie order is ascending connector/session regardless of the timestamp direction.
fn read_recent_session_rows(
    source: &dyn EventSearchRows,
    limits: &[(crate::ContextId, usize)],
    order: Option<EventSearchOrder>,
    into: &mut HashMap<crate::ContextId, Vec<CapturedSessionRow>>,
) -> Result<()> {
    if limits.is_empty() {
        return Ok(());
    }
    let mut params = HashMap::new();
    let filter = super::event_search_read::bind_contexts(
        &mut params,
        "recent_context",
        limits.iter().map(|(context_id, _)| *context_id),
    );
    let mut bounds = Vec::new();
    for (index, (context_id, limit)) in limits.iter().enumerate() {
        params.insert(
            format!("rank_context_{index}"),
            Value::Uuid(context_id.as_uuid()),
        );
        params.insert(
            format!("rank_limit_{index}"),
            Value::Int64(
                i64::try_from(*limit)
                    .map_err(|_| CgError::Engine("session limit exceeds INTEGER".into()))?,
            ),
        );
        bounds.push(format!(
            "(context_id = $rank_context_{index} AND session_rank <= $rank_limit_{index})"
        ));
    }
    let (predicate, ordering) = match order {
        Some(EventSearchOrder::Newest) => (
            "latest_occurred_at IS NOT NULL",
            "latest_occurred_at DESC, connector_id ASC, session_ref ASC",
        ),
        Some(EventSearchOrder::Oldest) => (
            "latest_occurred_at IS NOT NULL",
            "latest_occurred_at ASC, connector_id ASC, session_ref ASC",
        ),
        None => (
            "latest_occurred_at IS NULL",
            "connector_id ASC, session_ref ASC",
        ),
    };
    source.stream_rows(
        &format!("WITH ranked_sessions AS (SELECT context_id, connector_id, session_ref, event_count, latest_occurred_at, ROW_NUMBER() OVER (PARTITION BY context_id ORDER BY {ordering}) AS session_rank FROM event_context_sessions WHERE {filter} AND {predicate}) SELECT context_id, connector_id, session_ref, event_count, latest_occurred_at FROM ranked_sessions WHERE {} ORDER BY context_id ASC, session_rank ASC", bounds.join(" OR ")),
        &params,
        &mut |columns, row| {
            let context_id = crate::ContextId::from(get_uuid(row, columns, "context_id")?);
            let latest_occurred_at = match row.get(column_index(columns, "latest_occurred_at")?) {
                Some(Value::Timestamp(value) | Value::Int64(value)) => Some(*value),
                _ => None,
            };
            into.entry(context_id).or_default().push(CapturedSessionRow {
                key: SessionKey { connector_id: get_text(row, columns, "connector_id")?, session_ref: get_text(row, columns, "session_ref")? },
                events: get_u64(row, columns, "event_count")?, latest_occurred_at,
            });
            Ok(())
        },
    )
}

fn session_work_set(ordered: Vec<CapturedSessionRow>) -> SessionWorkSet {
    let selected_events = ordered.iter().map(|row| row.events).sum();
    let events = ordered
        .iter()
        .map(|row| (row.key.clone(), row.events))
        .collect();
    let rank = ordered
        .into_iter()
        .enumerate()
        .map(|(rank, row)| (row.key, rank))
        .collect();
    SessionWorkSet {
        frontiers: None,
        rank,
        events,
        selected_events,
    }
}

/// Policy-scoped coverage for every ready head in `heads`, in two statements: the generations'
/// searchable counts, and the per-session progress rows behind them.
pub(crate) fn policy_coverage_for(
    source: &dyn EventSearchRows,
    heads: &[(crate::ContextId, EventSearchGenerationId)],
    sessions: &HashMap<crate::ContextId, SessionWorkSet>,
    captured_events: &HashMap<crate::ContextId, u64>,
) -> Result<HashMap<crate::ContextId, PolicyCoverage>> {
    let mut coverage = HashMap::new();
    if heads.is_empty() {
        return Ok(coverage);
    }
    let mut params = HashMap::new();
    let context_filter = super::event_search_read::bind_contexts(
        &mut params,
        "coverage_context",
        heads.iter().map(|(context_id, _)| *context_id),
    );
    let generation_filter = super::event_search_read::bind_uuid_values(
        &mut params,
        "id",
        "coverage_generation",
        heads
            .iter()
            .map(|(_, generation_id)| generation_id.as_uuid()),
    );
    let generations = source.query_rows(
        &format!(
            "SELECT context_id, searchable_event_count FROM event_search_generations \
             WHERE {context_filter} AND {generation_filter}"
        ),
        &params,
    )?;
    let mut covered = HashMap::new();
    for row in &generations.rows {
        covered.insert(
            crate::ContextId::from(get_uuid(row, &generations.columns, "context_id")?),
            get_u64(row, &generations.columns, "searchable_event_count")?,
        );
    }

    let mut progress_params = HashMap::new();
    let mut keys = Vec::new();
    for (context_id, generation_id) in heads {
        if let Some(work) = sessions.get(context_id) {
            for session in work.events.keys() {
                let index = keys.len();
                for (name, value) in [
                    ("context", Value::Uuid(context_id.as_uuid())),
                    ("generation", Value::Uuid(generation_id.as_uuid())),
                    ("connector", Value::Text(session.connector_id.clone())),
                    ("session", Value::Text(session.session_ref.clone())),
                ] {
                    progress_params.insert(format!("progress_{name}_{index}"), value);
                }
                keys.push(format!("($progress_context_{index}, $progress_generation_{index}, $progress_connector_{index}, $progress_session_{index})"));
            }
        }
    }
    let mut selected: HashMap<crate::ContextId, (u64, u64)> = HashMap::new();
    if !keys.is_empty() {
        source.stream_rows(
            &format!(
                "SELECT context_id, connector_id, session_ref, searchable_event_count, \
             no_searchable_text_event_count FROM event_search_session_progress \
             WHERE (context_id, generation_id, connector_id, session_ref) IN ({})",
                keys.join(", ")
            ),
            &progress_params,
            &mut |columns, row| {
                let context_id = crate::ContextId::from(get_uuid(row, columns, "context_id")?);
                let Some(session_work) = sessions.get(&context_id) else {
                    return Ok(());
                };
                let key = SessionKey {
                    connector_id: get_text(row, columns, "connector_id")?,
                    session_ref: get_text(row, columns, "session_ref")?,
                };
                if !session_work.events.contains_key(&key) {
                    return Ok(());
                }
                let entry = selected.entry(context_id).or_default();
                entry.0 = entry
                    .0
                    .saturating_add(get_u64(row, columns, "searchable_event_count")?);
                entry.1 = entry.1.saturating_add(get_u64(
                    row,
                    columns,
                    "no_searchable_text_event_count",
                )?);
                Ok(())
            },
        )?;
    }

    for (context_id, _) in heads {
        let covered_events = *covered.get(context_id).ok_or_else(|| {
            CgError::Engine("event-search coverage generation returned no row".into())
        })?;
        let selected_events = sessions
            .get(context_id)
            .map(SessionWorkSet::selected_events)
            .unwrap_or(0);
        let (selected_searchable, selected_no_text) =
            selected.get(context_id).copied().unwrap_or((0, 0));
        let outside_scope_events = captured_events
            .get(context_id)
            .copied()
            .unwrap_or(0)
            .saturating_sub(selected_events);
        coverage.insert(
            *context_id,
            PolicyCoverage {
                covered_events,
                pending_events: selected_events
                    .saturating_sub(selected_searchable)
                    .saturating_sub(selected_no_text),
                outside_scope_events,
                outside_scope_indexed_events: covered_events
                    .saturating_sub(selected_searchable)
                    .min(outside_scope_events),
                no_searchable_text_events: selected_no_text,
            },
        );
    }
    Ok(coverage)
}

pub(crate) fn selected_captured_session_work(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    policy: EventSearchWorkPolicy,
) -> Result<SessionWorkSet> {
    Ok(selected_captured_session_work_for(
        source,
        &[context_id],
        &HashMap::from([(context_id, policy)]),
    )?
    .remove(&context_id)
    .unwrap_or_else(|| session_work_set(Vec::new())))
}

pub(crate) struct PolicyCoverage {
    pub(crate) covered_events: u64,
    pub(crate) pending_events: u64,
    pub(crate) outside_scope_events: u64,
    pub(crate) outside_scope_indexed_events: u64,
    pub(crate) no_searchable_text_events: u64,
}

fn selected_session_work_for_run(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    policy: EventSearchWorkPolicy,
    selected_work: Option<&EventSearchSelectedWork>,
) -> Result<SessionWorkSet> {
    let mut selected = if selected_work.is_some() {
        session_work_set(Vec::new())
    } else {
        selected_captured_session_work(store, context_id, policy)?
    };
    if let Some(work) = selected_work {
        selected.rank.clear();
        selected.events.clear();
        let mut frontiers = BTreeMap::new();
        for stream in &work.streams {
            let key = SessionKey {
                connector_id: stream.connector_id.clone(),
                session_ref: stream.session_ref.clone(),
            };
            let next = selected.rank.len();
            selected.rank.entry(key.clone()).or_insert(next);
            let params = HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                ("connector".into(), Value::Text(stream.connector_id.clone())),
                ("session".into(), Value::Text(stream.session_ref.clone())),
                (
                    "source".into(),
                    Value::Text(stream.source_instance_ref.clone()),
                ),
                (
                    "frontier".into(),
                    integer(stream.frontier, "selected frontier")?,
                ),
            ]);
            if !selected.events.contains_key(&key) {
                let progress = store.query_rows(
                    "SELECT committed_event_count FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND session_ref = $session",
                    &params,
                )?;
                let row = progress
                    .rows
                    .first()
                    .ok_or_else(|| stale_claim("selected session has no captured progress"))?;
                selected.events.insert(
                    key.clone(),
                    get_u64(row, &progress.columns, "committed_event_count")?,
                );
            }
            let cursor = store.query_rows(
                "SELECT committed_sequence FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session",
                &params,
            )?;
            let cursor_row = cursor
                .rows
                .first()
                .ok_or_else(|| stale_claim("selected stream has no initialized cursor"))?;
            let committed = get_u64(cursor_row, &cursor.columns, "committed_sequence")?;
            let selected_count = selected_events_through(|sql| {
                let rows = store.query_rows(sql, &params)?;
                get_u64(&rows.rows[0], &rows.columns, "event_count")
            })?;
            let overlap = if committed == 0 {
                0
            } else {
                let mut committed_params = params.clone();
                committed_params.insert(
                    "frontier".into(),
                    integer(
                        committed.min(stream.frontier),
                        "committed selected frontier",
                    )?,
                );
                selected_events_through(|sql| {
                    let rows = store.query_rows(sql, &committed_params)?;
                    get_u64(&rows.rows[0], &rows.columns, "event_count")
                })?
            };
            let pending = selected_count.checked_sub(overlap).ok_or_else(|| {
                stale_claim("selected frontier count is smaller than its committed prefix")
            })?;
            let total = selected
                .events
                .get_mut(&key)
                .expect("selected session was initialized");
            *total = total
                .checked_add(pending)
                .ok_or_else(|| CgError::Engine("selected event count exhausted u64".into()))?;
            frontiers.insert((key, stream.source_instance_ref.clone()), stream.frontier);
        }
        selected.frontiers = Some(frontiers);
        selected.selected_events = selected.events.values().sum();
    }
    selected_session_work_from_selection(store, context_id, generation_id, selected)
}

fn selected_session_work(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    policy: EventSearchWorkPolicy,
) -> Result<SessionWorkSet> {
    let selected = selected_captured_session_work(store, context_id, policy)?;
    selected_session_work_from_selection(store, context_id, generation_id, selected)
}

fn selected_session_work_from_selection(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    selected: SessionWorkSet,
) -> Result<SessionWorkSet> {
    let mut required = selected.events.keys().cloned().collect::<BTreeSet<_>>();
    let head = store
        .internal_db()
        .execute(
            "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    if let Some(row) = head.rows.first() {
        let active = EventSearchGenerationId::from(get_uuid(row, &head.columns, "generation_id")?);
        if active != generation_id && selected.frontiers.is_none() {
            store.stream_rows(
                "SELECT connector_id, session_ref FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND is_published = $published AND searchable_event_count > $zero ORDER BY connector_id ASC, session_ref ASC",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(active.as_uuid())),
                    ("published".into(), Value::Bool(true)),
                    ("zero".into(), Value::Int64(0)),
                ]),
                &mut |columns, row| {
                    required.insert(SessionKey {
                        connector_id: get_text(row, columns, "connector_id")?,
                        session_ref: get_text(row, columns, "session_ref")?,
                    });
                    Ok(())
                },
            )?;
        }
    }
    persist_required_sessions(
        store,
        context_id,
        generation_id,
        &required,
        selected.frontiers.as_ref().map(|_| &selected.events),
    )?;

    let mut counts = BTreeMap::new();
    store.stream_rows(
        "SELECT connector_id, session_ref, event_count FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND is_required = $required ORDER BY connector_id ASC, session_ref ASC",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
            ("required".into(), Value::Bool(true)),
        ]),
        &mut |columns, row| {
            counts.insert(
                SessionKey {
                    connector_id: get_text(row, columns, "connector_id")?,
                    session_ref: get_text(row, columns, "session_ref")?,
                },
                get_u64(row, columns, "event_count")?,
            );
            Ok(())
        },
    )?;
    let mut ordered = counts.keys().cloned().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        let left_rank = selected.rank.get(left).copied();
        let right_rank = selected.rank.get(right).copied();
        match (left_rank, right_rank) {
            (Some(left), Some(right)) => left.cmp(&right),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.cmp(right),
        }
    });
    if selected.frontiers.is_some() {
        for (key, count) in &selected.events {
            counts.insert(key.clone(), *count);
        }
    }
    let selected_events = counts.values().copied().sum();
    let rank = ordered
        .into_iter()
        .enumerate()
        .map(|(rank, key)| (key, rank))
        .collect();
    Ok(SessionWorkSet {
        frontiers: selected.frontiers,
        rank,
        events: counts,
        selected_events,
    })
}

fn persist_required_sessions(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    required: &BTreeSet<SessionKey>,
    selected_counts: Option<&BTreeMap<SessionKey, u64>>,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        let generation = store.internal_db().execute_in_tx(
            tx,
            "SELECT required_event_count FROM event_search_generations WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("id".into(), Value::Uuid(generation_id.as_uuid())),
            ]),
        ).map_err(map_engine_err)?;
        let row = generation.rows.first().ok_or_else(|| {
            stale_claim("event-search generation disappeared while its required scope was recorded")
        })?;
        let mut required_events = get_u64(row, &generation.columns, "required_event_count")?;
        for key in required {
            let params = HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                ("connector_id".into(), Value::Text(key.connector_id.clone())),
                ("session".into(), Value::Text(key.session_ref.clone())),
            ]);
            let progress = store.internal_db().execute_in_tx(
                tx,
                "SELECT event_count, is_required FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND session_ref = $session",
                &params,
            ).map_err(map_engine_err)?;
            let progress_row = progress.rows.first().ok_or_else(|| {
                stale_claim("required event-search session is absent from the captured generation")
            })?;
            if !get_bool(progress_row, &progress.columns, "is_required")? {
                required_events = required_events
                    .checked_add(get_u64(progress_row, &progress.columns, "event_count")?)
                    .ok_or_else(|| CgError::Engine("required event count exhausted u64".into()))?;
                store.internal_db().execute_in_tx(
                    tx,
                    "UPDATE event_search_session_progress SET is_required = $required WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND session_ref = $session",
                    &{
                        let mut values = params;
                        values.insert("required".into(), Value::Bool(true));
                        values
                    },
                ).map_err(map_engine_err)?;
            }
        }
        if let Some(selected_counts) = selected_counts {
            required_events = 0;
            store.stream_rows(
                "SELECT connector_id, session_ref, committed_event_count FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                ]),
                &mut |columns, row| {
                    let key = SessionKey {
                        connector_id: get_text(row, columns, "connector_id")?,
                        session_ref: get_text(row, columns, "session_ref")?,
                    };
                    required_events = required_events.checked_add(get_u64(row, columns, "committed_event_count")?
                        .max(selected_counts.get(&key).copied().unwrap_or(0)))
                        .ok_or_else(|| CgError::Engine("required event count exhausted u64".into()))?;
                    Ok(())
                },
            )?;
        }
        store.internal_db().execute_in_tx(
            tx,
            "UPDATE event_search_generations SET required_event_count = $required_events WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("required_events".into(), integer(required_events, "required event count")?),
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("id".into(), Value::Uuid(generation_id.as_uuid())),
            ]),
        ).map_err(map_engine_err)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)
}

fn generation_is_head(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
) -> Result<bool> {
    let head = source.query_rows(
        "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
    )?;
    Ok(head.rows.first().is_some_and(|row| {
        get_uuid(row, &head.columns, "generation_id").ok() == Some(generation_id.as_uuid())
    }))
}

pub(crate) fn policy_coverage(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    sessions: &SessionWorkSet,
    captured_events: u64,
) -> Result<PolicyCoverage> {
    let sessions = HashMap::from([(
        context_id,
        SessionWorkSet {
            frontiers: sessions.frontiers.clone(),
            rank: sessions.rank.clone(),
            events: sessions.events.clone(),
            selected_events: sessions.selected_events,
        },
    )]);
    policy_coverage_for(
        source,
        &[(context_id, generation_id)],
        &sessions,
        &HashMap::from([(context_id, captured_events)]),
    )?
    .remove(&context_id)
    .ok_or_else(|| CgError::Engine("event-search coverage generation returned no row".into()))
}

pub(crate) fn persisted_work_policy(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
) -> Result<EventSearchWorkPolicy> {
    Ok(persisted_work_policies(source, &[context_id])?
        .remove(&context_id)
        .unwrap_or_default())
}

/// One persisted policy row, decoded. A row that names an order or a session scope this build does
/// not know is refused rather than silently read as the default.
fn work_policy_from_row(row: &[Value], columns: &[String]) -> Result<EventSearchWorkPolicy> {
    let order = match get_text(row, columns, "search_order")?.as_str() {
        "newest" => EventSearchOrder::Newest,
        "oldest" => EventSearchOrder::Oldest,
        other => {
            return Err(CgError::Engine(format!(
                "event-search policy has unknown order {other}"
            )));
        }
    };
    let sessions =
        match row.get(column_index(columns, "search_sessions")?) {
            Some(Value::Null) => EventSearchSessionScope::All,
            Some(Value::Int64(value)) if *value > 0 => EventSearchSessionScope::Recent(
                NonZeroUsize::new(usize::try_from(*value).map_err(|_| {
                    CgError::Engine("event-search session scope exceeds usize".into())
                })?)
                .expect("positive session scope"),
            ),
            other => {
                return Err(CgError::Engine(format!(
                    "event-search policy search_sessions must be positive or null, got {other:?}"
                )));
            }
        };
    Ok(EventSearchWorkPolicy { order, sessions })
}

pub(crate) fn persist_work_policy(
    store: &Store,
    context_id: crate::ContextId,
    policy: EventSearchWorkPolicy,
) -> Result<()> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        (
            "search_order".into(),
            Value::Text(
                match policy.order {
                    EventSearchOrder::Newest => "newest",
                    EventSearchOrder::Oldest => "oldest",
                }
                .into(),
            ),
        ),
        (
            "search_sessions".into(),
            match policy.sessions {
                EventSearchSessionScope::All => Value::Null,
                EventSearchSessionScope::Recent(limit) => integer(
                    u64::try_from(limit.get()).unwrap_or(u64::MAX),
                    "search session scope",
                )?,
            },
        ),
    ]);
    let tx = store.begin_tx()?;
    let write = (|| -> Result<()> {
        let existing = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT context_id FROM event_search_work_policies WHERE context_id = $context_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let sql = if existing.rows.is_empty() {
            "INSERT INTO event_search_work_policies (context_id, search_order, search_sessions) VALUES ($context_id, $search_order, $search_sessions)"
        } else {
            "UPDATE event_search_work_policies SET search_order = $search_order, search_sessions = $search_sessions WHERE context_id = $context_id"
        };
        store
            .internal_db()
            .execute_in_tx(tx, sql, &params)
            .map_err(map_engine_err)?;
        Ok(())
    })();
    if let Err(error) = write {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)
}

pub(crate) fn live_projection_lease(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    now_ms: i64,
) -> Result<bool> {
    let leases = source.query_rows(
        "SELECT expires_at FROM event_search_leases WHERE context_id = $context_id",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
    )?;
    for row in &leases.rows {
        if get_timestamp(row, &leases.columns, "expires_at")? > now_ms {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn committed_generation_coverage(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
) -> Result<u64> {
    let answer = source.query_rows(
        "SELECT COUNT(*) AS n FROM event_search_source_events INNER JOIN event_search_cursors ON event_search_source_events.context_id = event_search_cursors.context_id AND event_search_source_events.generation_id = event_search_cursors.generation_id AND event_search_source_events.connector_id = event_search_cursors.connector_id AND event_search_source_events.source_instance_ref = event_search_cursors.source_instance_ref AND event_search_source_events.session_ref = event_search_cursors.session_ref WHERE event_search_source_events.context_id = $context_id AND event_search_source_events.generation_id = $generation_id AND event_search_source_events.source_sequence <= event_search_cursors.committed_sequence",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
        ]),
    )?;
    let row = answer
        .rows
        .first()
        .ok_or_else(|| CgError::Engine("event-search coverage count returned no row".into()))?;
    get_u64(row, &answer.columns, "n")
}

#[allow(clippy::too_many_arguments)]
fn ensure_generation(
    store: &Store,
    context_id: crate::ContextId,
    embedder: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    force_rebuild: bool,
    selection: InterpretationSelection,
    attempt: &mut Option<ProjectionAttempt>,
) -> Result<Generation> {
    profile.validate()?;
    ensure_event_context_revision(store, context_id)?;
    admit_projection_embedding_space(store, embedder)?;
    let mut connectors = event_connectors(store, context_id)?;
    let explicit_connector = match selection {
        InterpretationSelection::Ordinary => None,
        InterpretationSelection::ExplicitStaged(id) => {
            let connector = explicit_staged_connector(store, context_id, id)?;
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
    )?;
    let snapshot = snapshot_json(&bindings);
    let snapshot_sha256 = hash_json(&snapshot);
    let spec = EventSearchWorkSpec {
        profile: profile.clone(),
        embedding_space_id: embedder.space.embedding_space_id.clone(),
        encoding_profile_id: embedder.embedder.encoding_profile_id(),
    };
    if !force_rebuild
        && let Some((mut existing, was_capturing)) = select_generation(
            store,
            context_id,
            &snapshot,
            &snapshot_sha256,
            &spec,
            matches!(selection, InterpretationSelection::Ordinary),
        )?
    {
        let projection_attempt = begin_generation_attempt(store, context_id, existing.id)?;
        existing.attempt_tx = Some(projection_attempt.tx);
        existing.attempt_token = Some(projection_attempt.token);
        *attempt = Some(projection_attempt);
        let (captured, certified) =
            capture_source_membership(store, context_id, &bindings, Some(existing.id), bounds)?;
        if was_capturing {
            finalize_capture_attempt(store, context_id, existing.id, &captured)?;
            existing.event_count = captured.event_count;
            existing.ordered_source_set_sha256 = captured.ordered_source_set_sha256.clone();
            existing.source_content_sha256 = captured.source_content_sha256.clone();
        }
        existing.target = captured;
        existing.certified = certified;
        return Ok(existing);
    }

    let id = EventSearchGenerationId::new_v7();
    let (started_at, projection_attempt) = insert_capture_attempt(
        store,
        context_id,
        id,
        &bindings,
        &snapshot,
        &snapshot_sha256,
        embedder,
        profile,
    )?;
    *attempt = Some(projection_attempt);
    let (captured, certified) =
        capture_source_membership(store, context_id, &bindings, Some(id), bounds)?;
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
        certified,
        started_at,
        attempt_tx: Some(projection_attempt.tx),
        attempt_token: Some(projection_attempt.token),
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
) -> Result<(TxId, ProjectionAttempt)> {
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
    let attempt_token = uuid::Uuid::now_v7();
    let tx = store.begin_tx()?;
    let placeholder = format!("capturing:{}", id.as_uuid());
    let result = store.internal_db().execute_in_tx(
        tx,
        "INSERT INTO event_search_generations (id, context_id, interpretation_generation_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, started_at, last_attempt_at, last_attempt_token, completed_at, sanitized_failure) VALUES ($id, $context_id, $interpretation_generation_id, $snapshot, $snapshot_sha256, $zero, $ordered_sha256, $source_sha256, $zero, $renderer_id, $renderer_version, $profile, $space, $encoding, $state, $zero, $zero, $zero, $zero, $zero, $zero, $started_at, $started_at, $attempt_token, $null, $null)",
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
            ("attempt_token".into(), Value::Uuid(attempt_token)),
            ("null".into(), Value::Null),
        ]),
    );
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(map_engine_err(error));
    }
    let enroll = (|| -> Result<()> {
        if guard_event_context_revision_in_tx(store, tx, context_id)?.is_none() {
            return Err(stale_claim(
                "capture changed while enrolling target pending streams",
            ));
        }
        let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
        let groups = store.internal_db().execute_in_tx(tx,
            "SELECT connector_id, source_instance_ref, COUNT(*) AS event_count FROM events WHERE context_id = $context_id GROUP BY connector_id, source_instance_ref", &params).map_err(map_engine_err)?;
        for row in &groups.rows {
            let connector = get_text(row, &groups.columns, "connector_id")?;
            let source = get_text(row, &groups.columns, "source_instance_ref")?;
            let mut latest_params = params.clone();
            latest_params.insert("connector".into(), Value::Text(connector.clone()));
            latest_params.insert("source".into(), Value::Text(source.clone()));
            let newest = store.internal_db().execute_in_tx(tx,
                "SELECT session_ref FROM events WHERE context_id = $context_id AND connector_id = $connector AND source_instance_ref = $source ORDER BY seq DESC LIMIT 1", &latest_params).map_err(map_engine_err)?;
            let latest = newest.rows.first().ok_or_else(|| {
                stale_claim("captured stream disappeared during target enrollment")
            })?;
            let session = get_text(latest, &newest.columns, "session_ref")?;
            set_stream_pending_in_tx(
                store,
                tx,
                context_id,
                id.as_uuid(),
                &connector,
                &source,
                &session,
                true,
            )?;
        }
        Ok(())
    })();
    if let Err(error) = enroll {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) = BEFORE_ATTEMPT_COMMIT.with(|slot| slot.borrow_mut().take()) {
        hook(store);
    }
    store.internal_db().commit(tx).map_err(map_engine_err)?;
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) = AFTER_ATTEMPT_COMMIT_BEFORE_READBACK.with(|slot| slot.borrow_mut().take()) {
        hook(store);
    }
    let (attempt, started_at) = read_committed_attempt(store, context_id, id, attempt_token)?;
    if started_at != attempt.tx {
        return Err(CgError::Engine(
            "new event-search generation has different committed start and attempt timestamps"
                .into(),
        ));
    }
    Ok((started_at, attempt))
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
fn current_ready_head(
    store: &dyn EventSearchRows,
    context_id: crate::ContextId,
    snapshot: &JsonValue,
    snapshot_sha256: &str,
    spec: &EventSearchWorkSpec,
) -> Result<Option<Generation>> {
    let head = store.query_rows(
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
    let result = store.query_rows(
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
            != serde_json::to_value(&spec.profile).map_err(json_error)?
        || get_text(row, &result.columns, "embedding_space_id")? != spec.embedding_space_id
        || get_text(row, &result.columns, "encoding_profile")? != spec.encoding_profile_id.clone()
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
            certified: None,
            started_at: get_txid(row, &result.columns, "started_at")?,
            attempt_tx: None,
            attempt_token: None,
            pending_dual_head_activation: false,
            mode: GenerationMode::ExtendReady,
        }
        .with_activation_state(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn incomplete_generation(
    store: &dyn EventSearchRows,
    context_id: crate::ContextId,
    snapshot: &JsonValue,
    snapshot_sha256: &str,
    spec: &EventSearchWorkSpec,
) -> Result<Option<(Generation, bool)>> {
    let result = store.query_rows(
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
                Value::Json(serde_json::to_value(&spec.profile).map_err(json_error)?),
            ),
            (
                "space".into(),
                Value::Text(spec.embedding_space_id.clone()),
            ),
            (
                "encoding".into(),
                Value::Text(spec.encoding_profile_id.clone()),
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
            certified: None,
            started_at: get_txid(row, &result.columns, "started_at")?,
            attempt_tx: None,
            attempt_token: None,
            pending_dual_head_activation: false,
            mode: GenerationMode::Building,
        }
        .with_activation_state(),
        state == "capturing",
    )))
}

fn event_connectors(
    source: &dyn EventSearchRows,
    context_id: crate::ContextId,
) -> Result<BTreeSet<String>> {
    let answer = source.query_rows(
        "SELECT connector_id, source_instance_ref, COUNT(*) AS event_count FROM events WHERE context_id = $context_id GROUP BY connector_id, source_instance_ref",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
    )?;
    answer
        .rows
        .iter()
        .map(|row| get_text(row, &answer.columns, "connector_id"))
        .collect()
}

fn interpretation_bindings(
    store: &dyn EventSearchRows,
    context_id: crate::ContextId,
    connectors: &BTreeSet<String>,
    selection: InterpretationSelection,
    explicit_connector: Option<&str>,
) -> Result<BTreeMap<String, InterpretationBinding>> {
    let mut bindings = BTreeMap::new();
    for connector in connectors {
        let provenance = match (selection, explicit_connector == Some(connector.as_str())) {
            (InterpretationSelection::ExplicitStaged(id), true) => {
                validate_ready_interpretation(store, context_id, connector, id)?;
                InterpretationProvenance::ExplicitStaged(id)
            }
            _ => resolve_search_binding(store, context_id, connector)?,
        };
        bindings.insert(connector.clone(), InterpretationBinding { provenance });
    }
    Ok(bindings)
}

fn explicit_staged_connector(
    store: &dyn EventSearchRows,
    context_id: crate::ContextId,
    id: InterpretationGenerationId,
) -> Result<String> {
    let result = store.query_rows(
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
    store: &dyn EventSearchRows,
    context_id: crate::ContextId,
    connector: &str,
) -> Result<InterpretationProvenance> {
    let head = store.query_rows(
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
    validate_ready_interpretation(store, context_id, connector, id)?;
    Ok(InterpretationProvenance::Active(id))
}

fn validate_ready_interpretation(
    store: &dyn EventSearchRows,
    context_id: crate::ContextId,
    connector: &str,
    id: InterpretationGenerationId,
) -> Result<()> {
    let result = store.query_rows(
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
) -> Result<(SourceSnapshot, Option<CaptureCertificate>)> {
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    {
        MEMBERSHIP_CAPTURES.with(|count| count.set(count.get() + 1));
        if let Some(reason) = NEXT_MEMBERSHIP_CAPTURE_FAILURE.with(|slot| slot.borrow_mut().take())
        {
            return Err(reason);
        }
    }
    let Some(generation_id) = generation_id else {
        return Ok((observe_source_membership(store, context_id, bindings, bounds)?, None));
    };
    let (snapshot, certificate) =
        match capture_incremental_membership(store, context_id, bindings, generation_id, bounds)? {
            Some(captured) => captured,
            None => capture_full_membership(store, context_id, bindings, generation_id, bounds)?,
        };
    Ok((snapshot, Some(certificate)))
}

/// Which bound interpretation versions a generation's retained members were last verified
/// against: per connector, the ready interpretation generation and its capture/build progress, or
/// null for the captured base. This is mutable verification progress, kept apart from the binding
/// identity that decides reuse and rebuild.
fn current_interpretation_stamp(
    store: &Store,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    bounds: Bounds,
) -> Result<JsonValue> {
    let mut stamp = JsonMap::new();
    for (connector, binding) in bindings {
        let value = match binding.generation_id() {
            None => JsonValue::Null,
            Some(id) => {
                let result = query_rows(
                    store,
                    bounds,
                    "SELECT capture_ordinal, build_ordinal FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
                    &HashMap::from([
                        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                        ("id".into(), Value::Uuid(id.as_uuid())),
                    ]),
                )?;
                let row = result
                    .rows
                    .first()
                    .ok_or_else(|| stale_claim("bound interpretation generation disappeared"))?;
                json!({
                    "generation_id": id.to_string(),
                    "capture_ordinal": get_u64(row, &result.columns, "capture_ordinal")?,
                    "build_ordinal": get_u64(row, &result.columns, "build_ordinal")?,
                })
            }
        };
        stamp.insert(connector.clone(), value);
    }
    Ok(canonicalize_json(&JsonValue::Object(stamp)))
}

fn read_event_context_revision(
    store: &Store,
    context_id: crate::ContextId,
    bounds: Bounds,
) -> Result<TxId> {
    let result = query_rows(
        store,
        bounds,
        "SELECT revision_tx FROM event_context_revisions WHERE context_id = $context_id",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
    )?;
    let row = result
        .rows
        .first()
        .ok_or_else(|| stale_claim("event Context revision disappeared before capture"))?;
    get_txid(row, &result.columns, "revision_tx")
}

fn count_rows(
    store: &Store,
    bounds: Bounds,
    sql: &str,
    params: &HashMap<String, Value>,
) -> Result<u64> {
    let result = query_rows(store, bounds, sql, params)?;
    let row = result
        .rows
        .first()
        .ok_or_else(|| CgError::Engine("event-search count returned no row".into()))?;
    get_u64(row, &result.columns, "n")
}

/// Members of one page of events, decoded and resolved against the bindings, with successive
/// ordinals from `first_ordinal`. Nothing is persisted here.
fn claimed_members_from_rows(
    store: &Store,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    bounds: Bounds,
    columns: &[String],
    rows: &[Vec<Value>],
    first_ordinal: u64,
) -> Result<Vec<ClaimedEvent>> {
    let session = store
        .internal_db()
        .read_session(read_limits(store, bounds))
        .map_err(map_engine_err)?;
    let mut members = Vec::with_capacity(rows.len());
    let mut ordinal = first_ordinal;
    for row in rows {
        let raw = captured_raw_from_row(&session, bounds, context_id, columns, row)?;
        let captured_sha256 = captured_source_sha256(
            &raw.event_columns,
            &raw.event_row,
            &raw.anchor_columns,
            &raw.anchor_rows,
        );
        let (event, effective_generation_id, effective_sha256) =
            effective_captured_event(&session, bindings, raw.event, &captured_sha256)?;
        members.push(ClaimedEvent {
            membership_id: uuid::Uuid::now_v7(),
            ordinal,
            event,
            effective_interpretation_generation_id: effective_generation_id,
            captured_source_sha256: captured_sha256,
            effective_source_sha256: effective_sha256,
        });
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| CgError::Engine("event-search capture ordinal exhausted u64".into()))?;
    }
    Ok(members)
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
fn after_membership_read_page() {
    AFTER_MEMBERSHIP_READ_PAGE.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(not(any(context_graph_internal_test_support, feature = "test-support")))]
fn after_membership_read_page() {}

/// Extend a generation's membership with only the events appended since its last capture.
///
/// Changed source streams come from the capture marks append leaves on
/// `event_search_pending_streams`, paged in `(connector_id, source_instance_ref)` order; within a
/// stream, new events are those past the stream's captured maximum sequence, in sequence order.
/// That is exactly the order today's full walk gives newly encountered events, so ordinals and
/// chained digests equal a full capture's from the same persisted prefix.
///
/// Returns `None` whenever the incremental path cannot certify the result — a generation still in
/// its first capture, interpretation progress not yet verified, a carried digest that disagrees with
/// the last member, or a count that does not reconcile — and the caller then runs the full verifying
/// capture. A Context revision that moved during the run is stale, exactly as for the full walk.
fn capture_incremental_membership(
    store: &Store,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    generation_id: EventSearchGenerationId,
    bounds: Bounds,
) -> Result<Option<(SourceSnapshot, CaptureCertificate)>> {
    let generation_params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
    ]);
    let generation = query_rows(
        store,
        bounds,
        "SELECT capture_ordinal, state, capture_ordered_sha256, capture_content_sha256, interpretation_verified_stamp FROM event_search_generations WHERE context_id = $context_id AND id = $generation_id",
        &generation_params,
    )?;
    let row = generation
        .rows
        .first()
        .ok_or_else(|| stale_claim("event-search capture generation disappeared"))?;
    if get_text(row, &generation.columns, "state")? == "capturing" {
        return Ok(None);
    }
    let verified = match row.get(column_index(&generation.columns, "interpretation_verified_stamp")?) {
        Some(Value::Json(value)) => canonicalize_json(value),
        _ => return Ok(None),
    };
    let stamp = current_interpretation_stamp(store, context_id, bindings, bounds)?;
    if verified != stamp {
        return Ok(None);
    }
    let revision = read_event_context_revision(store, context_id, bounds)?;
    let capture_ordinal = get_u64(row, &generation.columns, "capture_ordinal")?;
    let Some(mut chain) = MembershipChain::carried(
        get_text(row, &generation.columns, "capture_ordered_sha256")?,
        get_text(row, &generation.columns, "capture_content_sha256")?,
        capture_ordinal,
    ) else {
        return Ok(None);
    };
    if capture_ordinal > 0 {
        let last = query_rows(
            store,
            bounds,
            "SELECT ordinal, ordered_after, content_after FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal DESC LIMIT 1",
            &generation_params,
        )?;
        let Some(last) = last.rows.first().map(|row| (row, &last.columns)) else {
            return Ok(None);
        };
        if get_u64(last.0, last.1, "ordinal")?.checked_add(1) != Some(capture_ordinal)
            || get_text(last.0, last.1, "ordered_after")? != chain.ordered
            || get_text(last.0, last.1, "content_after")? != chain.content
        {
            // The carried digest disagrees with the last member: never extend it. The full
            // verifier recomputes the chain from member fields and refuses a corrupted member.
            return Ok(None);
        }
    }
    let mut next_ordinal = capture_ordinal;
    let page_limit = bounds.events.max(1);
    let mut after: Option<(String, String, String)> = None;
    let mut last_stream: Option<(String, String)> = None;
    loop {
        let mut params = generation_params.clone();
        let continuation = if let Some((connector, source, session)) = &after {
            params.insert("after_connector".into(), Value::Text(connector.clone()));
            params.insert("after_source".into(), Value::Text(source.clone()));
            params.insert("after_session".into(), Value::Text(session.clone()));
            " AND (connector_id > $after_connector OR (connector_id = $after_connector AND source_instance_ref > $after_source) OR (connector_id = $after_connector AND source_instance_ref = $after_source AND session_ref > $after_session))"
        } else {
            ""
        };
        let marks = query_rows(
            store,
            bounds,
            &format!(
                "SELECT connector_id, source_instance_ref, session_ref, capture_pending_tx FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND capture_open = true{continuation} ORDER BY connector_id ASC, source_instance_ref ASC, session_ref ASC LIMIT {page_limit}"
            ),
            &params,
        )?;
        let Some(last_mark) = marks.rows.last() else { break };
        after = Some((
            get_text(last_mark, &marks.columns, "connector_id")?,
            get_text(last_mark, &marks.columns, "source_instance_ref")?,
            get_text(last_mark, &marks.columns, "session_ref")?,
        ));
        for mark in &marks.rows {
            let connector = get_text(mark, &marks.columns, "connector_id")?;
            let source = get_text(mark, &marks.columns, "source_instance_ref")?;
            let session = get_text(mark, &marks.columns, "session_ref")?;
            let marked_at = match mark.get(column_index(&marks.columns, "capture_pending_tx")?) {
                Some(Value::TxId(tx)) => Some(*tx),
                _ => None,
            };
            // A stream's marks (one per session) are adjacent; capture the stream once.
            let stream = (connector.clone(), source.clone());
            if last_stream.as_ref() != Some(&stream) {
                let Some(captured) = capture_stream_increment(
                    store,
                    context_id,
                    bindings,
                    generation_id,
                    bounds,
                    &connector,
                    &source,
                    next_ordinal,
                    &chain,
                )?
                else {
                    return Ok(None);
                };
                (next_ordinal, chain) = captured;
                last_stream = Some(stream);
            }
            if !close_capture_mark(
                store,
                context_id,
                generation_id,
                &connector,
                &source,
                &session,
                marked_at,
            )? {
                // An append to this stream landed after its events were read. The committed
                // pages stay; the retried run resumes past them.
                return Err(stale_claim(
                    "captured events changed while event-search membership was paged",
                ));
            }
        }
        if marks.rows.len() < page_limit {
            break;
        }
    }
    if !certify_capture(store, context_id, generation_id, revision, &stamp, next_ordinal)? {
        return Ok(None);
    }
    Ok(Some((
        chain.membership_source_snapshot(next_ordinal),
        CaptureCertificate { revision, stamp },
    )))
}

/// Capture one source stream's events past its captured maximum, in sequence order, and check the
/// stream's member count against its event count. `None` means the counts do not reconcile.
#[allow(clippy::too_many_arguments)]
fn capture_stream_increment(
    store: &Store,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    generation_id: EventSearchGenerationId,
    bounds: Bounds,
    connector: &str,
    source: &str,
    mut next_ordinal: u64,
    chain: &MembershipChain,
) -> Result<Option<(u64, MembershipChain)>> {
    let stream_params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
        ("connector_id".into(), Value::Text(connector.into())),
        ("source".into(), Value::Text(source.into())),
    ]);
    let maximum = query_rows(
        store,
        bounds,
        "SELECT source_sequence FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source ORDER BY source_sequence DESC LIMIT 1",
        &stream_params,
    )?;
    let mut after_sequence = match maximum.rows.first() {
        Some(row) => get_u64(row, &maximum.columns, "source_sequence")?,
        None => 0,
    };
    let mut chain = chain.clone();
    let page_limit = bounds.events.max(1);
    loop {
        let mut params = stream_params.clone();
        params.insert("after_seq".into(), integer(after_sequence, "capture sequence")?);
        let page = query_rows(
            store,
            bounds,
            &format!(
                "SELECT {CAPTURED_EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source AND seq > $after_seq ORDER BY seq ASC LIMIT {page_limit}"
            ),
            &params,
        )?;
        after_membership_read_page();
        let Some(last) = page.rows.last() else { break };
        after_sequence = get_u64(last, &page.columns, "seq")?;
        let members = claimed_members_from_rows(
            store,
            context_id,
            bindings,
            bounds,
            &page.columns,
            &page.rows,
            next_ordinal,
        )?;
        chain = persist_search_member_page(store, context_id, generation_id, &members, &chain)?;
        next_ordinal = next_ordinal
            .checked_add(members.len() as u64)
            .ok_or_else(|| CgError::Engine("event-search capture ordinal exhausted u64".into()))?;
        if page.rows.len() < page_limit {
            break;
        }
    }
    let events = count_rows(
        store,
        bounds,
        "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source",
        &stream_params,
    )?;
    let members = count_rows(
        store,
        bounds,
        "SELECT COUNT(*) AS n FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source",
        &stream_params,
    )?;
    if events != members {
        return Ok(None);
    }
    Ok(Some((next_ordinal, chain)))
}

/// Clear one capture mark, only if no append rewrote it since it was read. Returns whether the mark
/// was cleared.
fn close_capture_mark(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    connector: &str,
    source: &str,
    session: &str,
    marked_at: Option<TxId>,
) -> Result<bool> {
    let tx = store.begin_tx()?;
    let result = (|| -> Result<bool> {
        let params = HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
            ("connector".into(), Value::Text(connector.into())),
            ("source".into(), Value::Text(source.into())),
            ("session".into(), Value::Text(session.into())),
        ]);
        let current = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT capture_open, capture_pending_tx FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session",
                &params,
            )
            .map_err(map_engine_err)?;
        let Some(row) = current.rows.first() else {
            return Ok(false);
        };
        let now = match row.get(column_index(&current.columns, "capture_pending_tx")?) {
            Some(Value::TxId(tx)) => Some(*tx),
            _ => None,
        };
        if !get_bool(row, &current.columns, "capture_open")? || now != marked_at {
            return Ok(false);
        }
        let updated = store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_pending_streams SET capture_open = false, capture_pending_tx = NULL WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(updated.rows_affected == 1)
    })();
    match result {
        Ok(true) => {
            store.internal_db().commit(tx).map_err(map_engine_err)?;
            Ok(true)
        }
        Ok(false) => {
            let _ = store.internal_db().rollback(tx);
            Ok(false)
        }
        Err(error) => {
            let _ = store.internal_db().rollback(tx);
            Err(error)
        }
    }
}

/// Certify a capture at the Context revision it was discovered at, and only then publish the
/// interpretation stamp its retained members were verified against. The transaction guards that
/// exact revision, reconciles the generation's member count with the Context's event count, and
/// guards every bound interpretation generation at the stamped progress. `false` means the counts
/// do not reconcile; a moved revision or interpretation is stale.
fn certify_capture(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    revision: TxId,
    stamp: &JsonValue,
    capture_ordinal: u64,
) -> Result<bool> {
    let tx = store.begin_tx()?;
    let result = (|| -> Result<bool> {
        let context = Value::Uuid(context_id.as_uuid());
        if !store
            .internal_db()
            .guard_row_conditions_in_tx(
                tx,
                "event_context_revisions",
                "context_id",
                &context,
                &[("revision_tx".into(), Value::TxId(revision))],
            )
            .map_err(map_engine_err)?
        {
            return Err(stale_claim(
                "captured events changed while event-search membership was paged",
            ));
        }
        let params = HashMap::from([
            ("context_id".into(), context.clone()),
            ("id".into(), Value::Uuid(generation_id.as_uuid())),
        ]);
        let generation = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT capture_ordinal FROM event_search_generations WHERE context_id = $context_id AND id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        let row = generation
            .rows
            .first()
            .ok_or_else(|| stale_claim("event-search capture generation disappeared"))?;
        if get_u64(row, &generation.columns, "capture_ordinal")? != capture_ordinal
            || !store
                .internal_db()
                .guard_row_conditions_in_tx(
                    tx,
                    "event_search_generations",
                    "id",
                    &Value::Uuid(generation_id.as_uuid()),
                    &[(
                        "capture_ordinal".into(),
                        integer(capture_ordinal, "capture ordinal")?,
                    )],
                )
                .map_err(map_engine_err)?
        {
            return Err(stale_claim(
                "event-search generation changed before capture certification",
            ));
        }
        let events = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let events = events
            .rows
            .first()
            .map(|row| get_u64(row, &events.columns, "n"))
            .transpose()?
            .unwrap_or(0);
        if events != capture_ordinal {
            return Ok(false);
        }
        if let JsonValue::Object(bound) = stamp {
            for value in bound.values() {
                let JsonValue::Object(entry) = value else { continue };
                let id = entry
                    .get("generation_id")
                    .and_then(JsonValue::as_str)
                    .and_then(|id| uuid::Uuid::parse_str(id).ok())
                    .ok_or_else(|| CgError::Engine("interpretation stamp lacks its generation".into()))?;
                let ordinal = |name: &str| -> Result<Value> {
                    entry
                        .get(name)
                        .and_then(JsonValue::as_u64)
                        .ok_or_else(|| CgError::Engine(format!("interpretation stamp lacks {name}")))
                        .and_then(|value| integer(value, name))
                };
                if !store
                    .internal_db()
                    .guard_row_conditions_in_tx(
                        tx,
                        "event_interpretation_generations",
                        "id",
                        &Value::Uuid(id),
                        &[
                            ("capture_ordinal".into(), ordinal("capture_ordinal")?),
                            ("build_ordinal".into(), ordinal("build_ordinal")?),
                        ],
                    )
                    .map_err(map_engine_err)?
                {
                    return Err(stale_claim(
                        "bound interpretation advanced before capture certification",
                    ));
                }
            }
        }
        let mut update = params;
        update.insert("stamp".into(), Value::Json(stamp.clone()));
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_generations SET interpretation_verified_stamp = $stamp WHERE context_id = $context_id AND id = $id",
                &update,
            )
            .map_err(map_engine_err)?;
        Ok(true)
    })();
    match result {
        Ok(true) => {
            store.internal_db().commit(tx).map_err(map_engine_err)?;
            Ok(true)
        }
        Ok(false) => {
            let _ = store.internal_db().rollback(tx);
            Ok(false)
        }
        Err(error) => {
            let _ = store.internal_db().rollback(tx);
            Err(error)
        }
    }
}

/// The full verifying capture: fold and verify every persisted member's chain, re-read every event
/// of the Context in `(connector_id, source_instance_ref, seq)` order, re-verify each retained
/// member against it, append the new ones in that order, then certify and publish the stamp. This is
/// the path for a generation's first capture, interpretation progress not yet verified, a carried
/// digest that disagrees with its members, and a count that did not reconcile.
fn capture_full_membership(
    store: &Store,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    generation_id: EventSearchGenerationId,
    bounds: Bounds,
) -> Result<(SourceSnapshot, CaptureCertificate)> {
    let revision = read_event_context_revision(store, context_id, bounds)?;
    let stamp = current_interpretation_stamp(store, context_id, bindings, bounds)?;
    let persisted = fold_source_membership(store, context_id, generation_id, bounds)?;
    let mut next_ordinal = generation_capture_high_water(store, context_id, generation_id, bounds)?;
    if persisted.event_count != next_ordinal {
        return Err(stale_claim(
            "event-search persisted membership differs from its capture ordinal",
        ));
    }
    let mut chain = MembershipChain {
        ordered: persisted.ordered_source_set_sha256,
        content: persisted.source_content_sha256,
    };
    reseal_generation_chain(store, context_id, generation_id, next_ordinal, &chain)?;
    let starting_events = captured_event_counts(store, &[context_id])?
        .get(&context_id)
        .map_or(0, |counts| counts.captured);
    let mut count = 0_u64;
    let mut after: Option<(String, String, u64)> = None;
    loop {
        // Materialize one configured event batch before decoding or persisting any member.
        // Membership work can take much longer than the engine's read-statement lifetime.
        let mut params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
        let continuation = if let Some((connector, source, seq)) = &after {
            params.insert("after_connector".into(), Value::Text(connector.clone()));
            params.insert("after_source".into(), Value::Text(source.clone()));
            params.insert("after_seq".into(), integer(*seq, "capture sequence")?);
            " AND (connector_id > $after_connector OR (connector_id = $after_connector AND source_instance_ref > $after_source) OR (connector_id = $after_connector AND source_instance_ref = $after_source AND seq > $after_seq))"
        } else {
            ""
        };
        let page = query_rows(
            store,
            bounds,
            &format!(
                "SELECT {CAPTURED_EVENT_COLUMNS} FROM events WHERE context_id = $context_id{continuation} ORDER BY connector_id ASC, source_instance_ref ASC, seq ASC LIMIT {}",
                bounds.events.max(1)
            ),
            &params,
        )?;
        after_membership_read_page();
        let Some(last) = page.rows.last() else { break };
        after = Some((
            get_text(last, &page.columns, "connector_id")?,
            get_text(last, &page.columns, "source_instance_ref")?,
            get_u64(last, &page.columns, "seq")?,
        ));
        let candidates = claimed_members_from_rows(
            store,
            context_id,
            bindings,
            bounds,
            &page.columns,
            &page.rows,
            0,
        )?;
        let mut appended = Vec::new();
        for mut candidate in candidates {
            if retained_search_membership(store, context_id, generation_id, &candidate, bounds)?
                .is_none()
            {
                candidate.ordinal = next_ordinal
                    .checked_add(appended.len() as u64)
                    .ok_or_else(|| {
                        CgError::Engine("event-search capture ordinal exhausted u64".into())
                    })?;
                appended.push(candidate);
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| CgError::Engine("event-search source count exhausted u64".into()))?;
        }
        chain = persist_search_member_page(store, context_id, generation_id, &appended, &chain)?;
        next_ordinal += appended.len() as u64;
        if page.rows.len() < bounds.events.max(1) {
            break;
        }
    }
    let current_events = captured_event_counts(store, &[context_id])?
        .get(&context_id)
        .map_or(0, |counts| counts.captured);
    if current_events != starting_events || count != next_ordinal {
        return Err(stale_claim(
            "captured events changed while event-search membership was paged",
        ));
    }
    if !certify_capture(store, context_id, generation_id, revision, &stamp, next_ordinal)? {
        return Err(stale_claim(
            "event-search persisted membership differs from the fixed source snapshot",
        ));
    }
    Ok((
        chain.membership_source_snapshot(next_ordinal),
        CaptureCertificate { revision, stamp },
    ))
}

/// Make the generation's carried digests equal the chain just verified from its members. Guarded on
/// the capture ordinal the fold saw; a no-op when they already agree.
fn reseal_generation_chain(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    capture_ordinal: u64,
    chain: &MembershipChain,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let result = (|| -> Result<()> {
        if !store
            .internal_db()
            .guard_row_conditions_in_tx(
                tx,
                "event_search_generations",
                "id",
                &Value::Uuid(generation_id.as_uuid()),
                &[(
                    "capture_ordinal".into(),
                    integer(capture_ordinal, "capture ordinal")?,
                )],
            )
            .map_err(map_engine_err)?
        {
            return Err(stale_claim(
                "event-search generation changed before its chain was resealed",
            ));
        }
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_generations SET capture_ordered_sha256 = $ordered, capture_content_sha256 = $content WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("ordered".into(), Value::Text(chain.ordered.clone())),
                    ("content".into(), Value::Text(chain.content.clone())),
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(generation_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    })();
    match result {
        Ok(()) => store.internal_db().commit(tx).map_err(map_engine_err),
        Err(error) => {
            let _ = store.internal_db().rollback(tx);
            Err(error)
        }
    }
}

/// The membership a capture would assign with no generation to persist into: every event in walk
/// order, ordinals from zero.
fn observe_source_membership(
    store: &Store,
    context_id: crate::ContextId,
    bindings: &BTreeMap<String, InterpretationBinding>,
    bounds: Bounds,
) -> Result<SourceSnapshot> {
    let mut count = 0_u64;
    let mut chain = MembershipChain::start();
    let mut after: Option<(String, String, u64)> = None;
    loop {
        let mut params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
        let continuation = if let Some((connector, source, seq)) = &after {
            params.insert("after_connector".into(), Value::Text(connector.clone()));
            params.insert("after_source".into(), Value::Text(source.clone()));
            params.insert("after_seq".into(), integer(*seq, "capture sequence")?);
            " AND (connector_id > $after_connector OR (connector_id = $after_connector AND source_instance_ref > $after_source) OR (connector_id = $after_connector AND source_instance_ref = $after_source AND seq > $after_seq))"
        } else {
            ""
        };
        let page = query_rows(
            store,
            bounds,
            &format!(
                "SELECT {CAPTURED_EVENT_COLUMNS} FROM events WHERE context_id = $context_id{continuation} ORDER BY connector_id ASC, source_instance_ref ASC, seq ASC LIMIT {}",
                bounds.events.max(1)
            ),
            &params,
        )?;
        after_membership_read_page();
        let Some(last) = page.rows.last() else { break };
        after = Some((
            get_text(last, &page.columns, "connector_id")?,
            get_text(last, &page.columns, "source_instance_ref")?,
            get_u64(last, &page.columns, "seq")?,
        ));
        for member in claimed_members_from_rows(
            store,
            context_id,
            bindings,
            bounds,
            &page.columns,
            &page.rows,
            count,
        )? {
            chain = chain.extend(&member);
            count += 1;
        }
        if page.rows.len() < bounds.events.max(1) {
            break;
        }
    }
    Ok(chain.membership_source_snapshot(count))
}

/// The member already persisted for this event in the generation, verified field by field against
/// the event just re-read, or `None` when the event is not yet a member.
fn retained_search_membership(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    member: &ClaimedEvent,
    bounds: Bounds,
) -> Result<Option<ClaimedEvent>> {
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
        return Ok(Some(retained));
    }
    Ok(None)
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

const ORDERED_SOURCE_SET_DOMAIN: &[u8] = b"context-graph/event-search-ordered-source-set/v1";
const SOURCE_CONTENT_DOMAIN: &[u8] = b"context-graph/event-search-source-content/v1";
/// Prefix of every refusal for persisted membership whose stored chain does not match its fields.
const MEMBERSHIP_CHAIN_CORRUPT: &str = "event-search membership chain is corrupted: ";

/// The generation's two membership digests as per-member chains. Each member's value commits to
/// its predecessor's, so a capture extends the stored value with only the members it appends, and a
/// fold over the persisted members in ordinal order reproduces it exactly. The value depends only on
/// the ordered member list, never on how capture runs were split. An empty membership hashes the
/// bare domain, as before.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MembershipChain {
    ordered: String,
    content: String,
}

impl MembershipChain {
    fn start() -> Self {
        let mut ordered = Sha256::new();
        ordered.update(ORDERED_SOURCE_SET_DOMAIN);
        let mut content = Sha256::new();
        content.update(SOURCE_CONTENT_DOMAIN);
        Self {
            ordered: finish_hash(ordered),
            content: finish_hash(content),
        }
    }

    /// The carried value of a generation row; an untouched generation carries empty text.
    fn carried(ordered: String, content: String, capture_ordinal: u64) -> Option<Self> {
        if ordered.is_empty() && content.is_empty() {
            return (capture_ordinal == 0).then(Self::start);
        }
        Some(Self { ordered, content })
    }

    #[allow(clippy::too_many_arguments)]
    fn extend_parts(
        &self,
        ordinal: u64,
        event_id: uuid::Uuid,
        connector_id: &str,
        source_instance_ref: &str,
        session_ref: &str,
        source_sequence: u64,
        captured_source_sha256: &str,
        effective_source_sha256: &str,
    ) -> Self {
        let mut ordered = Sha256::new();
        ordered.update(ORDERED_SOURCE_SET_DOMAIN);
        hash_part(&mut ordered, self.ordered.as_bytes());
        hash_part(&mut ordered, &ordinal.to_be_bytes());
        hash_part(&mut ordered, event_id.as_bytes());
        hash_part(&mut ordered, connector_id.as_bytes());
        hash_part(&mut ordered, source_instance_ref.as_bytes());
        hash_part(&mut ordered, session_ref.as_bytes());
        hash_part(&mut ordered, &source_sequence.to_be_bytes());
        hash_part(&mut ordered, captured_source_sha256.as_bytes());
        let mut content = Sha256::new();
        content.update(SOURCE_CONTENT_DOMAIN);
        hash_part(&mut content, self.content.as_bytes());
        hash_part(&mut content, effective_source_sha256.as_bytes());
        Self {
            ordered: finish_hash(ordered),
            content: finish_hash(content),
        }
    }

    fn extend(&self, member: &ClaimedEvent) -> Self {
        self.extend_parts(
            member.ordinal,
            member.event.id.as_uuid(),
            &member.event.connector_id,
            &member.event.source_instance_ref,
            &member.event.session_ref,
            member.event.seq,
            &member.captured_source_sha256,
            &member.effective_source_sha256,
        )
    }

    fn membership_source_snapshot(&self, event_count: u64) -> SourceSnapshot {
        SourceSnapshot {
            event_count,
            ordered_source_set_sha256: self.ordered.clone(),
            source_content_sha256: self.content.clone(),
        }
    }
}

/// Recompute the membership chain from every persisted member's fields in ordinal order and refuse
/// any member whose stored chain value differs. This is the full verifier: it runs on every full
/// capture and on explicit verification, never on the ordinary incremental path.
fn fold_source_membership(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    bounds: Bounds,
) -> Result<SourceSnapshot> {
    let mut count = 0_u64;
    let mut chain = MembershipChain::start();
    stream_rows(
        store,
        bounds,
        &format!(
            "SELECT {SOURCE_MEMBER_COLUMNS}, ordered_after, content_after FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal ASC"
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
            chain = chain.extend_parts(
                ordinal,
                get_uuid(row, columns, "event_id")?,
                &get_text(row, columns, "connector_id")?,
                &get_text(row, columns, "source_instance_ref")?,
                &get_text(row, columns, "session_ref")?,
                get_u64(row, columns, "source_sequence")?,
                &get_text(row, columns, "captured_source_sha256")?,
                &get_text(row, columns, "effective_source_sha256")?,
            );
            if get_text(row, columns, "ordered_after")? != chain.ordered
                || get_text(row, columns, "content_after")? != chain.content
            {
                return Err(CgError::Engine(format!(
                    "{MEMBERSHIP_CHAIN_CORRUPT}member {ordinal} of generation {} does not match its stored chain value",
                    generation_id.as_uuid()
                )));
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| CgError::Engine("event-search source count exhausted u64".into()))?;
            Ok(())
        },
    )?;
    Ok(chain.membership_source_snapshot(count))
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

/// Append one page of new members, their chain values, the generation's capture ordinal and its
/// carried digests in one transaction. The generation is guarded on its ordinal, state and carried
/// digests, so a page applies only on top of exactly the chain it extends.
fn persist_search_member_page(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    members: &[ClaimedEvent],
    before: &MembershipChain,
) -> Result<MembershipChain> {
    let Some(first) = members.first() else {
        return Ok(before.clone());
    };
    let tx = store.begin_tx()?;
    let result = (|| -> Result<MembershipChain> {
        let generation = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT context_id, capture_ordinal, state, capture_ordered_sha256, capture_content_sha256 FROM event_search_generations WHERE context_id = $context_id AND id = $id",
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
        let capture_ordinal = get_u64(row, &generation.columns, "capture_ordinal")?;
        let carried = MembershipChain::carried(
            get_text(row, &generation.columns, "capture_ordered_sha256")?,
            get_text(row, &generation.columns, "capture_content_sha256")?,
            capture_ordinal,
        );
        if generation.rows.len() != 1
            || capture_ordinal != first.ordinal
            || carried.as_ref() != Some(before)
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
            &[
                "context_id",
                "capture_ordinal",
                "state",
                "capture_ordered_sha256",
                "capture_content_sha256",
            ],
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
        let mut chain = before.clone();
        let mut expected = first.ordinal;
        for member in members {
            if member.ordinal != expected {
                return Err(CgError::Engine(
                    "event-search membership page is not contiguous".into(),
                ));
            }
            chain = chain.extend(member);
            store
                .internal_db()
                .execute_in_tx(
                    tx,
                    "INSERT INTO event_search_source_events (membership_id, context_id, generation_id, ordinal, event_id, connector_id, source_instance_ref, session_ref, source_sequence, effective_interpretation_generation_id, captured_source_sha256, effective_source_sha256, ordered_after, content_after) VALUES ($membership_id, $context_id, $generation_id, $ordinal, $event_id, $connector_id, $source, $session, $sequence, $effective_generation_id, $captured_sha256, $effective_sha256, $ordered_after, $content_after)",
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
                        ("ordered_after".into(), Value::Text(chain.ordered.clone())),
                        ("content_after".into(), Value::Text(chain.content.clone())),
                    ]),
                )
                .map_err(map_engine_err)?;
            update_generation_session_capture_in_tx(store, tx, context_id, generation_id, member)?;
            ensure_generation_cursor_in_tx(
                store,
                tx,
                context_id,
                generation_id,
                &member.event.connector_id,
                &member.event.source_instance_ref,
                &member.event.session_ref,
            )?;
            expected = expected.checked_add(1).ok_or_else(|| {
                CgError::Engine("event-search capture ordinal exhausted u64".into())
            })?;
        }
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_generations SET capture_ordinal = $next_ordinal, capture_ordered_sha256 = $ordered, capture_content_sha256 = $content WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("next_ordinal".into(), integer(expected, "capture ordinal")?),
                    ("ordered".into(), Value::Text(chain.ordered.clone())),
                    ("content".into(), Value::Text(chain.content.clone())),
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(generation_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(chain)
    })();
    match result {
        Ok(chain) => {
            store.internal_db().commit(tx).map_err(map_engine_err)?;
            Ok(chain)
        }
        Err(error) => {
            let _ = store.internal_db().rollback(tx);
            Err(error)
        }
    }
}

fn update_generation_session_capture_in_tx(
    store: &Store,
    tx: TxId,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    member: &ClaimedEvent,
) -> Result<()> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
        (
            "connector_id".into(),
            Value::Text(member.event.connector_id.clone()),
        ),
        (
            "session".into(),
            Value::Text(member.event.session_ref.clone()),
        ),
    ]);
    let existing = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT event_count, is_required FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND session_ref = $session",
            &params,
        )
        .map_err(map_engine_err)?;
    if let Some(row) = existing.rows.first() {
        if existing.rows.len() != 1 {
            return Err(stale_claim(
                "event-search session progress primary key returned more than one row",
            ));
        }
        let next = get_u64(row, &existing.columns, "event_count")?
            .checked_add(1)
            .ok_or_else(|| CgError::Engine("event-search session count exhausted u64".into()))?;
        let mut values = params.clone();
        values.insert("event_count".into(), integer(next, "session event count")?);
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_session_progress SET event_count = $event_count WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND session_ref = $session",
                &values,
            )
            .map_err(map_engine_err)?;
        if get_bool(row, &existing.columns, "is_required")? {
            let generation = store.internal_db().execute_in_tx(
                tx,
                "SELECT required_event_count FROM event_search_generations WHERE context_id = $context_id AND id = $generation_id",
                &params,
            ).map_err(map_engine_err)?;
            let generation_row = generation.rows.first().ok_or_else(|| {
                stale_claim("event-search generation disappeared while required capture grew")
            })?;
            let required = get_u64(generation_row, &generation.columns, "required_event_count")?
                .checked_add(1)
                .ok_or_else(|| CgError::Engine("required event count exhausted u64".into()))?;
            let mut values = params.clone();
            values.insert(
                "required_events".into(),
                integer(required, "required event count")?,
            );
            store.internal_db().execute_in_tx(
                tx,
                "UPDATE event_search_generations SET required_event_count = $required_events WHERE context_id = $context_id AND id = $generation_id",
                &values,
            ).map_err(map_engine_err)?;
        }
    } else {
        let mut values = params;
        values.extend([
            ("one".into(), Value::Int64(1)),
            ("zero".into(), Value::Int64(0)),
            ("false".into(), Value::Bool(false)),
        ]);
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "INSERT INTO event_search_session_progress (context_id, generation_id, connector_id, session_ref, event_count, committed_event_count, eligible_event_count, excluded_event_count, searchable_event_count, no_searchable_text_event_count, is_required, is_published) VALUES ($context_id, $generation_id, $connector_id, $session, $one, $zero, $zero, $zero, $zero, $zero, $false, $false)",
                &values,
            )
            .map_err(map_engine_err)?;
    }
    Ok(())
}

fn ensure_generation_cursor_in_tx(
    store: &Store,
    tx: TxId,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
    connector: &str,
    source: &str,
    session: &str,
) -> Result<()> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
        ("connector_id".into(), Value::Text(connector.into())),
        ("source".into(), Value::Text(source.into())),
        ("session".into(), Value::Text(session.into())),
    ]);
    let existing = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT committed_sequence FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session",
            &params,
        )
        .map_err(map_engine_err)?;
    if !existing.rows.is_empty() {
        return Ok(());
    }
    let mut values = params;
    values.extend([
        ("empty_tail".into(), Value::Json(json!([]))),
        ("committed".into(), Value::Int64(0)),
        ("watermark".into(), Value::Text(String::new())),
        ("updated_at".into(), Value::TxId(tx)),
    ]);
    store
        .internal_db()
        .execute_in_tx(
            tx,
            "INSERT INTO event_search_cursors (context_id, generation_id, connector_id, source_instance_ref, session_ref, committed_sequence, open_tail_start_sequence, open_tail_members, open_tail_passage_id, watermark_sha256, updated_at) VALUES ($context_id, $generation_id, $connector_id, $source, $session, $committed, NULL, $empty_tail, NULL, $watermark, $updated_at)",
            &values,
        )
        .map_err(map_engine_err)?;
    set_stream_pending_in_tx(
        store,
        tx,
        context_id,
        generation_id.as_uuid(),
        connector,
        source,
        session,
        true,
    )?;
    Ok(())
}

/// Claim up to `event_limit` events of the first pending stream. The limit is the request's event
/// bound, or fewer after a claim of that many events rendered more passages than the passage bound.
fn claim_batch(
    store: &Store,
    generation: &Generation,
    sessions: &SessionWorkSet,
    bounds: Bounds,
    event_limit: usize,
) -> Result<ClaimResult> {
    let pending = first_pending_stream(store, generation, sessions, bounds)?;
    claim_pending_batch(store, generation, bounds, event_limit, pending, None)
}

fn claim_source_statement(event_limit: usize) -> String {
    format!(
        "SELECT {SOURCE_MEMBER_COLUMNS} FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session AND ordinal >= $start_ordinal ORDER BY ordinal ASC LIMIT {event_limit}"
    )
}

fn claim_pending_batch(
    store: &Store,
    generation: &Generation,
    bounds: Bounds,
    event_limit: usize,
    pending: Option<PendingStream>,
    selected_request_id: Option<uuid::Uuid>,
) -> Result<ClaimResult> {
    #[cfg(feature = "test-support")]
    LOADED_EVENT_ROWS.with(|rows| rows.set(0));
    let Some(pending) = pending else {
        return if committed_generation_coverage(store, generation.context_id, generation.id)?
            == generation.target.event_count
        {
            Ok(ClaimResult::Complete)
        } else {
            Ok(ClaimResult::ScopeComplete)
        };
    };
    let PendingStream {
        publication_event_count,
        through_sequence,
        start_ordinal,
        connector_id: connector,
        source_instance_ref: source,
        session_ref: session,
        committed_sequence: committed,
        open_tail_start_sequence: open_tail_start,
        open_tail_members,
        open_tail_passage_id,
    } = pending;
    let result = query_rows(
        store,
        bounds,
        &claim_source_statement(event_limit.clamp(1, bounds.events.max(1))),
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
            ("session".into(), Value::Text(session.clone())),
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
        if member_session != session
            || member_sequence <= committed
            || through_sequence.is_some_and(|frontier| member_sequence > frontier)
        {
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
        &open_tail_members,
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
    // A selected capture frontier is a publication boundary even if a deferred append
    // already exists. Retain the emitted tail in the cursor so widening can replace it.
    let session_ended = through_sequence.is_some_and(|frontier| end_sequence >= frontier)
        || !has_later_session_event(
            store,
            generation,
            &connector,
            &source,
            &session,
            end_sequence,
            bounds,
        )?;
    #[cfg(feature = "test-support")]
    CLAIM_LOADS.with(|loads| {
        loads.borrow_mut().push((
            events.len(),
            open_tail_events.len(),
            LOADED_EVENT_ROWS.with(std::cell::Cell::get),
        ))
    });
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
    if let Some(request_id) = selected_request_id {
        let associated = store.internal_db().execute_in_tx(
            tx,
            "UPDATE event_search_request_headers SET running_generation_id = $generation_id, running_owner_token = $owner_token WHERE context_id = $context_id AND request_id = $request_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(generation.context_id.as_uuid())),
                ("request_id".into(), Value::Uuid(request_id)),
                ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
                ("owner_token".into(), Value::Text(owner_token.clone())),
            ]),
        ).map_err(map_engine_err);
        match associated {
            Ok(updated) if updated.rows_affected == 1 => {}
            Ok(_) => {
                let _ = store.internal_db().rollback(tx);
                return Err(stale_claim(
                    "selected request disappeared before its claim was bound",
                ));
            }
            Err(error) => {
                let _ = store.internal_db().rollback(tx);
                return Err(error);
            }
        }
    }
    store.internal_db().commit(tx).map_err(map_engine_err)?;
    Ok(ClaimResult::Work(Box::new(Claim {
        publication_event_count,
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
        prior_open_tail_members: open_tail_members,
        prior_open_tail_passage_id: open_tail_passage_id,
        reopened_tail_passage_id,
        open_tail_events,
        superseded_tail,
        session_ended,
        events,
    })))
}

fn guard_attempt_in_tx(store: &Store, tx: TxId, generation: &Generation) -> Result<()> {
    let Some(attempt_tx) = generation.attempt_tx else {
        return Err(stale_claim("event-search generation has no owned attempt"));
    };
    let Some(attempt_token) = generation.attempt_token else {
        return Err(stale_claim(
            "event-search generation has no attempt ownership token",
        ));
    };
    if !store
        .internal_db()
        .guard_row_conditions_in_tx(
            tx,
            "event_search_generations",
            "id",
            &Value::Uuid(generation.id.as_uuid()),
            &[
                ("last_attempt_at".into(), Value::TxId(attempt_tx)),
                ("last_attempt_token".into(), Value::Uuid(attempt_token)),
            ],
        )
        .map_err(map_engine_err)?
    {
        return Err(stale_claim("event-search attempt was superseded"));
    }
    Ok(())
}

fn guard_generation_for_claim_in_tx(
    store: &Store,
    tx: TxId,
    generation: &Generation,
) -> Result<()> {
    guard_attempt_in_tx(store, tx, generation)?;
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
        .read_session(read_limits(store, bounds))
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
    #[cfg(feature = "test-support")]
    LOADED_EVENT_ROWS.with(|rows| rows.set(rows.get() + result.rows.len()));
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
    sessions: &SessionWorkSet,
    bounds: Bounds,
) -> Result<Option<PendingStream>> {
    let mut pending = None;
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    let mut stream_rows_examined = 0;
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    let mut cursor_lookups = 0;
    stream_rows(
        store,
        bounds,
        "SELECT connector_id, source_instance_ref, session_ref FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND pending = $pending ORDER BY connector_id ASC, session_ref ASC, source_instance_ref ASC",
        &HashMap::from([
            ("pending".into(), Value::Bool(true)),
            (
                "context_id".into(),
                Value::Uuid(generation.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
        ]),
        |columns, row| {
            #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
            {
                stream_rows_examined += 1;
            }
            let connector = get_text(row, columns, "connector_id")?;
            let source = get_text(row, columns, "source_instance_ref")?;
            let session = get_text(row, columns, "session_ref")?;
            let Some(candidate_rank) = sessions.rank(&connector, &session) else {
                return Ok(());
            };
            let cursor = query_rows(
                store,
                bounds,
                "SELECT committed_sequence, open_tail_start_sequence, open_tail_members, open_tail_passage_id FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session",
                &HashMap::from([
                    (
                        "context_id".into(),
                        Value::Uuid(generation.context_id.as_uuid()),
                    ),
                    ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
                    ("connector".into(), Value::Text(connector.clone())),
                    ("source".into(), Value::Text(source.clone())),
                    ("session".into(), Value::Text(session.clone())),
                ]),
            )?;
            // A stream captured after membership preparation is marked pending before its
            // membership and cursor exist. It is pending work, not a stale claim: the no-claim
            // path prepares its membership and cursor, then discovery finds it here.
            let Some(row) = cursor.rows.first() else {
                return Ok(());
            };
            let columns = &cursor.columns;
            let committed = get_u64(row, columns, "committed_sequence")?;
            let key = SessionKey {
                connector_id: connector.clone(),
                session_ref: session.clone(),
            };
            let through_sequence = match &sessions.frontiers {
                Some(frontiers) => match frontiers.get(&(key.clone(), source.clone())) {
                    Some(frontier) if *frontier > committed => Some(*frontier),
                    _ => return Ok(()),
                },
                None => None,
            };
            #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
            {
                cursor_lookups += 1;
            }
            let first = query_rows(
                store,
                bounds,
                FIRST_PENDING_SOURCE_STATEMENT,
                &HashMap::from([
                    (
                        "context_id".into(),
                        Value::Uuid(generation.context_id.as_uuid()),
                    ),
                    ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
                    ("connector_id".into(), Value::Text(connector.clone())),
                    ("source".into(), Value::Text(source.clone())),
                    ("session".into(), Value::Text(session.clone())),
                    (
                        "committed".into(),
                        integer(committed, "cursor committed sequence")?,
                    ),
                ]),
            )?;
            if let Some(candidate) = first.rows.first() {
                let candidate = PendingStream {
                    publication_event_count: sessions
                        .frontiers
                        .as_ref()
                        .and_then(|_| sessions.events.get(&key).copied()),
                    through_sequence,
                    start_ordinal: get_u64(candidate, &first.columns, "ordinal")?,
                    connector_id: connector,
                    source_instance_ref: source,
                    session_ref: session,
                    committed_sequence: committed,
                    open_tail_start_sequence: get_optional_u64(
                        row,
                        columns,
                        "open_tail_start_sequence",
                    )?,
                    open_tail_members: decode_tail_members(row, columns)?,
                    open_tail_passage_id: get_optional_uuid(row, columns, "open_tail_passage_id")?,
                };
                if pending.as_ref().is_none_or(|current: &PendingStream| {
                    let current_rank = sessions
                        .rank(&current.connector_id, &current.session_ref)
                        .expect("pending streams belong to selected sessions");
                    (candidate_rank, candidate.start_ordinal)
                        < (current_rank, current.start_ordinal)
                }) {
                    pending = Some(candidate);
                }
            }
            Ok(())
        },
    )?;
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    observe_event_search_discovery_for_test(stream_rows_examined, cursor_lookups);
    Ok(pending)
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
        SOURCE_POSITION_SESSION_STATEMENT,
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

fn decode_tail_members(row: &[Value], columns: &[String]) -> Result<Vec<uuid::Uuid>> {
    serde_json::from_value(get_json(row, columns, "open_tail_members")?)
        .map_err(|_| stale_claim("event-search open tail member IDs are invalid"))
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
    members: &[uuid::Uuid],
    bounds: Bounds,
) -> Result<Vec<ClaimedEvent>> {
    let Some(start) = open_tail_start else {
        return Ok(Vec::new());
    };
    if start > committed || members.is_empty() {
        return Err(stale_claim("event-search open tail is incomplete"));
    }
    let mut tail: Vec<ClaimedEvent> = Vec::with_capacity(members.len());
    for member in members {
        let result = query_rows(
            store,
            bounds,
            &format!(
                "SELECT {SOURCE_MEMBER_COLUMNS} FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND membership_id = $member"
            ),
            &HashMap::from([
                (
                    "context_id".into(),
                    Value::Uuid(generation.context_id.as_uuid()),
                ),
                ("generation_id".into(), Value::Uuid(generation.id.as_uuid())),
                ("member".into(), Value::Uuid(*member)),
            ]),
        )?;
        let [row] = result.rows.as_slice() else {
            return Err(stale_claim(
                "event-search open tail member disappeared or is not unique",
            ));
        };
        let event = load_claimed_member(store, generation, &result.columns, row, bounds)?;
        if event.membership_id != *member
            || event.event.connector_id != connector
            || event.event.source_instance_ref != source
            || event.event.session_ref != session
            || event.event.seq < start
            || event.event.seq > committed
            || event_body(&event.event).is_none()
            || tail.last().is_some_and(|prior| {
                prior.event.seq >= event.event.seq || prior.ordinal >= event.ordinal
            })
        {
            return Err(stale_claim(
                "event-search open tail member does not match its cursor",
            ));
        }
        tail.push(event);
    }
    if tail.first().map(|event| event.event.seq) != Some(start) {
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
        "SELECT id, context_id, generation_id, connector_id, session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_current, is_searchable, created_at FROM event_search_passages WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session AND is_current = $current AND id = $passage_id ORDER BY passage_ordinal ASC",
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
                        "is_searchable",
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
        HAS_LATER_SESSION_STATEMENT,
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
    embedding_workers: NonZeroUsize,
    renew: &mut dyn FnMut(i64) -> Result<i64>,
    observations: &mut RunObservations,
) -> Result<Rendered> {
    let embed_before = observations.embed;
    let mut whole_render_duration = Duration::ZERO;
    let result = measured(&mut whole_render_duration, || {
        render_and_embed_inner(
            claim,
            registration,
            profile,
            bounds,
            embedding_workers,
            renew,
            observations,
        )
    });
    observations.render_and_tokenize +=
        whole_render_duration.saturating_sub(observations.embed - embed_before);
    result
}

fn render_and_embed_inner(
    mut claim: Claim,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    embedding_workers: NonZeroUsize,
    renew: &mut dyn FnMut(i64) -> Result<i64>,
    observations: &mut RunObservations,
) -> Result<Rendered> {
    validate_profile_for_registration(profile, registration)?;
    let eligible = u64::try_from(
        claim
            .events
            .iter()
            .filter(|event| event_body(&event.event).is_some())
            .count(),
    )
    .map_err(|_| CgError::Engine("eligible event count exceeds u64".into()))?;
    let hard = usize::try_from(profile.hard_passage_tokens.get())
        .expect("u32 passage limits fit every supported usize");
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
    // Preparation and embedding can each outlast the lease. The same owned renewal reaches
    // oversized candidate selection; the renewed expiry is what the commit guard checks.
    let mut lease = LeaseRenewal {
        expires_at: claim.expires_at,
        renew,
    };
    // How many leading events rendered within the passage bound: a claim of just those never
    // exceeds it, because their trailing group stays open for the next claim.
    let mut fitting_events = 0_usize;
    for (index, event) in claim.events.iter().enumerate() {
        if passages.len() <= bounds.passages {
            fitting_events = index;
        }
        lease.keep()?;
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
            if passages.len() > bounds.passages && claim.events.len() > 1 {
                return Ok(over_passage_bound(&claim, fitting_events));
            }
            let slices = build_oversized_passages(
                &claim,
                claim.passage_ordinal
                    + u64::try_from(passages.len())
                        .map_err(|_| CgError::Engine("passage ordinal exceeds u64".into()))?,
                unit,
                registration,
                profile,
                hard,
                (claim.events.len() > 1).then(|| bounds.passages - passages.len()),
                &mut lease,
            )?;
            match slices {
                OversizedPassages::Complete(slices) => passages.extend(slices),
                OversizedPassages::OverPassageBound => {
                    return Ok(over_passage_bound(&claim, fitting_events));
                }
            }
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
                if passages.len() > bounds.passages && claim.events.len() > 1 {
                    return Ok(over_passage_bound(&claim, fitting_events));
                }
                // The unit fits alone, so a retained tail that crowds it out is shortened from
                // the front, down to no overlap at all, rather than failing legal input.
                while !ordinary.is_empty()
                    && prepared_group_tokens(registration, &ordinary, Some(&unit))? > hard
                {
                    ordinary.remove(0);
                }
            }
        }
        ordinary.push(unit);
    }
    let mut next_open_tail_members = ordinary
        .iter()
        .map(|unit| unit.event.membership_id)
        .collect::<Vec<_>>();
    let final_ordinary_start = ordinary.first().map(|unit| unit.event.event.seq);
    lease.keep()?;
    if claim.session_ended {
        emit_ordinary(
            &claim,
            &mut ordinary,
            &mut passages,
            registration,
            hard,
            ordinary_overlap,
            false,
        )?;
    }
    // The cursor is scoped to one session. A later source event from another session cannot
    // close this session's bounded tail: its next append must be able to replace that passage.
    let next_open_tail_start_sequence = final_ordinary_start;
    if next_open_tail_start_sequence.is_none() {
        next_open_tail_members.clear();
    }
    if passages.len() > bounds.passages && claim.events.len() > 1 {
        return Ok(over_passage_bound(&claim, fitting_events));
    }
    let passages = embed_passages(
        passages,
        &mut lease,
        registration,
        embedding_workers,
        bounds,
        observations,
    )?;
    claim.expires_at = lease.expires_at;
    Ok(Rendered::Batch(Box::new(BuiltBatch {
        claim,
        passages,
        eligible_events: eligible,
        excluded_events: excluded,
        next_open_tail_start_sequence,
        next_open_tail_members,
    })))
}

/// What rendering a claim produced.
enum Rendered {
    /// The claim's passages, embedded and ready to commit.
    Batch(Box<BuiltBatch>),
    /// The claim rendered more passages than the passage bound, so nothing was embedded; its first
    /// `fitting_events` events render within the bound and are what the next claim should take.
    /// A single event is never reported here: one oversized event is one claim however many
    /// slices it needs.
    OverPassageBound { fitting_events: usize },
}

fn over_passage_bound(claim: &Claim, fitting_events: usize) -> Rendered {
    Rendered::OverPassageBound {
        fitting_events: fitting_events.clamp(1, claim.events.len() - 1),
    }
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
    passages: &mut Vec<PendingPassage>,
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
) -> Result<PendingPassage> {
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
    remaining_allowance: Option<usize>,
    lease: &mut LeaseRenewal<'_>,
) -> Result<OversizedPassages> {
    let overlap =
        usize::try_from(profile.oversized_event_overlap_tokens.get()).unwrap_or(usize::MAX);
    lease.keep()?;
    let prepared_body = registration.embedder.prepare_document(&unit.body)?;
    let body_spans = prepared_body
        .byte_boundaries()
        .iter()
        .copied()
        .filter(|(start, end)| {
            observe_preparation_span_work();
            start < end
                && *end <= unit.body.len()
                && unit.body.is_char_boundary(*start)
                && unit.body.is_char_boundary(*end)
        })
        .collect::<Vec<_>>();
    let spans = SourceSpanIndex::new(&body_spans);
    let mut candidate_ends = body_spans
        .iter()
        .map(|(_, end)| {
            observe_preparation_span_work();
            *end
        })
        .chain(std::iter::once(unit.body.len()))
        .filter(|end| *end <= unit.body.len() && unit.body.is_char_boundary(*end))
        .collect::<Vec<_>>();
    if !spans.ordered {
        candidate_ends.sort_unstable();
    }
    candidate_ends.dedup();
    let span_limit = registration
        .embedder
        .event_search_slice_token_loss_bound(&unit.prefix)
        .and_then(|loss| hard.checked_add(loss));
    let mut start = 0_usize;
    let mut passages = Vec::new();
    while start < unit.body.len() {
        lease.keep()?;
        let first_span = spans.first_at_or_after(start);
        // Only the ORIGINAL span count is monotone. The first span above hard + loss
        // supplies an exclusive endpoint ceiling; duplicate spans each consume one count.
        let ceiling = span_limit.and_then(|limit| spans.excluded_end(first_span, limit));
        let retained_end = ceiling.map_or(candidate_ends.len(), |ceiling| {
            candidate_ends.partition_point(|end| *end < ceiling)
        });
        let retained_start = candidate_ends[..retained_end].partition_point(|end| *end <= start);
        let mut selected = None;
        for &end in candidate_ends[retained_start..retained_end].iter().rev() {
            lease.keep()?;
            let text = format!("{}{}", unit.prefix, &unit.body[start..end]);
            let prepared = registration.embedder.prepare_document(&text)?;
            if prepared.token_count() <= hard {
                selected = Some((end, text, prepared));
                break;
            }
        }
        let Some((end, text, prepared)) = selected else {
            return Err(CgError::Engine(
                "event-search renderer label leaves no model token for source text".into(),
            ));
        };
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
        passages.push(finish_prepared_passage(
            claim,
            ordinal,
            text,
            vec![source],
            prepared,
            hard,
        )?);
        if remaining_allowance.is_some_and(|allowance| passages.len() > allowance) {
            return Ok(OversizedPassages::OverPassageBound);
        }
        if end == unit.body.len() {
            break;
        }
        let next = spans
            .overlap_start(first_span, start, end, overlap)
            .ok_or_else(|| CgError::Engine("event-search oversized overlap underflow".into()))?;
        if next <= start {
            return Err(CgError::Engine(
                "event-search oversized overlap prevents forward progress".into(),
            ));
        }
        start = next;
    }
    Ok(OversizedPassages::Complete(passages))
}

enum OversizedPassages {
    Complete(Vec<PendingPassage>),
    OverPassageBound,
}

/// An index over the original spans, retaining duplicates and their original overlap ordering.
/// Both endpoints must be monotone. External preparers with another ordering retain the
/// exhaustive candidate and overlap paths; sorting their spans would change the overlap.
struct SourceSpanIndex<'a> {
    spans: &'a [(usize, usize)],
    ordered: bool,
}

impl<'a> SourceSpanIndex<'a> {
    fn new(spans: &'a [(usize, usize)]) -> Self {
        let ordered = spans.windows(2).all(|pair| {
            observe_preparation_span_work();
            pair[0].0 <= pair[1].0 && pair[0].1 <= pair[1].1
        });
        Self { spans, ordered }
    }

    fn first_at_or_after(&self, start: usize) -> usize {
        if !self.ordered {
            return 0;
        }
        self.spans.partition_point(|&(s, _)| {
            observe_preparation_span_work();
            s < start
        })
    }

    fn excluded_end(&self, first: usize, limit: usize) -> Option<usize> {
        if !self.ordered {
            return None;
        }
        observe_preparation_span_work();
        self.spans
            .get(first.checked_add(limit)?)
            .map(|&(_, end)| end)
    }

    fn overlap_start(
        &self,
        first: usize,
        start: usize,
        end: usize,
        overlap: usize,
    ) -> Option<usize> {
        if self.ordered {
            let past = self.spans.partition_point(|&(_, e)| {
                observe_preparation_span_work();
                e <= end
            });
            let contained = self.spans.get(first..past)?;
            contained
                .get(contained.len().checked_sub(overlap)?)
                .map(|&(s, _)| s)
        } else {
            let contained = self
                .spans
                .iter()
                .filter(|&&(s, e)| {
                    observe_preparation_span_work();
                    s >= start && e <= end
                })
                .collect::<Vec<_>>();
            contained
                .get(contained.len().checked_sub(overlap)?)
                .map(|&&(s, _)| s)
        }
    }
}

#[cfg(all(test, context_graph_internal_test_support))]
thread_local! {
    static PREPARATION_SPAN_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
fn observe_preparation_span_work() {
    #[cfg(all(test, context_graph_internal_test_support))]
    PREPARATION_SPAN_WORK.with(|work| work.set(work.get() + 1));
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
    sources: Vec<PassageEventSource>,
    registration: &crate::EmbedderRegistration,
    hard: usize,
) -> Result<PendingPassage> {
    let prepared = registration.embedder.prepare_document(&text)?;
    finish_prepared_passage(claim, ordinal, text, sources, prepared, hard)
}

fn finish_prepared_passage(
    claim: &Claim,
    ordinal: u64,
    text: String,
    mut sources: Vec<PassageEventSource>,
    prepared: crate::embedder::PreparedEventSearchDocument,
    hard: usize,
) -> Result<PendingPassage> {
    if prepared.token_count() > hard {
        return Err(CgError::Engine(format!(
            "rendered event-search passage has {} tokens, exceeding resolved hard limit {hard}",
            prepared.token_count()
        )));
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
    Ok(PendingPassage {
        passage: BuiltPassage {
            id,
            embedding_id: EmbeddingId::new_v7(),
            ordinal,
            first_sequence,
            last_sequence,
            content_sha256: sha256(text.as_bytes()),
            text,
            vector: Vec::new(),
            dimension: 0,
            producer: crate::EvidenceProducer {
                system: String::new(),
                model_name: String::new(),
                model_version: String::new(),
                pipeline_version: String::new(),
            },
            sources,
        },
        prepared,
    })
}

/// A rendered passage and its prepared model input, waiting for its forward call.
struct PendingPassage {
    passage: BuiltPassage,
    prepared: crate::embedder::PreparedEventSearchDocument,
}

/// The most passages one forward call embeds. Bounds the padded batch the model holds at once
/// (activations grow with batch size and with the square of the longest passage), and so how long
/// any single call can run between lease renewals.
const PASSAGES_PER_FORWARD_CALL: usize = 1;

/// Embed a claim's rendered passages in bounded waves of scoped forward calls, one prepared
/// passage per call. Work is admitted in prepared-length order while results return to their
/// rendered-order slots. The coordinator renews the claim while each wave is running and joins the
/// whole wave before it either admits more work or returns an error.
fn embed_passages(
    pending: Vec<PendingPassage>,
    lease: &mut LeaseRenewal<'_>,
    registration: &crate::EmbedderRegistration,
    embedding_workers: NonZeroUsize,
    bounds: Bounds,
    observations: &mut RunObservations,
) -> Result<Vec<BuiltPassage>> {
    let mut order = (0..pending.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| pending[*index].prepared.token_count());
    let helper_pools = concurrent_embedding_helper_pools(embedding_workers, pending.len())?;
    let result = measured(&mut observations.embed, || {
        embed_passages_in_waves(
            &pending,
            &order,
            lease,
            registration,
            embedding_workers,
            bounds,
            &mut observations.embedding_invocations,
            helper_pools.as_deref(),
        )
    });
    let outputs = result?;

    pending
        .into_iter()
        .zip(outputs)
        .map(|(pending, output)| {
            let output = output
                .ok_or_else(|| CgError::Engine("event-search passage was not embedded".into()))?;
            validate_passage_embedding(output, pending, registration)
        })
        .collect()
}

/// Construct one reusable single-thread Rayon pool for every concurrent passage worker.
///
/// The serial runner retains its existing embedder call, including any helper threading the
/// embedder itself uses. Concurrent waves install each call in one of these local pools so nested
/// Rayon work cannot join the process-global pool and multiply the requested passage concurrency.
fn concurrent_embedding_helper_pools(
    embedding_workers: NonZeroUsize,
    pending_passages: usize,
) -> Result<Option<Vec<rayon::ThreadPool>>> {
    let active_workers = embedding_workers.get().min(pending_passages);
    if active_workers <= 1 {
        return Ok(None);
    }

    let mut pools = Vec::with_capacity(active_workers);
    for _ in 0..active_workers {
        pools.push(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .map_err(|error| {
                    CgError::Engine(format!(
                        "event-search embedding helper pool could not start: {error}"
                    ))
                })?,
        );
    }
    Ok(Some(pools))
}

#[allow(clippy::too_many_arguments)]
fn embed_passages_in_waves(
    pending: &[PendingPassage],
    order: &[usize],
    lease: &mut LeaseRenewal<'_>,
    registration: &crate::EmbedderRegistration,
    embedding_workers: NonZeroUsize,
    bounds: Bounds,
    entered_calls: &mut u64,
    helper_pools: Option<&[rayon::ThreadPool]>,
) -> Result<Vec<Option<crate::EmbeddingOutput>>> {
    let mut outputs = (0..pending.len()).map(|_| None).collect::<Vec<_>>();
    for wave in order.chunks(embedding_workers.get()) {
        lease.keep()?;
        run_embedding_wave(
            pending,
            wave,
            lease,
            registration,
            bounds,
            entered_calls,
            helper_pools,
            &mut outputs,
        )?;
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
fn run_embedding_wave(
    pending: &[PendingPassage],
    wave: &[usize],
    lease: &mut LeaseRenewal<'_>,
    registration: &crate::EmbedderRegistration,
    bounds: Bounds,
    entered_calls: &mut u64,
    helper_pools: Option<&[rayon::ThreadPool]>,
    outputs: &mut [Option<crate::EmbeddingOutput>],
) -> Result<()> {
    let mut first_worker_error = None;
    let mut first_lease_error = None;
    let mut panic = None;
    std::thread::scope(|scope| {
        let (completed, completions) = sync_channel(wave.len());
        let mut handles = Vec::with_capacity(wave.len());
        for (worker, &index) in wave.iter().enumerate() {
            let completed = completed.clone();
            let prepared = &pending[index].prepared;
            let helper_pool = helper_pools.map(|pools| &pools[worker]);
            let spawn = std::thread::Builder::new()
                .name("event-search-embedding".into())
                .spawn_scoped(scope, move || {
                    let result = match helper_pool {
                        Some(pool) => pool.install(|| {
                            registration
                                .embedder
                                .embed_prepared_documents(std::slice::from_ref(prepared))
                        }),
                        None => registration
                            .embedder
                            .embed_prepared_documents(std::slice::from_ref(prepared)),
                    };
                    let _ = completed.send((index, result));
                });
            match spawn {
                Ok(handle) => {
                    // A successful spawn unconditionally enters its call. All handles are joined,
                    // including on error, panic and stale lease, before this count is observed.
                    *entered_calls += 1;
                    handles.push(handle);
                }
                Err(error) => {
                    first_worker_error = Some(CgError::Engine(format!(
                        "event-search embedding worker could not start: {error}"
                    )));
                    break;
                }
            }
        }
        #[cfg(feature = "test-support")]
        WAVE_OBSERVER.with(|observer| {
            if let Some(observer) = observer.borrow_mut().as_mut() {
                observer(handles.len());
            }
        });
        drop(completed);

        let mut received = 0_usize;
        while received < handles.len() {
            match completions.recv_timeout(lease_poll_interval(bounds)) {
                Ok((index, result)) => {
                    received += 1;
                    match result {
                        Ok(mut result) if result.len() == PASSAGES_PER_FORWARD_CALL => {
                            if first_worker_error.is_none() {
                                let output = result.pop().expect("one checked passage vector");
                                match validate_passage_embedding_output(&output, registration) {
                                    Ok(()) if first_lease_error.is_none() => {
                                        outputs[index] = Some(output);
                                    }
                                    Ok(()) => {}
                                    Err(error) => {
                                        first_worker_error = Some(error);
                                    }
                                }
                            }
                        }
                        Ok(_) => {
                            first_worker_error.get_or_insert_with(|| {
                                CgError::Engine(
                                    "event-search embedder returned the wrong number of passage vectors"
                                        .into(),
                                )
                            });
                        }
                        Err(error) => {
                            first_worker_error.get_or_insert(error);
                        }
                    }
                    if let Err(error) = lease.keep() {
                        first_lease_error.get_or_insert(error);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if let Err(error) = lease.keep() {
                        first_lease_error.get_or_insert(error);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        for handle in handles {
            if let Err(payload) = handle.join()
                && panic.is_none()
            {
                panic = Some(payload);
            }
        }
    });
    if let Some(payload) = panic {
        std::panic::resume_unwind(payload);
    }
    first_worker_error.or(first_lease_error).map_or(Ok(()), Err)
}

fn lease_poll_interval(bounds: Bounds) -> Duration {
    Duration::from_millis((bounds.lease_ms / 6).max(1))
}

fn validate_passage_embedding(
    output: crate::EmbeddingOutput,
    pending: PendingPassage,
    registration: &crate::EmbedderRegistration,
) -> Result<BuiltPassage> {
    validate_passage_embedding_output(&output, registration)?;
    let mut passage = pending.passage;
    passage.dimension = u32::try_from(output.dimension)
        .map_err(|_| CgError::Engine("event-search embedding dimension exceeds u32".into()))?;
    passage.vector = output.vector;
    passage.producer = output.producer;
    Ok(passage)
}

fn validate_passage_embedding_output(
    output: &crate::EmbeddingOutput,
    registration: &crate::EmbedderRegistration,
) -> Result<()> {
    if output.embedding_space_id != registration.space.embedding_space_id
        || output.dimension != registration.space.dimension
        || output.vector.len() != output.dimension
        || output.vector.iter().any(|value| !value.is_finite())
    {
        return Err(CgError::Engine(
            "event-search embedder returned an invalid passage vector".into(),
        ));
    }
    u32::try_from(output.dimension)
        .map_err(|_| CgError::Engine("event-search embedding dimension exceeds u32".into()))?;
    Ok(())
}

fn commit_batch(
    store: &Store,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    built: BuiltBatch,
    observations: &mut RunObservations,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let outcome = commit_batch_in_tx(
        store,
        registration,
        profile,
        bounds,
        tx,
        &built,
        observations,
    );
    if let Err(error) = outcome {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) = BEFORE_BATCH_COMMIT.with(|slot| slot.borrow_mut().take()) {
        hook(store);
    }
    measured(&mut observations.commit, || {
        store.internal_db().commit(tx).map_err(|error| match error {
            contextdb_core::Error::ConditionalUpdateConflict { .. } => {
                stale_claim("event-search claim changed before commit")
            }
            other => map_engine_err(other),
        })
    })?;
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) = AFTER_BATCH_COMMIT.with(|slot| slot.borrow_mut().take()) {
        hook(store);
    }
    Ok(())
}

fn commit_batch_in_tx(
    store: &Store,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    tx: TxId,
    built: &BuiltBatch,
    observations: &mut RunObservations,
) -> Result<()> {
    let claim = &built.claim;
    guard_attempt_in_tx(store, tx, &claim.generation)?;
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
        "SELECT context_id, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, required_event_count, searchable_event_count, searchable_excluded_event_count, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, passage_count FROM event_search_generations WHERE id = $id",
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
            "required_event_count",
            "searchable_event_count",
            "searchable_excluded_event_count",
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

    let progress = store.internal_db().execute_in_tx(
        tx,
        "SELECT context_id, generation_id, connector_id, session_ref, event_count, committed_event_count, eligible_event_count, excluded_event_count, searchable_event_count, no_searchable_text_event_count, is_required, is_published FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND session_ref = $session",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(claim.generation.context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
            ("connector_id".into(), Value::Text(claim.connector_id.clone())),
            ("session".into(), Value::Text(claim.session_ref.clone())),
        ]),
    ).map_err(map_engine_err)?;
    let progress_row = progress.rows.first().ok_or_else(|| {
        stale_claim("event-search claimed session has no transactional progress row")
    })?;
    if progress.rows.len() != 1 || !get_bool(progress_row, &progress.columns, "is_required")? {
        return Err(stale_claim(
            "event-search claimed session is no longer in the required scope",
        ));
    }
    let session_event_count = get_u64(progress_row, &progress.columns, "event_count")?;
    let prior_committed = get_u64(progress_row, &progress.columns, "committed_event_count")?;
    let prior_eligible = get_u64(progress_row, &progress.columns, "eligible_event_count")?;
    let prior_excluded = get_u64(progress_row, &progress.columns, "excluded_event_count")?;
    let prior_searchable = get_u64(progress_row, &progress.columns, "searchable_event_count")?;
    let prior_no_text = get_u64(
        progress_row,
        &progress.columns,
        "no_searchable_text_event_count",
    )?;
    let session_was_published = get_bool(progress_row, &progress.columns, "is_published")?;
    let committed = prior_committed
        .checked_add(built.eligible_events)
        .and_then(|count| count.checked_add(built.excluded_events))
        .ok_or_else(|| CgError::Engine("session committed count exhausted u64".into()))?;
    if committed > session_event_count {
        return Err(stale_claim(
            "event-search session progress exceeded its captured membership",
        ));
    }
    let session_eligible = prior_eligible
        .checked_add(built.eligible_events)
        .ok_or_else(|| CgError::Engine("session eligible count exhausted u64".into()))?;
    let session_excluded = prior_excluded
        .checked_add(built.excluded_events)
        .ok_or_else(|| CgError::Engine("session excluded count exhausted u64".into()))?;
    let session_is_published = session_was_published
        || committed >= claim.publication_event_count.unwrap_or(session_event_count);
    let (session_searchable, session_no_text) = if session_is_published {
        (session_eligible, session_excluded)
    } else {
        (prior_searchable, prior_no_text)
    };

    let replacement_is_searchable = session_was_published;
    for passage in &built.passages {
        measured(&mut observations.vector_insert, || {
            store.internal_db().execute_in_tx(
            tx,
            "INSERT INTO event_search_passages (id, context_id, generation_id, connector_id, session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_current, is_searchable, created_at) VALUES ($id, $context_id, $generation_id, $connector_id, $session, $source, $ordinal, $first, $last, $text, $sha, $vector, $dimension, $current, $searchable, $created_at)",
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
                (
                    "searchable".into(),
                    Value::Bool(replacement_is_searchable),
                ),
                ("created_at".into(), Value::TxId(tx)),
            ]),
        ).map_err(map_engine_err)
        })?;
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
        &built.next_open_tail_members,
        next_open_tail_passage_id,
    )?;
    store.internal_db().execute_in_tx(
        tx,
        "UPDATE event_search_session_progress SET committed_event_count = $committed, eligible_event_count = $eligible, excluded_event_count = $excluded, searchable_event_count = $searchable, no_searchable_text_event_count = $no_text, is_published = $published WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND session_ref = $session",
        &HashMap::from([
            ("committed".into(), integer(committed, "session committed count")?),
            ("eligible".into(), integer(session_eligible, "session eligible count")?),
            ("excluded".into(), integer(session_excluded, "session excluded count")?),
            ("searchable".into(), integer(session_searchable, "session searchable count")?),
            ("no_text".into(), integer(session_no_text, "session no-text count")?),
            ("published".into(), Value::Bool(session_is_published)),
            (
                "context_id".into(),
                Value::Uuid(claim.generation.context_id.as_uuid()),
            ),
            (
                "generation_id".into(),
                Value::Uuid(claim.generation.id.as_uuid()),
            ),
            ("connector_id".into(), Value::Text(claim.connector_id.clone())),
            ("session".into(), Value::Text(claim.session_ref.clone())),
        ]),
    ).map_err(map_engine_err)?;
    if session_is_published {
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_search_passages SET is_searchable = $searchable WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND session_ref = $session",
                &HashMap::from([
                    ("searchable".into(), Value::Bool(true)),
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
                    ("session".into(), Value::Text(claim.session_ref.clone())),
                ]),
            )
            .map_err(map_engine_err)?;
        if !claim.generation.pending_dual_head_activation {
            activate_search_head_for_published_session_in_tx(
                store,
                tx,
                claim.generation.context_id,
                claim.generation.id,
            )?;
        }
    }
    let eligible = get_u64(generation_row, &generation.columns, "eligible_event_count")?
        .checked_add(built.eligible_events)
        .ok_or_else(|| CgError::Engine("eligible event count exhausted u64".into()))?;
    let excluded = get_u64(generation_row, &generation.columns, "excluded_event_count")?
        .checked_add(built.excluded_events)
        .ok_or_else(|| CgError::Engine("excluded event count exhausted u64".into()))?;
    let searchable = get_u64(
        generation_row,
        &generation.columns,
        "searchable_event_count",
    )?
    .checked_add(session_searchable.saturating_sub(prior_searchable))
    .ok_or_else(|| CgError::Engine("searchable event count exhausted u64".into()))?;
    let searchable_excluded = get_u64(
        generation_row,
        &generation.columns,
        "searchable_excluded_event_count",
    )?
    .checked_add(session_no_text.saturating_sub(prior_no_text))
    .ok_or_else(|| CgError::Engine("searchable excluded count exhausted u64".into()))?;
    let passage_count = get_u64(generation_row, &generation.columns, "passage_count")?
        .checked_add(
            u64::try_from(built.passages.len())
                .map_err(|_| CgError::Engine("passage count exceeds u64".into()))?,
        )
        .ok_or_else(|| CgError::Engine("passage count exhausted u64".into()))?;
    store.internal_db().execute_in_tx(
        tx,
        "UPDATE event_search_generations SET searchable_event_count = $searchable, searchable_excluded_event_count = $searchable_excluded, eligible_event_count = $eligible, rendered_event_count = $eligible, embedded_event_count = $eligible, excluded_event_count = $excluded, passage_count = $passage_count WHERE id = $id",
        &HashMap::from([
            ("searchable".into(), integer(searchable, "searchable count")?),
            (
                "searchable_excluded".into(),
                integer(searchable_excluded, "searchable excluded count")?,
            ),
            ("eligible".into(), integer(eligible, "eligible count")?),
            ("excluded".into(), integer(excluded, "excluded count")?),
            ("passage_count".into(), integer(passage_count, "passage count")?),
            ("id".into(), Value::Uuid(claim.generation.id.as_uuid())),
        ]),
    ).map_err(map_engine_err)?;
    let deleted = store.internal_db().execute_in_tx(
        tx,
        "DELETE FROM event_search_leases WHERE context_id = $context_id AND generation_id = $generation_id AND owner_token = $owner_token AND lease_epoch = $lease_epoch",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(claim.generation.context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
            ("owner_token".into(), Value::Text(claim.owner_token.clone())),
            ("lease_epoch".into(), Value::Int64(claim.lease_epoch)),
        ]),
    ).map_err(map_engine_err)?;
    if deleted.rows_affected != 1 {
        return Err(stale_claim(
            "event-search owned claim disappeared before its committed lease deletion",
        ));
    }
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
    next_open_tail_members: &[uuid::Uuid],
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
        ("session".into(), Value::Text(claim.session_ref.clone())),
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
        (
            "open_tail_members".into(),
            Value::Json(json!(next_open_tail_members)),
        ),
        ("updated_at".into(), Value::TxId(tx)),
    ]);
    let existing = store.internal_db().execute_in_tx(
        tx,
        "SELECT committed_sequence, open_tail_start_sequence, open_tail_members, open_tail_passage_id FROM event_search_cursors WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session",
        &params,
    ).map_err(map_engine_err)?;
    let Some(row) = existing.rows.first() else {
        return Err(stale_claim("event-search cursor disappeared before commit"));
    };
    if get_u64(row, &existing.columns, "committed_sequence")? != claim.prior_committed_sequence
        || get_optional_u64(row, &existing.columns, "open_tail_start_sequence")?
            != claim.prior_open_tail_start_sequence
        || decode_tail_members(row, &existing.columns)? != claim.prior_open_tail_members
        || get_optional_uuid(row, &existing.columns, "open_tail_passage_id")?
            != claim.prior_open_tail_passage_id
    {
        return Err(stale_claim(
            "event-search cursor or open tail changed before commit",
        ));
    }
    store
        .internal_db()
        .execute_in_tx(
            tx,
            "UPDATE event_search_cursors SET committed_sequence = $committed, open_tail_start_sequence = $open_tail, open_tail_members = $open_tail_members, open_tail_passage_id = $open_tail_passage, watermark_sha256 = $watermark, updated_at = $updated_at WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector_id AND source_instance_ref = $source AND session_ref = $session",
            &params,
        )
        .map_err(map_engine_err)?;
    if guard_event_context_revision_in_tx(store, tx, claim.generation.context_id)?.is_none() {
        return Err(stale_claim("capture changed while clearing pending stream"));
    }
    let newest = store.internal_db().execute_in_tx(tx,
        "SELECT seq FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source ORDER BY seq DESC LIMIT 1", &params).map_err(map_engine_err)?;
    let row = newest
        .rows
        .first()
        .ok_or_else(|| stale_claim("claimed source disappeared before pending marker update"))?;
    let pending = claim.end_sequence < get_u64(row, &newest.columns, "seq")?;
    for target in [claim.generation.id.as_uuid(), UNASSIGNED_SEARCH_GENERATION] {
        set_stream_pending_in_tx(
            store,
            tx,
            claim.generation.context_id,
            target,
            &claim.connector_id,
            &claim.source_instance_ref,
            &claim.session_ref,
            pending,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finish_or_continue(
    store: &Store,
    generation: &Generation,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    selection: InterpretationSelection,
    retry_count: u64,
    observations: &mut RunObservations,
) -> Result<FinishResult> {
    let mut connectors = event_connectors(store, generation.context_id)?;
    let explicit_connector = match selection {
        InterpretationSelection::Ordinary => None,
        InterpretationSelection::ExplicitStaged(id) => {
            let connector = explicit_staged_connector(store, generation.context_id, id)?;
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
    )?;
    if hash_json(&snapshot_json(&bindings)) != generation.snapshot_sha256 {
        return Ok(FinishResult::Rebuild);
    }
    // The run's certified capture still describes the Context when neither its event revision nor
    // the bound interpretations moved; anything else captures again. Completion re-checks the
    // revision under its own guard.
    let unchanged = match &generation.certified {
        Some(certificate) => {
            read_event_context_revision(store, generation.context_id, bounds)?
                == certificate.revision
                && current_interpretation_stamp(store, generation.context_id, &bindings, bounds)?
                    == certificate.stamp
        }
        None => false,
    };
    let (target, certified) = if unchanged {
        (generation.target.clone(), generation.certified.clone())
    } else {
        measured(&mut observations.event_read, || {
            capture_source_membership(
                store,
                generation.context_id,
                &bindings,
                Some(generation.id),
                bounds,
            )
        })?
    };
    let mut next = generation.clone();
    next.bindings = bindings;
    next.target = target;
    next.certified = certified;
    if next.target != generation.target {
        return Ok(FinishResult::Continue(Box::new(next)));
    }
    match finish_stable(
        store,
        &next,
        registration,
        profile,
        bounds,
        retry_count,
        observations,
    )? {
        Some(receipt) => Ok(FinishResult::Ready(Box::new(receipt))),
        None => Ok(FinishResult::Continue(Box::new(next))),
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_stable(
    store: &Store,
    generation: &Generation,
    registration: &crate::EmbedderRegistration,
    profile: &EventSearchProfile,
    bounds: Bounds,
    retry_count: u64,
    observations: &mut RunObservations,
) -> Result<Option<EventSearchBuildReceipt>> {
    let tx = store.begin_tx()?;
    if let Err(error) = guard_attempt_in_tx(store, tx, generation) {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    let result = store.internal_db().execute_in_tx(
        tx,
        "SELECT context_id, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, required_event_count, searchable_event_count, searchable_excluded_event_count, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, completed_at FROM event_search_generations WHERE id = $id",
        &HashMap::from([("id".into(), Value::Uuid(generation.id.as_uuid()))]),
    ).map_err(map_engine_err)?;
    let row = result.rows.first().ok_or(CgError::NotFound {
        id: Some(generation.id.to_string()),
        missing: Some("event search generation"),
    })?;
    let eligible = get_u64(row, &result.columns, "eligible_event_count")?;
    let excluded = get_u64(row, &result.columns, "excluded_event_count")?;
    let required = get_u64(row, &result.columns, "required_event_count")?;
    let searchable = get_u64(row, &result.columns, "searchable_event_count")?;
    let searchable_excluded = get_u64(row, &result.columns, "searchable_excluded_event_count")?;
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
        || eligible.checked_add(excluded) != Some(required)
        || searchable.checked_add(searchable_excluded) != Some(required)
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
            "required_event_count",
            "searchable_event_count",
            "searchable_excluded_event_count",
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
    // Never finalize past a revision the certified capture did not see: the next pass captures
    // the newer events first.
    if generation
        .certified
        .as_ref()
        .is_some_and(|certificate| certificate.revision != event_revision)
    {
        let _ = store.internal_db().rollback(tx);
        return Ok(None);
    }
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
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(pause) = FINALIZATION_PAUSE.with(|slot| slot.borrow_mut().take()) {
        pause(store);
    }
    let unchanged_ready = generation.mode == GenerationMode::ExtendReady
        && generation.target.event_count == generation.event_count
        && generation.target.ordered_source_set_sha256 == generation.ordered_source_set_sha256
        && generation.target.source_content_sha256 == generation.source_content_sha256;
    if unchanged_ready {
        get_optional_txid(row, &result.columns, "completed_at")?.ok_or_else(|| {
            stale_claim("event-search ready generation has no completion transaction")
        })?;
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
    }
    let finalized = store.internal_db().execute_in_tx(
        tx,
        "UPDATE event_search_generations SET finalized_event_revision_tx = $revision_tx, sanitized_failure = NULL WHERE id = $id",
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
        activate_search_head_in_tx(store, tx, generation.context_id, generation.id)?;
        // A first generation may have published its first complete session before the full
        // required scope sealed. This completion still reports that the generation became the
        // active head, even when the final call observes that same head already installed.
        true
    };
    let rendered = get_u64(row, &result.columns, "rendered_event_count")?;
    let embedded = get_u64(row, &result.columns, "embedded_event_count")?;
    let failed = get_u64(row, &result.columns, "failed_event_count")?;
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) = BEFORE_COMPLETION_COMMIT.with(|slot| slot.borrow_mut().take()) {
        hook(store);
    }
    let commit_result = measured(&mut observations.commit, || store.internal_db().commit(tx));
    match commit_result {
        Ok(()) => {}
        Err(EngineError::ConditionalUpdateConflict { .. }) => {
            let _ = store.internal_db().rollback(tx);
            return Ok(None);
        }
        Err(error) => return Err(map_engine_err(error)),
    }
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    if let Some(hook) =
        AFTER_COMPLETION_COMMIT_BEFORE_READBACK.with(|slot| slot.borrow_mut().take())
    {
        hook(store);
    }
    let completed_at = read_committed_completion(store, generation)?;
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
            event_count: required,
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

/// Publish the first completed session of a first build, or retain an already active generation
/// while it extends. A replacement generation keeps the prior ready head until finalization has
/// validated the whole replacement and performs the ordinary head switch.
fn activate_search_head_for_published_session_in_tx(
    store: &Store,
    tx: TxId,
    context_id: crate::ContextId,
    generation_id: EventSearchGenerationId,
) -> Result<()> {
    let existing = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    if existing.rows.is_empty()
        || existing.rows.first().is_some_and(|row| {
            get_uuid(row, &existing.columns, "generation_id").ok() == Some(generation_id.as_uuid())
        })
    {
        activate_search_head_in_tx(store, tx, context_id, generation_id)?;
    }
    Ok(())
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
            "SELECT context_id, interpretation_generation_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, state, required_event_count, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, completed_at, finalized_event_revision_tx, sanitized_failure FROM event_search_generations WHERE id = $id",
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
    let required_event_count = get_u64(row, columns, "required_event_count")?;
    let eligible = get_u64(row, columns, "eligible_event_count")?;
    if get_u64(row, columns, "failed_event_count")? != 0
        || get_u64(row, columns, "capture_ordinal")? != event_count
        || eligible.checked_add(get_u64(row, columns, "excluded_event_count")?)
            != Some(required_event_count)
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
        .read_session(read_limits(store, bounds))
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
        .read_session(read_limits(store, bounds))
        .map_err(map_engine_err)?;
    session.execute(sql, params).map_err(map_engine_err)
}

fn read_limits(store: &Store, bounds: Bounds) -> ReadLimits {
    let mut limits = store.read_limits();
    #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
    PROJECTION_READ_ACTIVE_MS.with(|slot| {
        if let Some(active_ms) = slot.get() {
            limits.active_ms = active_ms;
            limits.cursor_lifetime_ms = active_ms;
        }
    });
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

#[cfg(all(test, context_graph_internal_test_support))]
mod bounded_preparation_tests {
    use super::*;
    use crate::embedder::{EventSearchCapability, EventSearchSeal, PreparedEventSearchDocument};
    use crate::{Embedder, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EvidenceKind};
    use std::sync::Arc;
    use std::sync::mpsc::{self, Receiver, Sender};

    #[derive(Default, Debug, Clone, Copy)]
    struct PreparationWork {
        calls: usize,
        bytes: usize,
        embeds: usize,
    }

    struct ScalarPreparer {
        work: Sender<PreparationWork>,
        nonmonotone: bool,
        unordered: bool,
        forbidden: Option<&'static str>,
    }

    fn scalar_preparer(
        nonmonotone: bool,
        unordered: bool,
        forbidden: Option<&'static str>,
    ) -> (Arc<ScalarPreparer>, Receiver<PreparationWork>) {
        let (work, receiver) = mpsc::channel();
        (
            Arc::new(ScalarPreparer {
                work,
                nonmonotone,
                unordered,
                forbidden,
            }),
            receiver,
        )
    }

    fn preparation_work(receiver: &Receiver<PreparationWork>) -> PreparationWork {
        receiver
            .try_iter()
            .fold(PreparationWork::default(), |mut total, delta| {
                total.calls += delta.calls;
                total.bytes += delta.bytes;
                total.embeds += delta.embeds;
                total
            })
    }

    impl Embedder for ScalarPreparer {
        fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
            panic!("plain inference is never authorized by these preparation proofs")
        }
        fn supported_inputs(&self) -> Vec<EvidenceKind> {
            vec![EvidenceKind::Text]
        }
        fn embedding_space(&self) -> EmbeddingSpace {
            crate::embedder::mock::MockEmbedder::new("preparation-proof", 8).embedding_space()
        }
        fn event_search_capability(&self) -> Option<EventSearchCapability> {
            EventSearchCapability::new(8192)
        }
        fn prepare_document_hook(
            &self,
            text: &str,
            seal: &EventSearchSeal,
        ) -> Result<PreparedEventSearchDocument> {
            if let Some(forbidden) = self.forbidden {
                assert!(
                    !text.contains(forbidden),
                    "prepared an event beyond overflow"
                );
            }
            self.work
                .send(PreparationWork {
                    calls: 1,
                    bytes: text.len(),
                    embeds: 0,
                })
                .expect("preparation proof receiver");
            let mut spans = vec![(0, 0); 5];
            for (start, ch) in text.char_indices() {
                spans.extend([(start, start + ch.len_utf8()); 2]);
            }
            if self.nonmonotone && text.starts_with("User") && text.ends_with('!') {
                // A farther endpoint fits again after many nearer endpoints do not. There is
                // deliberately no slice-loss declaration for this tokenizer.
                spans.truncate(5);
            }
            if self.unordered {
                // Permute adjacent pairs, preserving duplicates and forward overlap progress.
                for group in spans[5..].chunks_mut(4) {
                    group.reverse();
                }
            }
            seal.document_builder()
                .token_count(spans.len())
                .payload(vec![1; spans.len()])
                .byte_boundaries(spans)
                .seal()
        }
        fn embed_prepared_document_hook(
            &self,
            inference: &crate::embedder::EventSearchInference,
        ) -> Result<EmbeddingOutput> {
            self.work
                .send(PreparationWork {
                    calls: 0,
                    bytes: 0,
                    embeds: 1,
                })
                .expect("preparation proof receiver");
            let space = self.embedding_space();
            Ok(EmbeddingOutput {
                embedding_space_id: space.embedding_space_id,
                dimension: space.dimension,
                producer: crate::EvidenceProducer {
                    system: "preparation-proof".into(),
                    model_name: "scalar".into(),
                    model_version: "1".into(),
                    pipeline_version: "1".into(),
                },
                vector: vec![inference.token_count() as f32; space.dimension],
            })
        }
    }

    struct BoundedScalarPreparer {
        inner: Arc<ScalarPreparer>,
        loss: usize,
    }

    impl Embedder for BoundedScalarPreparer {
        fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
            self.inner.embed(input)
        }
        fn supported_inputs(&self) -> Vec<EvidenceKind> {
            self.inner.supported_inputs()
        }
        fn embedding_space(&self) -> EmbeddingSpace {
            self.inner.embedding_space()
        }
        fn event_search_capability(&self) -> Option<EventSearchCapability> {
            self.inner.event_search_capability()
        }
        fn event_search_slice_token_loss_bound(&self, _: &str) -> Option<usize> {
            Some(self.loss)
        }
        fn prepare_document_hook(
            &self,
            text: &str,
            seal: &EventSearchSeal,
        ) -> Result<PreparedEventSearchDocument> {
            self.inner.prepare_document_hook(text, seal)
        }
        fn embed_prepared_document_hook(
            &self,
            inference: &crate::embedder::EventSearchInference,
        ) -> Result<EmbeddingOutput> {
            self.inner.embed_prepared_document_hook(inference)
        }
    }

    fn registration(embedder: Arc<dyn Embedder>) -> crate::EmbedderRegistration {
        crate::EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }
    }

    fn fixture_claim(bodies: &[&str]) -> Claim {
        let context_id = crate::ContextId::new_v7();
        let events = bodies
            .iter()
            .enumerate()
            .map(|(index, body)| ClaimedEvent {
                membership_id: uuid::Uuid::nil(),
                ordinal: index as u64,
                effective_interpretation_generation_id: None,
                captured_source_sha256: String::new(),
                effective_source_sha256: String::new(),
                event: ConversationEvent {
                    id: crate::EventId::new_v7(),
                    context_id,
                    connector_id: "proof".into(),
                    adapter_version: "1".into(),
                    anchor_declaration_version: "1".into(),
                    source_instance_ref: "proof".into(),
                    session_ref: "proof".into(),
                    source_event_key: index.to_string(),
                    source_position: JsonValue::Null,
                    seq: index as u64 + 1,
                    content_hash: String::new(),
                    anchor_count: 0,
                    anchor_set_hash: String::new(),
                    occurred_at: None,
                    author: "proof".into(),
                    kind: "text".into(),
                    render_role: EventRenderRole::HumanMessage,
                    interaction_ref: None,
                    text: Some((*body).into()),
                    payload: JsonValue::Null,
                    anchors: json!([]),
                    source_remainder: JsonValue::Null,
                    coverage: EventCoverage::Complete,
                },
            })
            .collect();
        Claim {
            publication_event_count: None,
            generation: Generation {
                id: EventSearchGenerationId::new_v7(),
                context_id,
                bindings: BTreeMap::new(),
                snapshot_sha256: String::new(),
                event_count: bodies.len() as u64,
                ordered_source_set_sha256: String::new(),
                source_content_sha256: String::new(),
                target: SourceSnapshot {
                    event_count: 0,
                    ordered_source_set_sha256: String::new(),
                    source_content_sha256: String::new(),
                },
                certified: None,
                started_at: TxId(1),
                attempt_tx: None,
                attempt_token: None,
                pending_dual_head_activation: false,
                mode: GenerationMode::Building,
            },
            owner_token: "proof".into(),
            lease_epoch: 1,
            expires_at: 10,
            connector_id: "proof".into(),
            source_instance_ref: "proof".into(),
            session_ref: "proof".into(),
            start_sequence: 1,
            end_sequence: bodies.len() as u64,
            cursor_sha256: String::new(),
            passage_ordinal: 0,
            prior_committed_sequence: 0,
            prior_open_tail_start_sequence: None,
            prior_open_tail_members: Vec::new(),
            prior_open_tail_passage_id: None,
            reopened_tail_passage_id: None,
            open_tail_events: Vec::new(),
            superseded_tail: Vec::new(),
            session_ended: true,
            events,
        }
    }

    fn small_profile() -> EventSearchProfile {
        EventSearchProfile {
            target_passage_tokens: std::num::NonZeroU32::new(64).unwrap(),
            hard_passage_tokens: std::num::NonZeroU32::new(96).unwrap(),
            oversized_event_overlap_tokens: std::num::NonZeroU32::new(4).unwrap(),
            ..EventSearchProfile::default()
        }
    }

    // Independent pre-repair oracle: consider all endpoints in reverse order without pruning,
    // then filter ALL original spans for overlap. No bound, index or optimized helper.
    fn exhaustive_ranges(
        embedder: &dyn Embedder,
        prefix: &str,
        body: &str,
        hard: usize,
        overlap: usize,
    ) -> Vec<(usize, usize, usize)> {
        let prepared = embedder.prepare_document(body).unwrap();
        let spans = prepared
            .byte_boundaries()
            .iter()
            .copied()
            .filter(|&(s, e)| {
                s < e && e <= body.len() && body.is_char_boundary(s) && body.is_char_boundary(e)
            })
            .collect::<Vec<_>>();
        let ends = spans
            .iter()
            .map(|&(_, e)| e)
            .chain([body.len()])
            .collect::<BTreeSet<_>>();
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < body.len() {
            let (end, count) = ends
                .iter()
                .rev()
                .copied()
                .filter(|&end| end > start)
                .find_map(|end| {
                    let count = embedder
                        .prepare_document(&format!("{prefix}{}", &body[start..end]))
                        .unwrap()
                        .token_count();
                    (count <= hard).then_some((end, count))
                })
                .expect("oracle has a fitting endpoint");
            ranges.push((start, end, count));
            if end == body.len() {
                break;
            }
            let contained = spans
                .iter()
                .filter(|&&(s, e)| s >= start && e <= end)
                .collect::<Vec<_>>();
            let next = contained[contained.len() - overlap].0;
            assert!(next > start);
            start = next;
        }
        ranges
    }

    fn split(
        claim: &Claim,
        registration: &crate::EmbedderRegistration,
        profile: &EventSearchProfile,
    ) -> Vec<PendingPassage> {
        let mut renew = |expiry| Ok(expiry);
        let mut lease = LeaseRenewal {
            expires_at: 10,
            renew: &mut renew,
        };
        match build_oversized_passages(
            claim,
            0,
            render_unit(&claim.events[0]).unwrap(),
            registration,
            profile,
            profile.hard_passage_tokens.get() as usize,
            None,
            &mut lease,
        )
        .unwrap()
        {
            OversizedPassages::Complete(passages) => passages,
            OversizedPassages::OverPassageBound => panic!("single oversized event must complete"),
        }
    }

    fn assert_exact(
        claim: &Claim,
        registration: &crate::EmbedderRegistration,
        profile: &EventSearchProfile,
    ) {
        let unit = render_unit(&claim.events[0]).unwrap();
        let expected = exhaustive_ranges(
            registration.embedder.as_ref(),
            &unit.prefix,
            &unit.body,
            profile.hard_passage_tokens.get() as usize,
            profile.oversized_event_overlap_tokens.get() as usize,
        );
        let actual = split(claim, registration, profile);
        assert_eq!(actual.len(), expected.len());
        for (ordinal, (actual, (start, end, count))) in actual.iter().zip(expected).enumerate() {
            let passage = &actual.passage;
            assert_eq!(passage.ordinal, ordinal as u64);
            assert_eq!(actual.prepared.token_count(), count);
            assert_eq!(
                passage.text,
                format!("{}{}", unit.prefix, &unit.body[start..end])
            );
            assert_eq!(passage.content_sha256, sha256(passage.text.as_bytes()));
            assert_eq!(passage.sources.len(), 1);
            let source = &passage.sources[0];
            assert_eq!(source.event_id, unit.event.event.id);
            assert_eq!(source.context_id, claim.generation.context_id);
            assert_eq!(
                (source.event_start_byte, source.event_end_byte),
                (Some(start as u64), Some(end as u64))
            );
            assert_eq!(
                (source.passage_start_byte, source.passage_end_byte),
                (unit.prefix.len() as u64, passage.text.len() as u64)
            );
            assert_eq!(
                &passage.text[source.passage_start_byte as usize..source.passage_end_byte as usize],
                &unit.body[start..end]
            );
        }
    }

    #[test]
    #[cfg(context_graph_internal_test_support)]
    fn bounded_preparation_exhaustive_oracle_preserves_farther_nonmonotone_fit() {
        for unordered in [false, true] {
            let (embedder, _work) = scalar_preparer(true, unordered, None);
            assert_eq!(embedder.event_search_slice_token_loss_bound("User: "), None);
            let body = format!("{}!{}", "é中α".repeat(35), "β".repeat(50));
            let claim = fixture_claim(&[&body]);
            let registration = registration(embedder);
            assert_exact(&claim, &registration, &small_profile());
            assert_eq!(
                split(&claim, &registration, &small_profile())[0]
                    .passage
                    .sources[0]
                    .event_end_byte,
                Some((body.find('!').unwrap() + 1) as u64)
            );
        }
    }

    #[test]
    #[cfg(context_graph_internal_test_support)]
    fn bounded_preparation_work_scales_with_source_size() {
        let mut observations = Vec::new();
        let mut span_work = Vec::new();
        for n in [128, 256, 512] {
            let (embedder, work_receiver) = scalar_preparer(false, false, None);
            let body = "α".repeat(n);
            let claim = fixture_claim(&[&body]);
            PREPARATION_SPAN_WORK.with(|work| work.set(0));
            let passages = split(
                &claim,
                &registration(Arc::new(BoundedScalarPreparer {
                    inner: embedder.clone(),
                    loss: 0,
                })),
                &small_profile(),
            );
            let work = preparation_work(&work_receiver);
            let visited = PREPARATION_SPAN_WORK.with(std::cell::Cell::get);
            span_work.push(visited);
            eprintln!(
                "preparation n={n} passages={} calls={} bytes={} span_work={visited}",
                passages.len(),
                work.calls,
                work.bytes
            );
            observations.push(work);
        }
        for pair in observations.windows(2) {
            assert!(
                pair[1].calls <= 3 * pair[0].calls,
                "calls: {observations:?}"
            );
            assert!(
                pair[1].bytes <= 3 * pair[0].bytes,
                "bytes: {observations:?}"
            );
        }
        for pair in span_work.windows(2) {
            assert!(pair[1] <= 3 * pair[0], "span work: {span_work:?}");
        }
    }

    #[test]
    #[cfg(context_graph_internal_test_support)]
    fn bounded_preparation_declared_bound_preserves_every_cut_and_unordered_fallback() {
        for unordered in [false, true] {
            for loss in [0, 6, usize::MAX] {
                let (inner, _work) = scalar_preparer(false, unordered, None);
                let registration = registration(Arc::new(BoundedScalarPreparer { inner, loss }));
                let body = format!("{}  \n", "e\u{301}é中α🦀".repeat(32));
                assert_exact(&fixture_claim(&[&body]), &registration, &small_profile());
            }
        }
    }

    #[test]
    #[cfg(context_graph_internal_test_support)]
    fn bounded_preparation_span_index_matches_direct_counting_and_overlap() {
        for original in [
            vec![(0, 1), (0, 1), (1, 4), (1, 4), (4, 5), (6, 8)],
            vec![(0, 4), (1, 3), (4, 5), (6, 8)], // only ends are unordered
            vec![(1, 3), (0, 4), (4, 5), (6, 8)], // only starts are unordered
        ] {
            let index = SourceSpanIndex::new(&original);
            assert_eq!(
                index.ordered,
                original
                    .windows(2)
                    .all(|w| w[0].0 <= w[1].0 && w[0].1 <= w[1].1)
            );
            for start in 0..=8 {
                let first = index.first_at_or_after(start);
                for end in start..=8 {
                    let contained = original
                        .iter()
                        .filter(|&&(s, e)| s >= start && e <= end)
                        .collect::<Vec<_>>();
                    for limit in 0..=8 {
                        if let Some(excluded) = index.excluded_end(first, limit) {
                            assert_eq!(end >= excluded, contained.len() > limit);
                        } else if index.ordered {
                            assert!(contained.len() <= limit);
                        }
                    }
                    for overlap in 1..=8 {
                        let expected = contained
                            .len()
                            .checked_sub(overlap)
                            .map(|at| contained[at].0);
                        assert_eq!(index.overlap_start(first, start, end, overlap), expected);
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(context_graph_internal_test_support)]
    fn bounded_preparation_single_event_completes_and_renews_inside_candidate_loop() {
        let body = "α".repeat(512);
        let (inner, work_receiver) = scalar_preparer(false, false, None);
        let registration = registration(Arc::new(BoundedScalarPreparer {
            inner: inner.clone(),
            loss: 0,
        }));
        let mut keeps = 0;
        let mut renew = |expiry| {
            keeps += 1;
            Ok(expiry + 1)
        };
        let result = render_and_embed_inner(
            fixture_claim(&[&body]),
            &registration,
            &small_profile(),
            bounds(8, 1, 8192, 1000),
            NonZeroUsize::MIN,
            &mut renew,
            &mut RunObservations::default(),
        )
        .unwrap();
        let Rendered::Batch(batch) = result else {
            panic!("single oversized event is unlimited");
        };
        assert!(batch.passages.len() > 1);
        assert_eq!(batch.claim.expires_at, 10 + keeps);
        let work = preparation_work(&work_receiver);
        assert!(
            keeps as usize >= work.calls,
            "renewal checks must reach candidate preparation: {keeps} for {work:?}"
        );
        assert_eq!(work.embeds, batch.passages.len());

        // An ownership refusal during preparation stops before any inference; it never invents
        // a replacement lease or swallows the caller's stale-claim result.
        let (inner, work_receiver) = scalar_preparer(false, false, None);
        let mut keeps = 0;
        let result = render_and_embed_inner(
            fixture_claim(&[&body]),
            &registration_for_bound(inner.clone()),
            &small_profile(),
            bounds(8, 1, 8192, 1000),
            NonZeroUsize::MIN,
            &mut |expiry| {
                keeps += 1;
                if keeps == 5 {
                    Err(stale_claim("proof takeover"))
                } else {
                    Ok(expiry + 1)
                }
            },
            &mut RunObservations::default(),
        );
        assert!(result.is_err_and(|error| is_stale_claim(&error)));
        assert_eq!(preparation_work(&work_receiver).embeds, 0);
    }

    fn registration_for_bound(inner: Arc<ScalarPreparer>) -> crate::EmbedderRegistration {
        registration(Arc::new(BoundedScalarPreparer { inner, loss: 0 }))
    }

    #[cfg(feature = "nomic-embedder")]
    fn offline_nomic() -> (
        Arc<crate::embedder::nomic::NomicTextEmbedder>,
        crate::EmbedderRegistration,
    ) {
        use crate::embedder::nomic::{
            EventHistoryTextEmbedder, NomicTextEmbedder, default_cache_root,
        };
        let model = Arc::new(NomicTextEmbedder::new(crate::EmbedderConfig {
            source: crate::ModelSource::LocalPath(
                default_cache_root().join("nomic-embed-text-v1.5"),
            ),
            offline: true,
            cache_dir: None,
            pinned_revision: Some(crate::embedder::NOMIC_DEFAULT_REVISION.into()),
            network_policy: "deny-download".into(),
        }));
        let registration =
            EventHistoryTextEmbedder::registration(model.clone()).expect("cached offline Nomic");
        (model, registration)
    }

    #[cfg(feature = "nomic-embedder")]
    #[test]
    #[cfg(context_graph_internal_test_support)]
    fn bounded_preparation_real_nomic_matches_exhaustive_unicode_cuts_and_profiles() {
        let (model, registration) = offline_nomic();
        assert_eq!(
            model.event_search_slice_token_loss_bound("User: "),
            Some(200)
        );
        assert_eq!(
            registration
                .embedder
                .event_search_slice_token_loss_bound("User: "),
            Some(200)
        );
        for prefix in ["", "User:", "User:\t", "User:\n", "User:\u{a0}"] {
            assert_eq!(model.event_search_slice_token_loss_bound(prefix), None);
        }
        let cases = [
            "café e\u{301} İSTANBUL naïve 中國語 한국어 🦀👩\u{200d}💻 ",
            "electroencephalographically counterrevolutionaries hyperparameterization ",
            "[UNK][CLS]word[MASK] [SEP] [PAD] a[MASK]b ",
            "  \t\r\n leading\u{a0}space\u{2003}trailing\u{200b} \0 a\u{0007}b ",
        ];
        let mut profiles = vec![
            small_profile(),
            EventSearchProfile::version_2(),
            EventSearchProfile::version_1(),
        ];
        // Also force starts inside continuation words under a narrow valid profile.
        let mut narrow = small_profile();
        narrow.hard_passage_tokens = std::num::NonZeroU32::new(32).unwrap();
        narrow.target_passage_tokens = std::num::NonZeroU32::new(24).unwrap();
        profiles.push(narrow);
        let mut cuts = 0;
        let mut continuation_starts = 0;
        for profile in &profiles {
            for fixture in cases {
                let repetitions = match profile.hard_passage_tokens.get() {
                    0..=32 => 4,
                    33..=96 => 10,
                    97..=1000 => 18,
                    _ => 2,
                };
                let body = format!(
                    "{} {} {}  \n",
                    fixture.repeat(repetitions),
                    "x".repeat(101),
                    fixture.repeat(repetitions)
                );
                let claim = fixture_claim(&[&body]);
                assert_exact(&claim, &registration, profile);
                let unit = render_unit(&claim.events[0]).unwrap();
                let prepared = model.prepare_document(&body).unwrap();
                let loss = registration
                    .embedder
                    .event_search_slice_token_loss_bound(&unit.prefix)
                    .unwrap();
                for passage in split(&claim, &registration, profile) {
                    cuts += 1;
                    let source = &passage.passage.sources[0];
                    let start = source.event_start_byte.unwrap() as usize;
                    let end = source.event_end_byte.unwrap() as usize;
                    let m = prepared
                        .byte_boundaries()
                        .iter()
                        .filter(|&&(s, e)| s < e && s >= start && e <= end)
                        .count();
                    assert!(passage.prepared.token_count() >= m.saturating_sub(loss));
                    if start > 0
                        && body[..start].chars().next_back().unwrap().is_alphanumeric()
                        && body[start..].chars().next().unwrap().is_alphanumeric()
                    {
                        continuation_starts += 1;
                    }
                }
            }
        }
        assert!(
            continuation_starts > 0,
            "fixtures must exercise real continuation starts"
        );
        assert!(cuts > 20);
        eprintln!(
            "real Nomic differential: {} fixtures/profiles, {cuts} exact cuts, {continuation_starts} continuation starts",
            profiles.len() * cases.len()
        );
        // The retained original profile must also exercise splitting, not only a fitting body.
        let body = "word ".repeat(1640);
        let claim = fixture_claim(&[&body]);
        assert_exact(&claim, &registration, &EventSearchProfile::version_1());
        assert!(split(&claim, &registration, &EventSearchProfile::version_1()).len() > 1);

        // Probe the declared inequality on arbitrary UTF-8 source cuts, including boundaries
        // which are not token endpoints, independently of whichever cuts the renderer selects.
        let body = cases.join("").repeat(12);
        let prepared = model.prepare_document(&body).unwrap();
        let boundaries = body
            .char_indices()
            .map(|(at, _)| at)
            .chain([body.len()])
            .collect::<Vec<_>>();
        let mut checked = 0;
        for start in boundaries.iter().step_by(173).copied() {
            for end in boundaries
                .iter()
                .step_by(227)
                .copied()
                .filter(|&end| end > start)
            {
                let m = prepared
                    .byte_boundaries()
                    .iter()
                    .filter(|&&(s, e)| s < e && s >= start && e <= end)
                    .count();
                let count = model
                    .prepare_document(&format!("User: {}", &body[start..end]))
                    .unwrap()
                    .token_count();
                assert!(
                    count >= m.saturating_sub(200),
                    "invalid bound at {start}..{end}: {count} vs {m}"
                );
                checked += 1;
            }
        }
        eprintln!(
            "real Nomic: original-profile oversized differential and {checked} arbitrary UTF-8 slice inequalities passed"
        );
    }

    #[cfg(feature = "nomic-embedder")]
    #[test]
    #[cfg(context_graph_internal_test_support)]
    #[ignore = "bounded offline preparation throughput receipt, run explicitly after correctness"]
    fn bounded_preparation_real_nomic_throughput() {
        use crate::embedder::nomic::take_preparation_work_for_test;
        let (_, registration) = offline_nomic();
        registration
            .embedder
            .prepare_document("warm the cached tokenizer and weights without inference")
            .unwrap();
        let mut observations = Vec::new();
        for n in [128, 256, 512] {
            let body = "Deployment rollback reviewed café e\u{301} 中國語 counterrevolutionaries [MASK] \n".repeat(n);
            let claim = fixture_claim(&[&body]);
            take_preparation_work_for_test();
            PREPARATION_SPAN_WORK.with(|work| work.set(0));
            let mut elapsed = Duration::ZERO;
            let passages = measured(&mut elapsed, || {
                split(&claim, &registration, &EventSearchProfile::default())
            });
            let (calls, bytes) = take_preparation_work_for_test();
            let spans = PREPARATION_SPAN_WORK.with(std::cell::Cell::get);
            eprintln!(
                "offline preparation: source_bytes={} passages={} calls={calls} prepared_bytes={bytes} span_work={spans} elapsed_ms={:.3} source_KiB_s={:.3}; forward_calls=0",
                body.len(),
                passages.len(),
                elapsed.as_secs_f64() * 1000.0,
                body.len() as f64 / 1024.0 / elapsed.as_secs_f64()
            );
            observations.push((calls, bytes, spans));
        }
        for pair in observations.windows(2) {
            assert!(
                pair[1].0 <= pair[0].0 * 3
                    && pair[1].1 <= pair[0].1 * 3
                    && pair[1].2 <= pair[0].2 * 3,
                "real preparation work: {observations:?}"
            );
        }
    }

    #[test]
    #[cfg(context_graph_internal_test_support)]
    fn bounded_preparation_overflow_stops_before_later_event_preparation() {
        let body = "α".repeat(200);
        let (embedder, work_receiver) = scalar_preparer(false, false, Some("DO_NOT_PREPARE"));
        let result = render_and_embed_inner(
            fixture_claim(&[&body, "DO_NOT_PREPARE"]),
            &registration(embedder.clone()),
            &small_profile(),
            bounds(8, 1, 8192, 1000),
            NonZeroUsize::MIN,
            &mut |expiry| Ok(expiry + 1),
            &mut RunObservations::default(),
        )
        .unwrap();
        assert!(matches!(
            result,
            Rendered::OverPassageBound { fitting_events: 1 }
        ));
        let work = preparation_work(&work_receiver);
        assert_eq!(work.embeds, 0);
        // Only the first two slices can be constructed when the allowance is one. Without
        // the inside-loop check this small fallback tokenizer exceeds 800 preparations.
        assert!(
            work.calls < 400,
            "oversized loop did not stop at overflow: {work:?}"
        );

        // Ordinary emissions must stop too, before overlap preparation or the next event.
        let (embedder, work_receiver) = scalar_preparer(false, false, Some("DO_NOT_PREPARE"));
        let result = render_and_embed_inner(
            fixture_claim(&[
                "αααααααααααααααααααα",
                "ββββββββββββββββββββ",
                "γγγγγγγγγγγγγγγγγγγγ",
                "DO_NOT_PREPARE",
            ]),
            &registration(embedder.clone()),
            &small_profile(),
            bounds(8, 1, 8192, 1000),
            NonZeroUsize::MIN,
            &mut |expiry| Ok(expiry),
            &mut RunObservations::default(),
        )
        .unwrap();
        assert!(matches!(
            result,
            Rendered::OverPassageBound { fitting_events: 2 }
        ));
        assert_eq!(preparation_work(&work_receiver).embeds, 0);
    }
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn assert_claim_batch_examines_only_pending_tail_for_test(
    fixture: impl FnMut() -> contextdb_engine::Database,
) {
    assert_session_read_work_for_test(fixture, false);
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn assert_selected_frontier_count_is_bounded_for_test(
    fixture: impl FnMut() -> contextdb_engine::Database,
) {
    // Engine gap: a declared MAINTAIN COUNT is currently DeclaredNotServed on
    // the live read path. Keep this oracle strict until that upstream route lands.
    assert_session_read_work_for_test(fixture, true);
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
fn assert_session_read_work_for_test(
    mut fixture: impl FnMut() -> contextdb_engine::Database,
    count_only: bool,
) {
    fn examined(
        db: contextdb_engine::Database,
        completed: i64,
        neighbours: i64,
        count_only: bool,
    ) -> (u64, bool) {
        // Preserve the projection columns and primary key; omit unrelated parent FKs
        // and natural-key uniqueness checks from setup. Exercise shipped access routes
        // and statements; public import/resume has its own identity/order proofs.
        db.execute("CREATE TABLE event_search_source_events (membership_id UUID NOT NULL, context_id UUID NOT NULL, generation_id UUID NOT NULL, ordinal INTEGER NOT NULL, event_id UUID NOT NULL, connector_id TEXT NOT NULL, source_instance_ref TEXT NOT NULL, session_ref TEXT NOT NULL, source_sequence INTEGER NOT NULL, effective_interpretation_generation_id UUID, captured_source_sha256 TEXT NOT NULL, effective_source_sha256 TEXT NOT NULL, PRIMARY KEY (context_id, generation_id, ordinal)) IMMUTABLE SYNC OFF", &HashMap::new()).expect("create projection table");
        for index in crate::schema::LOCAL_DERIVED_INDEXES
            .iter()
            .filter(|index| index.table == "event_search_source_events")
        {
            db.execute(index.ddl, &HashMap::new())
                .expect("install shipped projection index");
        }
        if count_only {
            db.execute("ALTER TABLE event_search_source_events SET MAINTAIN COUNT (context_id, generation_id, connector_id, source_instance_ref, session_ref)", &HashMap::new()).expect("declare exact stream count for upstream route proof");
        }
        let context = uuid::Uuid::new_v4();
        let generation = uuid::Uuid::new_v4();
        let mut rows = Vec::new();
        for (source, session, count, sequence_start) in [
            ("selected-source", "selected-session", completed, 0),
            (
                "selected-source",
                "a-neighbour-session",
                neighbours,
                completed,
            ),
            (
                "selected-source",
                "z-neighbour-session",
                neighbours,
                completed + neighbours,
            ),
            ("a-neighbour-source", "selected-session", neighbours, 0),
            ("z-neighbour-source", "selected-session", neighbours, 0),
            (
                "selected-source",
                "selected-session",
                5,
                completed + 2 * neighbours,
            ),
        ] {
            for offset in 1..=count {
                rows.push(HashMap::from([
                    ("membership_id".into(), Value::Uuid(uuid::Uuid::new_v4())),
                    ("context_id".into(), Value::Uuid(context)),
                    ("generation_id".into(), Value::Uuid(generation)),
                    ("ordinal".into(), Value::Int64(rows.len() as i64 + 1)),
                    ("event_id".into(), Value::Uuid(uuid::Uuid::new_v4())),
                    ("connector_id".into(), Value::Text("claim-fixture".into())),
                    ("source_instance_ref".into(), Value::Text(source.into())),
                    ("session_ref".into(), Value::Text(session.into())),
                    (
                        "source_sequence".into(),
                        Value::Int64(sequence_start + offset),
                    ),
                    ("effective_interpretation_generation_id".into(), Value::Null),
                    (
                        "captured_source_sha256".into(),
                        Value::Text("fixture".into()),
                    ),
                    (
                        "effective_source_sha256".into(),
                        Value::Text("fixture".into()),
                    ),
                ]));
            }
        }
        for batch in rows.chunks(64) {
            let tx = db.begin().expect("begin fixture batch");
            for row in batch {
                db.insert_row(tx, "event_search_source_events", row.clone())
                    .expect("insert projection membership");
            }
            db.commit(tx).expect("commit fixture batch");
        }
        let mut params = HashMap::from([
            ("context_id".into(), Value::Uuid(context)),
            ("generation_id".into(), Value::Uuid(generation)),
            ("connector_id".into(), Value::Text("claim-fixture".into())),
            ("source".into(), Value::Text("selected-source".into())),
            ("session".into(), Value::Text("selected-session".into())),
            ("committed".into(), Value::Int64(completed)),
        ]);
        let session = db
            .read_session(ReadLimits {
                result_rows: 3,
                ..ReadLimits::default()
            })
            .expect("bounded live projection session");
        if count_only {
            params.insert("connector".into(), Value::Text("claim-fixture".into()));
            params.insert(
                "frontier".into(),
                Value::Int64(completed + 2 * neighbours + 3),
            );
            // Measure the production composition itself: every statement it issues is charged.
            let mut examined = 0;
            let mut served = false;
            let count = selected_events_through(|sql| {
                let result = session.execute(sql, &params).unwrap();
                examined += result.trace.rows_examined;
                if sql == SELECTED_STREAM_COUNT_STATEMENT {
                    assert_eq!(result.rows, vec![vec![Value::Int64(completed + 5)]]);
                    served = matches!(
                        result.trace.maintained_count.as_deref(),
                        Some(
                            contextdb_engine::maintained_counts::MaintainedCountDisclosure::Used { .. }
                        )
                    );
                }
                get_u64(&result.rows[0], &result.columns, "event_count")
            })
            .unwrap();
            assert_eq!(count, (completed + 3) as u64);
            return (examined, served);
        }
        let discovery = session
            .execute(FIRST_PENDING_SOURCE_STATEMENT, &params)
            .expect("discover the selected pending tail");
        eprintln!(
            "pending_discovery: completed={completed}, neighbours={neighbours}, examined={}, index={:?}, sort_elided={}",
            discovery.trace.rows_examined, discovery.trace.index_used, discovery.trace.sort_elided
        );
        assert_eq!(
            discovery.trace.index_used.as_deref(),
            Some("idx_event_search_source_events_session_sequence"),
            "{:#?}",
            discovery.trace
        );
        assert!(discovery.trace.sort_elided, "{:#?}", discovery.trace);
        assert_eq!(discovery.trace.rows_examined, 1, "{:#?}", discovery.trace);
        assert_eq!(
            discovery.rows,
            vec![vec![Value::Int64(completed + 4 * neighbours + 1)]]
        );
        params.insert("start".into(), Value::Int64(completed + 2 * neighbours + 1));
        let source_position = session
            .execute(SOURCE_POSITION_SESSION_STATEMENT, &params)
            .unwrap();
        assert_eq!(
            source_position.rows,
            vec![vec![Value::Text("selected-session".into())]]
        );
        assert_eq!(
            source_position.trace.index_used.as_deref(),
            Some("idx_event_search_source_events_cursor_range")
        );
        // Relational, no literal: the equality probe examines exactly the rows it returns, so
        // any over-read fails; one row is returned at every history size, so it is also constant.
        assert_eq!(
            source_position.trace.rows_examined,
            source_position.rows.len() as u64,
            "source-position equality probe examines only the rows it returns"
        );
        eprintln!(
            "source_position: completed={completed}, neighbours={neighbours}, examined={}, index={:?}",
            source_position.trace.rows_examined, source_position.trace.index_used
        );
        params.insert("after".into(), Value::Int64(completed));
        let later = session
            .execute(HAS_LATER_SESSION_STATEMENT, &params)
            .unwrap();
        eprintln!(
            "has_later: completed={completed}, neighbours={neighbours}, examined={}, index={:?}, sort_elided={}",
            later.trace.rows_examined, later.trace.index_used, later.trace.sort_elided
        );
        assert_eq!(later.trace.rows_examined, 1);
        assert_eq!(
            later.trace.index_used.as_deref(),
            Some("idx_event_search_source_events_session_sequence")
        );
        assert!(later.trace.sort_elided);
        assert_eq!(
            later.rows,
            vec![vec![Value::Int64(completed + 2 * neighbours + 1)]]
        );
        params.insert("start_ordinal".into(), discovery.rows[0][0].clone());
        let sql = claim_source_statement(3);
        let claim = session
            .execute(&sql, &params)
            .expect("claim three pending entries");
        assert_eq!(claim.rows.len(), 3);
        for (offset, row) in claim.rows.iter().enumerate() {
            assert_eq!(
                row[1],
                Value::Int64(completed + 4 * neighbours + 1 + offset as i64)
            );
            assert_eq!(row[3], Value::Text("claim-fixture".into()));
            assert_eq!(row[4], Value::Text("selected-source".into()));
            assert_eq!(row[5], Value::Text("selected-session".into()));
            assert_eq!(
                row[6],
                Value::Int64(completed + 2 * neighbours + 1 + offset as i64)
            );
        }
        eprintln!(
            "claim_batch_examines_only_pending_tail: completed={completed}, neighbours={neighbours}, examined={}, index={:?}, sort_elided={}",
            claim.trace.rows_examined, claim.trace.index_used, claim.trace.sort_elided
        );
        assert_eq!(
            claim.trace.index_used.as_deref(),
            Some("idx_event_search_source_events_session_claim"),
            "{:#?}",
            claim.trace
        );
        assert!(claim.trace.sort_elided, "{:#?}", claim.trace);
        let explanation = contextdb_engine::cli_render::render_explain(&db, &sql, &params)
            .expect("explain the shipped claim statement");
        assert!(
            explanation.contains("idx_event_search_source_events_session_claim"),
            "{explanation}"
        );
        assert!(explanation.contains("sort_elided: true"), "{explanation}");
        eprintln!("claim EXPLAIN at completed={completed}:\n{explanation}");
        assert!(
            claim.trace.rows_examined >= 3,
            "account for returned entries"
        );
        (claim.trace.rows_examined, true)
    }
    let receipts: Vec<_> = [(10, 16), (10, 1_016), (2_010, 16), (2_010, 1_016)]
        .into_iter()
        .map(|(completed, neighbours)| examined(fixture(), completed, neighbours, count_only))
        .collect();
    if count_only {
        assert!(
            receipts.iter().all(|(_, served)| *served),
            "upstream LIVE maintained-count gap: DeclaredNotServed; receipts={receipts:?}"
        );
    }
    for receipt in &receipts[1..] {
        assert_eq!(
            receipt.0, receipts[0].0,
            "completed history and neighbouring streams must not increase examined work: {receipts:?}"
        );
    }
}
