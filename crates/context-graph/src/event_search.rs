//! Public event-history search projection and retrieval contracts: the build and rebuild requests
//! and receipts for the derived passage projection, and the bounded, Context-authorized search and
//! passage reads over its active head.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    ConnectorId, ConsumerReader, ContextId, ContextSelector, EmbedderRegistration, EmbeddingId,
    EventId, EventSearchGenerationId, EventSearchPassageId, EventWatermark,
    InterpretationGenerationId, Result, Store, StoreConfig, TxId,
};

pub const EVENT_SEARCH_RENDERER_ID: &str = "context_graph_event_history";
pub const EVENT_SEARCH_RENDERER_VERSION: &str = "1";
pub const EVENT_SEARCH_DEFAULT_LIMIT: u32 = 10;
pub const EVENT_SEARCH_MAX_LIMIT: u32 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSearchOrder {
    Newest,
    Oldest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSearchSessionScope {
    Recent(NonZeroUsize),
    All,
}

/// Work-selection policy for event-history embedding. It never participates in passage,
/// renderer, model, or generation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchWorkPolicy {
    pub order: EventSearchOrder,
    pub sessions: EventSearchSessionScope,
}

impl Default for EventSearchWorkPolicy {
    fn default() -> Self {
        Self {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::Recent(
                NonZeroUsize::new(10).expect("the default session bound is positive"),
            ),
        }
    }
}

/// Durable source frontiers selected by an accepted search request, in session order.
/// Selection is execution state and does not participate in vector identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchSelectedWork {
    /// Durable identity when loaded from an accepted request; absent for an unsaved selection.
    pub(crate) request_id: Option<uuid::Uuid>,
    pub streams: Vec<EventSearchSelectedStream>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchSelectedStream {
    pub connector_id: String,
    pub session_ref: String,
    pub source_instance_ref: String,
    pub frontier: u64,
}

/// The pending source stream before a projection batch is admitted.
pub struct EventSearchPendingBatch<'a> {
    pub connector_id: &'a str,
    pub session_ref: &'a str,
    pub source_instance_ref: &'a str,
    pub committed_sequence: u64,
    pub selected_frontier: &'a dyn Fn(uuid::Uuid) -> Result<Option<u64>>,
}

/// A batch retains its admitted ceiling and may stop at a request's captured frontier.
pub struct EventSearchBatchWriteLimit {
    pub memory_limit_bytes: NonZeroUsize,
    pub through_sequence: Option<u64>,
}

/// Resolve the library default for bounded passage embedding to available CPU parallelism.
///
/// Callers that need a different positive cap set it directly on an event-search request.
pub fn default_event_search_embedding_workers() -> std::io::Result<NonZeroUsize> {
    std::thread::available_parallelism()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSearchPreferredBoundary {
    BeforeHumanMessage,
}

/// Fully resolved passage construction settings stored on each generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchProfile {
    pub target_passage_tokens: NonZeroU32,
    pub hard_passage_tokens: NonZeroU32,
    pub ordinary_overlap_events: NonZeroU32,
    pub oversized_event_overlap_tokens: NonZeroU32,
    pub preferred_boundary: EventSearchPreferredBoundary,
}

impl EventSearchProfile {
    /// Reject an incoherent passage profile before a runner creates a generation or captures
    /// membership. A target equal to the hard ceiling is valid.
    pub fn validate(&self) -> Result<()> {
        if self.target_passage_tokens > self.hard_passage_tokens {
            return Err(crate::CgError::EventSearchProfileInvalid {
                target_passage_tokens: self.target_passage_tokens.get(),
                hard_passage_tokens: self.hard_passage_tokens.get(),
            });
        }
        if self.hard_passage_tokens <= self.oversized_event_overlap_tokens {
            return Err(crate::CgError::EventSearchOverlapInvalid {
                hard_passage_tokens: self.hard_passage_tokens.get(),
                oversized_event_overlap_tokens: self.oversized_event_overlap_tokens.get(),
            });
        }
        Ok(())
    }

    /// Reject a profile above an embedder's declared prepared-token ceiling without changing the
    /// profile or consulting model assets. Source-specific rendering fit remains a projection-time
    /// result because it depends on the rendered source and tokenizer.
    pub fn validate_for(&self, capability: crate::EventSearchCapability) -> Result<()> {
        self.validate()?;
        let requested_hard_passage_tokens = self.hard_passage_tokens.get();
        if u64::from(requested_hard_passage_tokens) > capability.token_limit() as u64 {
            return Err(crate::CgError::EventSearchProfileExceedsCapability {
                requested_hard_passage_tokens,
                maximum_prepared_tokens: capability.token_limit(),
            });
        }
        Ok(())
    }

    /// Version 1: 1,024-token target, 1,536-token hard stop. Still selectable, so a generation
    /// built under it can be matched and extended; new generations use [`Self::version_2`].
    pub fn version_1() -> Self {
        Self {
            target_passage_tokens: NonZeroU32::new(1_024).expect("1,024 is non-zero"),
            hard_passage_tokens: NonZeroU32::new(1_536).expect("1,536 is non-zero"),
            ordinary_overlap_events: NonZeroU32::new(1).expect("one event is non-zero"),
            oversized_event_overlap_tokens: NonZeroU32::new(128).expect("128 is non-zero"),
            preferred_boundary: EventSearchPreferredBoundary::BeforeHumanMessage,
        }
    }

    /// Version 2: 256-token target, 384-token hard stop, with version 1's user-boundary preference
    /// and one-event overlap. Smaller passages are sharper finding aids and far cheaper to embed,
    /// since encoder cost grows with the square of passage length.
    pub fn version_2() -> Self {
        Self {
            target_passage_tokens: NonZeroU32::new(256).expect("256 is non-zero"),
            hard_passage_tokens: NonZeroU32::new(384).expect("384 is non-zero"),
            ordinary_overlap_events: NonZeroU32::new(1).expect("one event is non-zero"),
            oversized_event_overlap_tokens: NonZeroU32::new(32).expect("32 is non-zero"),
            preferred_boundary: EventSearchPreferredBoundary::BeforeHumanMessage,
        }
    }
}

impl Default for EventSearchProfile {
    fn default() -> Self {
        Self::version_2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passage_profile_capability_ceiling_is_pure_and_preserves_admitted_values() {
        let capability =
            crate::embedder::nomic::NomicTextEmbedder::declared_event_search_capability();
        assert_eq!(capability.token_limit(), 8_192);

        for profile in [
            EventSearchProfile::version_1(),
            EventSearchProfile::version_2(),
        ] {
            assert_eq!(profile.validate_for(capability), Ok(()));
        }

        let accepted = EventSearchProfile {
            target_passage_tokens: NonZeroU32::new(32).expect("non-zero target"),
            hard_passage_tokens: NonZeroU32::new(40).expect("non-zero hard limit"),
            oversized_event_overlap_tokens: NonZeroU32::new(32).expect("non-zero overlap"),
            ..EventSearchProfile::default()
        };
        let accepted_before = accepted.clone();
        assert_eq!(accepted.validate_for(capability), Ok(()));
        assert_eq!(
            accepted, accepted_before,
            "admission never rewrites a profile"
        );

        let equal = EventSearchProfile {
            target_passage_tokens: NonZeroU32::new(8_192).expect("non-zero target"),
            hard_passage_tokens: NonZeroU32::new(8_192).expect("non-zero hard limit"),
            ..EventSearchProfile::default()
        };
        assert_eq!(equal.validate_for(capability), Ok(()));

        for requested_hard_passage_tokens in [8_193, 10_000] {
            let refused = EventSearchProfile {
                hard_passage_tokens: NonZeroU32::new(requested_hard_passage_tokens)
                    .expect("non-zero hard limit"),
                ..EventSearchProfile::default()
            };
            assert!(matches!(
                refused.validate_for(capability),
                Err(crate::CgError::EventSearchProfileExceedsCapability {
                    requested_hard_passage_tokens: requested,
                    maximum_prepared_tokens: 8_192,
                }) if requested == requested_hard_passage_tokens
            ));
        }

        let custom = crate::EventSearchCapability::new(40).expect("non-zero custom ceiling");
        assert_eq!(accepted.validate_for(custom), Ok(()));
        assert!(matches!(
            equal.validate_for(custom),
            Err(crate::CgError::EventSearchProfileExceedsCapability {
                requested_hard_passage_tokens: 8_192,
                maximum_prepared_tokens: 40,
            })
        ));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSearchGenerationState {
    Building,
    Ready,
    Failed,
    Retired,
}

#[derive(Debug, Clone)]
pub struct EventSearchBuildRequest {
    pub context_id: ContextId,
    pub embedder: EmbedderRegistration,
    pub profile: EventSearchProfile,
    /// Maximum passage embedding calls this held-Store build may run concurrently. Use
    /// [`default_event_search_embedding_workers`] for the available-CPU default.
    pub embedding_workers: NonZeroUsize,
    pub max_events_per_batch: NonZeroU32,
    pub max_passages_per_batch: NonZeroU32,
    pub max_bytes_per_batch: NonZeroUsize,
    pub lease_duration_ms: NonZeroU64,
}

#[derive(Debug, Clone)]
pub struct EventSearchRebuildRequest {
    pub context_id: ContextId,
    /// An explicit staged interpretation target. Its connector lookup is pending implementation.
    pub interpretation_generation_id: Option<InterpretationGenerationId>,
    pub embedder: EmbedderRegistration,
    pub profile: EventSearchProfile,
    /// Maximum passage embedding calls this held-Store rebuild may run concurrently. Use
    /// [`default_event_search_embedding_workers`] for the available-CPU default.
    pub embedding_workers: NonZeroUsize,
    pub max_events_per_batch: NonZeroU32,
    pub max_passages_per_batch: NonZeroU32,
    pub max_bytes_per_batch: NonZeroUsize,
    pub lease_duration_ms: NonZeroU64,
}

#[derive(Debug, Clone)]
pub struct EventSearchRunRequest {
    pub store: StoreConfig,
    pub contexts: ContextSelector,
    pub embedder: EmbedderRegistration,
    pub profile: EventSearchProfile,
    /// Maximum passage embedding calls this run may hold concurrently.
    ///
    /// This execution control is deliberately separate from projection identity.
    pub embedding_workers: NonZeroUsize,
    pub max_events_per_batch: NonZeroU32,
    pub max_passages_per_batch: NonZeroU32,
    pub max_bytes_per_batch: NonZeroUsize,
    pub lease_duration_ms: NonZeroU64,
}

/// The requested projection identity supplied by a caller. Renderer and interpretation identity
/// are resolved by the library, without loading an embedder or its artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchWorkSpec {
    pub profile: EventSearchProfile,
    pub embedding_space_id: String,
    pub encoding_profile_id: String,
}

/// Committed coverage for the same generation a runner would use for this request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchWorkAssessment {
    pub context_id: ContextId,
    pub captured_events: u64,
    pub generation_id: Option<EventSearchGenerationId>,
    pub covered_events: u64,
    pub pending_events: u64,
    pub outside_scope_events: u64,
    /// Outside-scope events that remain searchable from prior accepted work.
    #[serde(default)]
    pub outside_scope_indexed_events: u64,
    pub no_searchable_text_events: u64,
    pub work_required: bool,
    pub live_projection_lease: bool,
}

/// Observations survive a failed run, including an inference API call that returned an error.
#[derive(Debug)]
pub struct EventSearchRunOutcome {
    pub result: Result<EventSearchBuildReceipt>,
    /// Whether this run durably recorded its failure against the attempt it actually owned.
    ///
    /// Callers use this instead of reconstructing generation state after an error. In particular,
    /// a detached worker may acknowledge a request after [`EventSearchFailureDisposition::Recorded`],
    /// leaves it pending after [`EventSearchFailureDisposition::Unrecorded`], and treats
    /// [`EventSearchFailureDisposition::Stale`] as another attempt taking ownership.
    pub failure_disposition: EventSearchFailureDisposition,
    pub embedding_invocations: u64,
    pub timings: EventSearchBuildTimings,
}

/// The durable disposition of one observed projection run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventSearchFailureDisposition {
    /// The run succeeded, so there was no failure to record.
    NoFailure,
    /// The matching current attempt was updated with the core's sanitized failure reason.
    Recorded,
    /// The run failed before it owned an attempt, or the matching failure write was refused.
    Unrecorded,
    /// Another attempt owns the generation metadata; this run must not mark it failed.
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EventSearchBuildTimings {
    pub event_read_ms: u64,
    pub render_and_tokenize_ms: u64,
    pub embed_ms: u64,
    pub vector_insert_ms: u64,
    pub commit_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchBuildReceipt {
    pub context_id: ContextId,
    pub generation_id: EventSearchGenerationId,
    pub interpretation_snapshot_sha256: String,
    pub interpretation_generation_ids: Vec<InterpretationGenerationId>,
    pub source_watermark: EventWatermark,
    pub source_content_sha256: String,
    pub generation_content_sha256: String,
    pub renderer_id: String,
    pub renderer_version: String,
    pub profile: EventSearchProfile,
    pub embedding_space_id: String,
    pub encoding_profile_id: String,
    pub eligible_event_count: u64,
    pub rendered_event_count: u64,
    pub embedded_event_count: u64,
    pub excluded_event_count: u64,
    pub failed_event_count: u64,
    pub passage_count: u64,
    pub retry_count: u64,
    pub failure_count: u64,
    pub state: EventSearchGenerationState,
    pub active_generation_changed: bool,
    pub started_at: TxId,
    pub completed_at: Option<TxId>,
    pub timings: EventSearchBuildTimings,
    pub sanitized_failure: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchFilter {
    pub connector_id: Option<ConnectorId>,
    pub session_ref: Option<String>,
    pub source_instance_ref: Option<String>,
    pub limit: NonZeroU32,
}

impl Default for EventSearchFilter {
    fn default() -> Self {
        Self {
            connector_id: None,
            session_ref: None,
            source_instance_ref: None,
            limit: NonZeroU32::new(EVENT_SEARCH_DEFAULT_LIMIT)
                .expect("the default event-search limit is non-zero"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSearchContextState {
    Empty,
    NotReady,
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchContextCoverage {
    pub context_id: ContextId,
    pub state: EventSearchContextState,
    pub generation_id: Option<EventSearchGenerationId>,
    pub captured_event_count: u64,
    pub searchable_event_count: u64,
    pub pending_event_count: u64,
    /// Captured events outside the latest accepted session-selection policy.
    #[serde(default)]
    pub outside_scope_event_count: u64,
    /// Outside-scope events that remain searchable from prior accepted work.
    #[serde(default)]
    pub outside_scope_indexed_event_count: u64,
    /// Selected events in completed sessions that produced no searchable passage membership.
    #[serde(default)]
    pub no_searchable_text_event_count: u64,
    pub source_watermark: Option<EventWatermark>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassageEventSource {
    pub context_id: ContextId,
    pub passage_id: EventSearchPassageId,
    pub event_id: EventId,
    pub connector_id: ConnectorId,
    pub session_ref: String,
    pub source_instance_ref: String,
    pub source_order: u64,
    pub event_start_byte: Option<u64>,
    pub event_end_byte: Option<u64>,
    pub passage_start_byte: u64,
    pub passage_end_byte: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSearchPassage {
    pub id: EventSearchPassageId,
    pub context_id: ContextId,
    pub generation_id: EventSearchGenerationId,
    pub connector_id: ConnectorId,
    pub session_ref: String,
    pub source_instance_ref: String,
    pub ordinal: u64,
    pub first_event_seq: u64,
    pub last_event_seq: u64,
    pub first_occurred_at: Option<DateTime<Utc>>,
    pub last_occurred_at: Option<DateTime<Utc>>,
    pub text: String,
    pub content_sha256: String,
    pub embedding_id: EmbeddingId,
    pub embedding_space_id: String,
    pub dimension: u32,
    pub is_current: bool,
    pub created_at: TxId,
    pub sources: Vec<PassageEventSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventSearchHit {
    pub passage_id: EventSearchPassageId,
    pub context_id: ContextId,
    pub connector_id: ConnectorId,
    pub session_ref: String,
    pub source_instance_ref: String,
    pub first_event_seq: u64,
    pub last_event_seq: u64,
    pub first_occurred_at: Option<DateTime<Utc>>,
    pub last_occurred_at: Option<DateTime<Utc>>,
    pub text: String,
    pub cosine_similarity: f32,
    pub applied_filter: EventSearchFilter,
    pub embedding_space_id: String,
    pub generation_id: EventSearchGenerationId,
    pub sources: Vec<PassageEventSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventSearchResult {
    pub selected_context_ids: Vec<ContextId>,
    pub searched_context_ids: Vec<ContextId>,
    pub coverage: Vec<EventSearchContextCoverage>,
    pub hits: Vec<EventSearchHit>,
}

#[derive(Debug, Default)]
pub struct EventSearchProjectionRunner;

impl EventSearchProjectionRunner {
    pub fn catch_up(request: EventSearchRunRequest) -> Result<EventSearchBuildReceipt> {
        Self::catch_up_observed(request).result
    }

    pub fn catch_up_observed(request: EventSearchRunRequest) -> EventSearchRunOutcome {
        crate::store::event_search_projection::catch_up_standalone(request)
    }

    pub fn catch_up_observed_with_policy(
        request: EventSearchRunRequest,
        policy: EventSearchWorkPolicy,
    ) -> EventSearchRunOutcome {
        crate::store::event_search_projection::catch_up_standalone_with_policy(request, policy)
    }

    /// Resolve the execution-only worker cap at each projection claim.
    pub fn catch_up_observed_with_embedding_workers(
        request: EventSearchRunRequest,
        resolver: impl FnMut() -> NonZeroUsize,
    ) -> EventSearchRunOutcome {
        crate::store::event_search_projection::catch_up_standalone_with_embedding_workers(
            request, resolver,
        )
    }

    pub fn catch_up_observed_with_policy_and_embedding_workers(
        request: EventSearchRunRequest,
        policy: EventSearchWorkPolicy,
        resolver: impl FnMut() -> NonZeroUsize,
    ) -> EventSearchRunOutcome {
        crate::store::event_search_projection::catch_up_standalone_with_policy_and_embedding_workers(
            request, policy, resolver,
        )
    }

    /// Resolve the writer ceiling before each batch admission and concurrency before embedding.
    /// The resolver receives a predicate for whether a pending policy selects this batch.
    /// Shared generation metadata accepts every policy. A changed ceiling releases the previous
    /// writable state before reopening. All writes of
    /// an admitted batch, including lease renewal and commit, retain that batch's ceiling.
    pub fn catch_up_observed_with_policy_and_batch_limits(
        request: EventSearchRunRequest,
        policy: EventSearchWorkPolicy,
        embedding_workers: impl FnMut() -> NonZeroUsize,
        write_memory_limit: impl FnMut(
            &dyn Fn(EventSearchWorkPolicy) -> Result<bool>,
        ) -> Result<NonZeroUsize>,
    ) -> EventSearchRunOutcome {
        crate::store::event_search_projection::catch_up_standalone_with_batch_limits(
            request,
            policy,
            embedding_workers,
            write_memory_limit,
        )
    }

    /// Run selected session identities with request-specific source frontier admission.
    /// `None` denotes shared generation metadata; a source batch names its durable identity.
    pub fn catch_up_observed_with_selected_work_and_batch_limits(
        request: EventSearchRunRequest,
        policy: EventSearchWorkPolicy,
        selected_work: EventSearchSelectedWork,
        embedding_workers: impl FnMut() -> NonZeroUsize,
        write_memory_limit: impl FnMut(
            Option<EventSearchPendingBatch<'_>>,
        ) -> Result<EventSearchBatchWriteLimit>,
    ) -> EventSearchRunOutcome {
        crate::store::event_search_projection::catch_up_standalone_with_selected_work(
            request,
            policy,
            selected_work,
            embedding_workers,
            write_memory_limit,
        )
    }

    /// Record a caller-side refusal against an existing matching generation, so status answers
    /// `failed` with the reason instead of `pending` for work no process is doing.
    ///
    /// `reason` names a condition in the caller's own words; it must carry no path, process or
    /// source text, because status shows it verbatim. This starts and stamps one metadata attempt
    /// atomically. It neither creates a generation nor rebuilds its source membership.
    pub fn record_refusal(request: EventSearchRunRequest, reason: &str) -> Result<()> {
        crate::store::event_search_projection::record_refusal_standalone(request, reason)
    }
}

impl Store {
    /// Assess requested search work using only committed metadata, without changing the store.
    pub fn event_search_work(
        &self,
        selector: ContextSelector,
        spec: &EventSearchWorkSpec,
    ) -> Result<Vec<EventSearchWorkAssessment>> {
        self.event_search_work_with_policy(selector, spec, EventSearchWorkPolicy::default())
    }

    /// Read one durable request's selected source frontiers.
    pub fn event_search_selected_work(
        &self,
        context_id: ContextId,
        request_id: uuid::Uuid,
    ) -> Result<EventSearchSelectedWork> {
        crate::store::event_search_projection::load_selected_work(self, context_id, request_id)
    }

    /// Read the original source frontiers retained when a refused selection was retired.
    pub fn event_search_refusal_work(
        &self,
        context_id: ContextId,
        request_id: uuid::Uuid,
    ) -> Result<EventSearchSelectedWork> {
        crate::store::event_search_projection::load_refusal_work(self, context_id, request_id)
    }

    /// Preserve a refused selection's completion evidence before retiring its runnable rows.
    pub fn preserve_event_search_refusal_work(
        &self,
        context_id: ContextId,
        request_id: uuid::Uuid,
    ) -> Result<()> {
        crate::store::event_search_projection::preserve_refusal_work(self, context_id, request_id)
    }

    /// Reclaim completion evidence absent from the bounded worker queue after writable admission.
    pub fn reconcile_event_search_refusal_work(&self, retained: &[uuid::Uuid]) -> Result<()> {
        crate::store::event_search_projection::reconcile_refusal_work(self, retained)
    }

    /// Freeze the session identities and source frontiers selected by this policy.
    pub fn select_event_search_work(
        &self,
        context_id: ContextId,
        policy: EventSearchWorkPolicy,
    ) -> Result<EventSearchSelectedWork> {
        crate::store::event_search_projection::select_work(self, context_id, policy)
    }

    /// Whether the active compatible projection has committed every accepted source frontier.
    pub fn event_search_selected_work_complete(
        &self,
        context_id: ContextId,
        spec: &EventSearchWorkSpec,
        selected: &EventSearchSelectedWork,
    ) -> Result<bool> {
        crate::store::event_search_projection::selected_work_complete(
            self, context_id, spec, selected,
        )
    }

    pub fn event_search_work_with_policy(
        &self,
        selector: ContextSelector,
        spec: &EventSearchWorkSpec,
        policy: EventSearchWorkPolicy,
    ) -> Result<Vec<EventSearchWorkAssessment>> {
        crate::store::event_search_projection::event_search_work(self, selector, spec, policy)
    }

    /// Persist an accepted request's selected work while its capture Store is still held.
    pub fn persist_event_search_selected_work(
        &self,
        context_id: ContextId,
        selected: &EventSearchSelectedWork,
    ) -> Result<uuid::Uuid> {
        crate::store::event_search_projection::persist_selected_work(self, context_id, selected)
    }

    /// Retain saved queued frontiers that a differing in-flight profile may still publish.
    /// Identities must come from the bounded queue read after acquiring this writable Store.
    pub fn persist_event_search_selected_work_with_retained_requests(
        &self,
        context_id: ContextId,
        selected: &EventSearchSelectedWork,
        retained_requests: &[uuid::Uuid],
        desired_spec: &EventSearchWorkSpec,
    ) -> Result<uuid::Uuid> {
        crate::store::event_search_projection::persist_selected_work_with_retained_requests(
            self,
            context_id,
            selected,
            retained_requests,
            desired_spec,
        )
    }

    /// Reclaim selections absent from a bounded published queue. The caller must read that
    /// queue after acquiring this writable Store, which serializes selection acceptance.
    pub fn reconcile_event_search_selected_work(&self, retained: &[uuid::Uuid]) -> Result<()> {
        crate::store::event_search_projection::reconcile_selected_work(self, retained)
    }

    /// Remove acknowledged selections. Only the supplied request identities are removed.
    pub fn release_event_search_selected_work(
        &self,
        context_id: ContextId,
        requests: &[uuid::Uuid],
    ) -> Result<()> {
        crate::store::event_search_projection::release_selected_work(self, context_id, requests)
    }

    /// Persist the resolved search selection for a Context even when this invocation only
    /// captures history and does not start a projection.
    pub fn persist_event_search_work_policy(
        &self,
        context_id: ContextId,
        policy: EventSearchWorkPolicy,
    ) -> Result<()> {
        crate::store::event_search_projection::persist_work_policy(self, context_id, policy)
    }

    pub fn catch_up_event_search(
        &self,
        request: EventSearchBuildRequest,
    ) -> Result<EventSearchBuildReceipt> {
        crate::store::event_search_projection::catch_up_store(self, request)
    }

    /// Build missing event-search work under an explicit session selection policy.
    ///
    /// [`Self::catch_up_event_search`] remains the compatibility entrypoint and uses
    /// [`EventSearchWorkPolicy::default`].
    pub fn catch_up_event_search_with_policy(
        &self,
        request: EventSearchBuildRequest,
        policy: EventSearchWorkPolicy,
    ) -> Result<EventSearchBuildReceipt> {
        crate::store::event_search_projection::catch_up_store_with_policy(self, request, policy)
    }

    pub fn rebuild_event_search(
        &self,
        request: EventSearchRebuildRequest,
    ) -> Result<EventSearchBuildReceipt> {
        crate::store::event_search_projection::rebuild_store(self, request)
    }

    /// Rebuild event search under an explicit session selection policy.
    ///
    /// [`Self::rebuild_event_search`] remains the compatibility entrypoint and uses
    /// [`EventSearchWorkPolicy::default`].
    pub fn rebuild_event_search_with_policy(
        &self,
        request: EventSearchRebuildRequest,
        policy: EventSearchWorkPolicy,
    ) -> Result<EventSearchBuildReceipt> {
        crate::store::event_search_projection::rebuild_store_with_policy(self, request, policy)
    }

    pub fn find_event_history(
        &self,
        selector: ContextSelector,
        query: &str,
        filter: EventSearchFilter,
    ) -> Result<EventSearchResult> {
        crate::store::event_search_read::find_event_history(self, selector, query, filter)
    }

    pub fn get_event_search_passage(
        &self,
        selector: ContextSelector,
        passage_id: EventSearchPassageId,
    ) -> Result<Option<EventSearchPassage>> {
        crate::store::event_search_read::get_event_search_passage(self, selector, passage_id)
    }

    pub fn list_events_for_passage(
        &self,
        selector: ContextSelector,
        passage_id: EventSearchPassageId,
    ) -> Result<Vec<PassageEventSource>> {
        crate::store::event_search_read::list_events_for_passage(self, selector, passage_id)
    }
}

impl ConsumerReader {
    /// Read-only counterpart of [`Store::event_search_work`]. Declare the metadata tables in
    /// [`crate::event_history_reader_declaration`]; no embedder registration is required.
    pub fn event_search_work(
        &self,
        selector: ContextSelector,
        spec: &EventSearchWorkSpec,
    ) -> Result<Vec<EventSearchWorkAssessment>> {
        self.event_search_work_with_policy(selector, spec, EventSearchWorkPolicy::default())
    }

    /// Read one durable request's selected source frontiers.
    pub fn event_search_selected_work(
        &self,
        context_id: ContextId,
        request_id: uuid::Uuid,
    ) -> Result<EventSearchSelectedWork> {
        crate::store::event_search_projection::load_selected_work(self, context_id, request_id)
    }

    /// Freeze the session identities and source frontiers selected by this policy.
    pub fn select_event_search_work(
        &self,
        context_id: ContextId,
        policy: EventSearchWorkPolicy,
    ) -> Result<EventSearchSelectedWork> {
        crate::store::event_search_projection::select_work(self, context_id, policy)
    }

    /// Whether the active compatible projection has committed every accepted source frontier.
    pub fn event_search_selected_work_complete(
        &self,
        context_id: ContextId,
        spec: &EventSearchWorkSpec,
        selected: &EventSearchSelectedWork,
    ) -> Result<bool> {
        crate::store::event_search_projection::selected_work_complete(
            self, context_id, spec, selected,
        )
    }

    pub fn event_search_work_with_policy(
        &self,
        selector: ContextSelector,
        spec: &EventSearchWorkSpec,
        policy: EventSearchWorkPolicy,
    ) -> Result<Vec<EventSearchWorkAssessment>> {
        crate::store::event_search_projection::event_search_work(self, selector, spec, policy)
    }

    /// Future read-only find callers open the reader first, then attach the exact event-history
    /// registration without writing the Store.
    pub fn attach_event_search_embedder_registrations(
        &mut self,
        registrations: Vec<EmbedderRegistration>,
    ) -> Result<()> {
        self.attach_event_search_registrations(registrations)
    }

    pub fn find_event_history(
        &self,
        selector: ContextSelector,
        query: &str,
        filter: EventSearchFilter,
    ) -> Result<EventSearchResult> {
        crate::store::event_search_read::find_event_history(self, selector, query, filter)
    }

    pub fn get_event_search_passage(
        &self,
        selector: ContextSelector,
        passage_id: EventSearchPassageId,
    ) -> Result<Option<EventSearchPassage>> {
        crate::store::event_search_read::get_event_search_passage(self, selector, passage_id)
    }

    pub fn list_events_for_passage(
        &self,
        selector: ContextSelector,
        passage_id: EventSearchPassageId,
    ) -> Result<Vec<PassageEventSource>> {
        crate::store::event_search_read::list_events_for_passage(self, selector, passage_id)
    }
}
