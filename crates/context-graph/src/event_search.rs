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

impl Default for EventSearchProfile {
    fn default() -> Self {
        Self {
            target_passage_tokens: NonZeroU32::new(1_024).expect("1,024 is non-zero"),
            hard_passage_tokens: NonZeroU32::new(1_536).expect("1,536 is non-zero"),
            ordinary_overlap_events: NonZeroU32::new(1).expect("one event is non-zero"),
            oversized_event_overlap_tokens: NonZeroU32::new(128).expect("128 is non-zero"),
            preferred_boundary: EventSearchPreferredBoundary::BeforeHumanMessage,
        }
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
    pub max_events_per_batch: NonZeroU32,
    pub max_passages_per_batch: NonZeroU32,
    pub max_bytes_per_batch: NonZeroUsize,
    pub lease_duration_ms: NonZeroU64,
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
        crate::store::event_search_projection::catch_up_standalone(request)
    }

    /// Record that this Context's projection could not run, so status answers `failed` with the
    /// reason instead of `pending` for work no process is doing.
    ///
    /// `reason` names a condition in the caller's own words; it must carry no path, process or
    /// source text, because status shows it verbatim.
    pub fn record_refusal(request: EventSearchRunRequest, reason: &str) -> Result<()> {
        crate::store::event_search_projection::record_refusal_standalone(request, reason)
    }
}

impl Store {
    pub fn catch_up_event_search(
        &self,
        request: EventSearchBuildRequest,
    ) -> Result<EventSearchBuildReceipt> {
        crate::store::event_search_projection::catch_up_store(self, request)
    }

    pub fn rebuild_event_search(
        &self,
        request: EventSearchRebuildRequest,
    ) -> Result<EventSearchBuildReceipt> {
        crate::store::event_search_projection::rebuild_store(self, request)
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
