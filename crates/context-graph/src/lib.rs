// The README's example is compiled by the doctest harness. An example nothing builds drifts
// silently: the symbols it names are renamed or moved by ordinary work, and the first person to
// find out is a reader who copied it.

/// Keeps the README's example compiling under the doctest harness without splicing the README
/// text into the crate's expanded source.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctest;

pub mod capture_sync_status;
pub mod consumer_read;
pub mod embedder;
pub mod embedding_space;
pub mod embedding_target;
pub mod error;
mod event_search;
pub mod fabric;
pub mod grounding;
pub mod harness;
pub mod ingest;
pub mod injection;
pub mod interpretation;
pub mod llm;
pub mod messaging;
pub mod overlay;
pub mod owner_control;
pub mod producer;
pub mod review_surface;
pub mod schema;
pub mod skill_store;
pub mod status_reads;
pub mod store;
pub mod tier0;
pub mod types;
pub mod understanding;
pub mod write_budget;

pub use capture_sync_status::*;
pub use consumer_read::graph::ConsumerGraphView;
pub use consumer_read::{ConsumerDeclaration, ConsumerReader};
pub use contextdb_core::{ScopeLabel, TxId};
pub use embedder::{
    Embedder, EmbedderConfig, EmbedderRegistration, EmbeddingInput, EmbeddingOutput,
    EventSearchCapability, EventSearchEmbedder, EventSearchInference, EventSearchRole, ModelSource,
    PreparedEventSearchDocument, PreparedEventSearchQuery,
    load_observer::{EmbedderLoadEvent, EmbedderLoadObserver},
};
pub use error::{
    CgError, InterpretationActivationRefusalReason, InterpretationAdmissionRefusalReason,
    InterpretationLeaseRefusalReason, ReaderIdentity, SourceHistoryDivergence,
};
pub use event_search::*;
pub use injection::{
    INJECTION_TOKEN_BUDGET, InjectionIndex, InjectionLine, InjectionTier,
    VERBATIM_SIMILARITY_THRESHOLD, build_injection_index,
};
pub use interpretation::*;
pub use llm::*;
pub use messaging::{MessageBody, MismatchKind, OverrideFailureReason};
pub use owner_control::{ControlHandler, OwnerControlError};
pub use producer::CG_EDGE_SYSTEM_TAG;
pub use store::{
    ConsumerSchema, ConsumerTable, DigestGraphWalkBoundary, DigestSearchFilter, DigestSearchHit,
    EntityMatch, EntityReferenceMatchOptions, ReindexEvent, ReindexOptions, Store,
};
pub use store::{
    EventHistoryContextStatus, EventHistoryInterpretationHead, EventHistoryMemoryCounts,
    EventHistorySearchState, EventHistorySearchStatus, EventHistoryStatusSet,
    EventSourcePrefixCheck, EventSourceStreamStatus, ForgetReceipt, ForgetSelection,
    ForgetTableCount, RetiredInterpretation, event_history_reader_declaration,
};
pub use types::*;

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub mod test_support {
    /// Observes whether the store registry has a live `Database` owner for `path`.
    ///
    /// This only reports `Store`'s shared `Database` ownership. It cannot prove
    /// that arbitrary engine-internal buffers have been released.
    pub use crate::store::database_is_live_for_test as store_database_is_live_for_test;
    #[cfg(feature = "test-support")]
    pub use crate::store::event_search_projection::{
        observe_embedding_waves_for_test, take_claim_loads_for_test,
    };
    #[cfg(feature = "test-support")]
    pub use crate::store::open_retry::{
        observe_store_open_retries_for_test, set_store_open_retry_allowance_for_test,
    };

    /// Pauses inside the next event-search head-switch transaction on the calling thread.
    pub use crate::store::pause_next_event_search_finalization_for_test;

    pub use crate::store::event_search_projection::{
        after_next_event_search_membership_read_page_for_test,
        observe_event_search_batch_admission_for_test,
        set_event_search_projection_read_deadline_ms_for_test,
    };
    /// How many Stores the event-search runner has opened on the calling thread.
    pub use crate::store::event_search_store_opens_for_test;

    /// Every statement the event-history reads issue on this thread, so a proof can assert what a
    /// `find`, a `status` or a projection pass COSTS in statements -- never how long it took.
    pub use crate::store::event_search_read::{
        record_event_history_statements_for_test, take_recorded_event_history_statements_for_test,
    };

    /// Failure-ownership controls for the projection's internal regression suite.
    pub use crate::store::{
        after_next_event_search_attempt_commit_before_readback_for_test,
        after_next_event_search_completion_commit_before_readback_for_test,
        before_next_event_search_attempt_commit_for_test,
        before_next_event_search_completion_commit_for_test,
        before_next_event_search_failure_record_for_test,
        fail_next_event_search_membership_capture_for_test,
        fail_next_event_search_membership_capture_with_error_for_test,
        take_event_search_membership_captures_for_test,
    };

    /// Consumer-side row and cursor work performed by each pending-stream discovery pass.
    pub use crate::store::{
        EventSearchDiscoveryObservation, clear_event_search_discovery_observations_for_test,
        event_search_discovery_observations_for_test,
    };

    #[cfg(context_graph_internal_test_support)]
    pub mod event_correlation {
        pub use crate::store::{
            EventCorrelationObservationGuard, EventCorrelationTestObserver,
            observe_public_store_resolver_for_test,
        };
    }

    #[cfg(context_graph_internal_test_support)]
    pub mod embedding {
        pub use crate::store::embedding::{
            CompositeVectorClause, FusionPolicy, VectorSearchFilters, VectorSearchResult,
            compare_embedding_targets_across_spaces, composite_vector_search_with_policy,
            dedup_compare_digest_spaces, evidence_search, semantic_search, vector_search,
        };
        pub use crate::store::entity_reference_embeddings::entity_reference_cosine_match;
    }
}

#[cfg(trybuild)]
#[doc(hidden)]
pub mod compile_support {
    pub mod embedding {
        pub use crate::store::embedding::{
            CompositeVectorClause, FusionPolicy, VectorSearchFilters,
            composite_vector_search_with_policy,
        };
    }
}

pub type Result<T> = std::result::Result<T, CgError>;
pub type AgentId = String;
