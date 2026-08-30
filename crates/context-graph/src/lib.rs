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
pub mod fabric;
pub mod grounding;
pub mod harness;
pub mod ingest;
pub mod injection;
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

pub use capture_sync_status::*;
pub use consumer_read::graph::ConsumerGraphView;
pub use consumer_read::{ConsumerDeclaration, ConsumerReader};
pub use contextdb_core::{ScopeLabel, TxId};
pub use embedder::{
    Embedder, EmbedderConfig, EmbedderRegistration, EmbeddingInput, EmbeddingOutput, ModelSource,
    load_observer::{EmbedderLoadEvent, EmbedderLoadObserver},
};
pub use error::{CgError, ReaderIdentity};
pub use injection::{
    INJECTION_TOKEN_BUDGET, InjectionIndex, InjectionLine, InjectionTier,
    VERBATIM_SIMILARITY_THRESHOLD, build_injection_index,
};
pub use llm::*;
pub use messaging::{MessageBody, MismatchKind, OverrideFailureReason};
pub use owner_control::{ControlHandler, OwnerControlError};
pub use producer::CG_EDGE_SYSTEM_TAG;
pub use store::{
    ConsumerSchema, ConsumerTable, DigestGraphWalkBoundary, DigestSearchFilter, DigestSearchHit,
    EntityMatch, EntityReferenceMatchOptions, ReindexEvent, ReindexOptions, Store,
};
pub use types::*;

#[cfg(context_graph_internal_test_support)]
#[doc(hidden)]
pub mod test_support {
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
