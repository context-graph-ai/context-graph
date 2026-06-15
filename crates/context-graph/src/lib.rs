pub mod embedder;
pub mod embedding_space;
pub mod embedding_target;
pub mod error;
pub mod ingest;
pub mod llm;
pub mod messaging;
pub mod producer;
pub mod schema;
pub mod store;
pub mod types;

pub use contextdb_core::TxId;
pub use embedder::{
    Embedder, EmbedderConfig, EmbedderRegistration, EmbeddingInput, EmbeddingOutput, ModelSource,
    load_observer::{EmbedderLoadEvent, EmbedderLoadObserver},
};
pub use error::CgError;
pub use llm::*;
pub use messaging::{MessageBody, MismatchKind, OverrideFailureReason};
pub use producer::CG_EDGE_SYSTEM_TAG;
pub use store::{
    DigestGraphWalkBoundary, DigestSearchFilter, DigestSearchHit, ReindexEvent, ReindexOptions,
    Store,
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
