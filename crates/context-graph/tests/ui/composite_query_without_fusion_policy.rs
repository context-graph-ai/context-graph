use context_graph::compile_support::embedding::{
    CompositeVectorClause, VectorSearchFilters, composite_vector_search_with_policy,
};
use context_graph::{Store, StoreConfig};

fn main() {
    let store = Store::open(StoreConfig::default()).unwrap();
    let clauses = vec![
            CompositeVectorClause {
                query_vector: vec![1.0],
                embedding_space_id: "text_default_embeddinggemma_300m_768".to_string(),
            },
            CompositeVectorClause {
                query_vector: vec![0.0],
                embedding_space_id: "image_clip_v1_4".to_string(),
            },
        ];
    let _ = composite_vector_search_with_policy(
        &store,
        clauses,
        VectorSearchFilters::default(),
    );
}
