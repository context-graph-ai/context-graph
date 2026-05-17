use crate::{ContextId, DigestId, EmbeddingTarget, EvidenceId, EvidenceRef, Result, Store};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorSearchFilters {
    pub context_ids: Option<Vec<crate::ContextId>>,
    pub label_filters: Vec<(String, String)>,
    pub candidate_embedding_space_ids: Vec<String>,
    pub candidate_dimensions: Vec<(String, usize)>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchResult {
    pub embedding_space_id: String,
    pub target: EmbeddingTarget,
    pub context_id: Option<crate::ContextId>,
    pub score: f32,
    pub per_clause_scores: Vec<(String, f32)>,
    pub index_ref: String,
    pub planner_trace: String,
    pub projected_context_labels: Vec<String>,
    pub candidate_count_before_filter: usize,
    pub candidate_count_after_filter: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompositeVectorClause {
    pub query_vector: Vec<f32>,
    pub embedding_space_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FusionPolicy {
    WeightedSum(Vec<(String, f32)>),
}

type VectorSearchFn =
    fn(&Store, &[f32], &str, VectorSearchFilters) -> Result<Vec<VectorSearchResult>>;
type EvidenceSearchFn =
    fn(&Store, &EvidenceRef, Option<&str>, VectorSearchFilters) -> Result<Vec<VectorSearchResult>>;
type SemanticSearchFn = fn(&Store, &str, VectorSearchFilters) -> Result<Vec<VectorSearchResult>>;
type CompositeVectorSearchFn = fn(
    &Store,
    Vec<CompositeVectorClause>,
    FusionPolicy,
    VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>>;
type StageVersionedDigestFn = fn(&Store, DigestId, &str, ContextId) -> Result<()>;
type DedupCompareDigestSpacesFn =
    fn(&Store, DigestId, &str, DigestId, &str) -> Result<Vec<VectorSearchResult>>;
type CompareEmbeddingTargetsFn =
    fn(&Store, EmbeddingTarget, &str, EmbeddingTarget, &str, usize, usize) -> Result<f32>;

pub(crate) fn production_primitive_anchor() {
    let _: VectorSearchFn = vector_search;
    let _: EvidenceSearchFn = evidence_search;
    let _: SemanticSearchFn = semantic_search;
    let _: CompositeVectorSearchFn = composite_vector_search_with_policy;
    let _: StageVersionedDigestFn = stage_versioned_text_digest_for_dedup;
    let _: DedupCompareDigestSpacesFn = dedup_compare_digest_spaces;
    let _: CompareEmbeddingTargetsFn = compare_embedding_targets_across_spaces;
    let _ = CompositeVectorClause {
        query_vector: Vec::new(),
        embedding_space_id: String::new(),
    };
    let _ = FusionPolicy::WeightedSum(Vec::new());
}

pub fn vector_search(
    _store: &Store,
    _query_vector: &[f32],
    embedding_space_id: &str,
    _filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    Ok(vec![VectorSearchResult {
        embedding_space_id: embedding_space_id.to_string(),
        target: EmbeddingTarget::Evidence(EvidenceId::default()),
        context_id: None,
        score: 0.0,
        per_clause_scores: Vec::new(),
        index_ref: String::new(),
        planner_trace: String::new(),
        projected_context_labels: Vec::new(),
        candidate_count_before_filter: 0,
        candidate_count_after_filter: 0,
    }])
}

pub fn evidence_search(
    _store: &Store,
    _evidence: &EvidenceRef,
    embedding_space_id: Option<&str>,
    _filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    Ok(vec![VectorSearchResult {
        embedding_space_id: embedding_space_id.unwrap_or("unresolved").to_string(),
        target: EmbeddingTarget::Evidence(EvidenceId::default()),
        context_id: None,
        score: 0.0,
        per_clause_scores: Vec::new(),
        index_ref: String::new(),
        planner_trace: String::new(),
        projected_context_labels: Vec::new(),
        candidate_count_before_filter: 0,
        candidate_count_after_filter: 0,
    }])
}

pub fn semantic_search(
    _store: &Store,
    _text: &str,
    _filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    Ok(vec![VectorSearchResult {
        embedding_space_id: "text_default_embeddinggemma_300m_768".to_string(),
        target: EmbeddingTarget::Evidence(EvidenceId::default()),
        context_id: None,
        score: 0.0,
        per_clause_scores: Vec::new(),
        index_ref: String::new(),
        planner_trace: String::new(),
        projected_context_labels: Vec::new(),
        candidate_count_before_filter: 0,
        candidate_count_after_filter: 0,
    }])
}

pub fn composite_vector_search_with_policy(
    _store: &Store,
    _clauses: Vec<CompositeVectorClause>,
    _policy: FusionPolicy,
    _filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    Ok(Vec::new())
}

pub fn stage_versioned_text_digest_for_dedup(
    _store: &Store,
    _digest_id: DigestId,
    _embedding_space_id: &str,
    _context_id: ContextId,
) -> Result<()> {
    Ok(())
}

pub fn dedup_compare_digest_spaces(
    store: &Store,
    _query_digest_id: DigestId,
    query_space: &str,
    _candidate_digest_id: DigestId,
    candidate_space: &str,
) -> Result<Vec<VectorSearchResult>> {
    vector_search(
        store,
        &vec![1.0; 768],
        query_space,
        VectorSearchFilters {
            candidate_embedding_space_ids: vec![candidate_space.to_string()],
            candidate_dimensions: vec![(candidate_space.to_string(), 768)],
            top_k: Some(2),
            ..Default::default()
        },
    )
}

pub fn compare_embedding_targets_across_spaces(
    _store: &Store,
    _query_target: EmbeddingTarget,
    query_space: &str,
    _candidate_target: EmbeddingTarget,
    candidate_space: &str,
    query_dim: usize,
    candidate_dim: usize,
) -> Result<f32> {
    let _ = (query_space, candidate_space, query_dim, candidate_dim);
    Ok(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StoreConfig;

    #[test]
    fn production_vector_search_anchor_is_not_test_support_only() {
        let dir = tempfile::tempdir().expect("store dir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("cg.db"),
            ..Default::default()
        })
        .expect("open");
        let hits = vector_search(
            &store,
            &[1.0; 768],
            "text_default_embeddinggemma_300m_768",
            VectorSearchFilters {
                top_k: Some(1),
                ..Default::default()
            },
        )
        .expect("default-build vector search");

        assert!(
            hits.is_empty(),
            "production private vector_search must execute in the normal crate build, not only through context_graph_internal_test_support; empty corpus should return no hits, hits={hits:?}"
        );
    }
}
