use std::collections::{HashMap, HashSet};

use contextdb_core::{TxId, Value};

use crate::{
    CgError, ContextId, DigestId, EmbeddingInput, EmbeddingTarget, EvidenceId, EvidenceKind,
    EvidenceRef, Result, Store,
};

use super::convert::map_engine_err;
use super::encode::{column_index, get_json, get_txid, get_uuid, tags_from_json};
use super::evidence::vector_binding_for_target;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorSearchFilters {
    pub context_ids: Option<Vec<crate::ContextId>>,
    pub label_filters: Vec<(String, String)>,
    pub candidate_embedding_space_ids: Vec<String>,
    pub candidate_dimensions: Vec<(String, usize)>,
    pub top_k: Option<usize>,
    pub include_planner_trace: bool,
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
type DedupCompareDigestSpacesFn =
    fn(&Store, DigestId, &str, DigestId, &str) -> Result<Vec<VectorSearchResult>>;
type CompareEmbeddingTargetsFn =
    fn(&Store, EmbeddingTarget, &str, EmbeddingTarget, &str, usize, usize) -> Result<f32>;

pub(crate) fn production_primitive_anchor() {
    let _: VectorSearchFn = vector_search;
    let _: EvidenceSearchFn = evidence_search;
    let _: SemanticSearchFn = semantic_search;
    let _: CompositeVectorSearchFn = composite_vector_search_with_policy;
    let _: DedupCompareDigestSpacesFn = dedup_compare_digest_spaces;
    let _: CompareEmbeddingTargetsFn = compare_embedding_targets_across_spaces;
    let _ = CompositeVectorClause {
        query_vector: Vec::new(),
        embedding_space_id: String::new(),
    };
    let _ = FusionPolicy::WeightedSum(Vec::new());
}

pub fn vector_search(
    store: &Store,
    query_vector: &[f32],
    embedding_space_id: &str,
    filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    if matches!(filters.context_ids.as_ref(), Some(contexts) if contexts.is_empty()) {
        return Ok(Vec::new());
    }
    let space = scanned_embedding_space(store, embedding_space_id)?.ok_or_else(|| {
        CgError::UnknownEmbeddingSpace {
            id: embedding_space_id.to_string(),
        }
    })?;
    if query_vector.len() != space.dimension {
        return Err(CgError::EmbeddingDimensionMismatch {
            space_id: embedding_space_id.to_string(),
            expected: space.dimension as u32,
            found: query_vector.len() as u32,
        });
    }
    for candidate in &filters.candidate_embedding_space_ids {
        if candidate != embedding_space_id {
            let candidate_dim = filters
                .candidate_dimensions
                .iter()
                .find_map(|(space, dim)| (space == candidate).then_some(*dim))
                .or_else(|| {
                    scanned_embedding_space(store, candidate)
                        .ok()
                        .flatten()
                        .map(|space| space.dimension)
                });
            return Err(CgError::CrossSpaceQueryRefused {
                query_space: embedding_space_id.to_string(),
                candidate_space: candidate.clone(),
                query_dim: Some(space.dimension),
                candidate_dim,
            });
        }
    }

    let context_labels = if filters.label_filters.is_empty() {
        HashMap::new()
    } else {
        context_label_map(store)?
    };
    let allowed_contexts = allowed_contexts(&context_labels, &filters);
    let limit = filters.top_k.unwrap_or(10);
    let mut hits = Vec::new();
    for binding in &space.bindings {
        if !matches!(binding.table_name.as_str(), "evidence" | "digests") {
            continue;
        }
        let Some(handle_table) = handle_table_for_vector_table(&binding.table_name) else {
            continue;
        };
        let enforce_space_filter = vector_column_has_foreign_space_bindings_scanned(
            store,
            &binding.table_name,
            &binding.vector_column,
            embedding_space_id,
        )?;
        let active_target_ids = if enforce_space_filter {
            active_target_ids_for_vector_binding(
                store,
                handle_table,
                &binding.table_name,
                &binding.vector_column,
                embedding_space_id,
            )?
        } else {
            None
        };
        let (candidate_count_before_filter, candidate_count_after_filter) =
            if filters.include_planner_trace {
                candidate_counts_for_diagnostics_scanned(
                    store,
                    &binding.table_name,
                    handle_table,
                    embedding_space_id,
                    allowed_contexts.as_deref(),
                    enforce_space_filter,
                    active_target_ids.as_deref(),
                )?
            } else if allowed_contexts
                .as_ref()
                .is_some_and(|contexts| contexts.is_empty())
            {
                (0, 0)
            } else {
                (0, 1)
            };
        if filters.include_planner_trace && candidate_count_after_filter == 0 {
            continue;
        }
        let searched = execute_ranked_vector_queries(
            store,
            RankedVectorQuery {
                table: &binding.table_name,
                vector_column: &binding.vector_column,
                handle_table,
                embedding_space_id,
                query_vector,
                contexts: allowed_contexts.as_deref(),
                limit,
                include_planner_trace: filters.include_planner_trace,
                enforce_space_filter,
                active_target_ids: active_target_ids.as_deref(),
            },
        )?;
        for row in searched {
            let values = &row.values;
            let Some(target) = target_for_row(&binding.table_name, values) else {
                continue;
            };
            let context_id = row_context_id(values.get("context_id"));
            hits.push(VectorSearchResult {
                embedding_space_id: embedding_space_id.to_string(),
                target,
                context_id,
                score: row.score,
                per_clause_scores: Vec::new(),
                index_ref: format!("{}.{}", binding.table_name, binding.vector_column),
                planner_trace: row.planner_trace,
                projected_context_labels: context_id
                    .and_then(|id| context_labels.get(&id).cloned())
                    .unwrap_or_default(),
                candidate_count_before_filter,
                candidate_count_after_filter,
            });
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(limit);
    Ok(hits)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateCountMode {
    Full,
    Skip,
    SkipEmptyAdditionalBindings,
}

fn vector_search_with_count_mode(
    store: &Store,
    query_vector: &[f32],
    embedding_space_id: &str,
    filters: VectorSearchFilters,
    candidate_count_mode: CandidateCountMode,
    table_filter: Option<&str>,
) -> Result<Vec<VectorSearchResult>> {
    if matches!(filters.context_ids.as_ref(), Some(contexts) if contexts.is_empty()) {
        return Ok(Vec::new());
    }
    let space = store.embedding_space_for_id(embedding_space_id)?;
    if query_vector.len() != space.dimension {
        return Err(CgError::EmbeddingDimensionMismatch {
            space_id: embedding_space_id.to_string(),
            expected: space.dimension as u32,
            found: query_vector.len() as u32,
        });
    }
    for candidate in &filters.candidate_embedding_space_ids {
        if candidate != embedding_space_id {
            let candidate_dim = filters
                .candidate_dimensions
                .iter()
                .find_map(|(space, dim)| (space == candidate).then_some(*dim))
                .or_else(|| {
                    store
                        .embedding_space_for_id(candidate)
                        .ok()
                        .map(|space| space.dimension)
                });
            return Err(CgError::CrossSpaceQueryRefused {
                query_space: embedding_space_id.to_string(),
                candidate_space: candidate.clone(),
                query_dim: Some(space.dimension),
                candidate_dim,
            });
        }
    }

    let context_labels = if filters.label_filters.is_empty() {
        HashMap::new()
    } else {
        context_label_map(store)?
    };
    let allowed_contexts = allowed_contexts(&context_labels, &filters);
    let limit = filters.top_k.unwrap_or(10);
    let mut hits = Vec::new();
    for binding in &space.bindings {
        if table_filter.is_some_and(|table| binding.table_name != table) {
            continue;
        }
        if !matches!(binding.table_name.as_str(), "evidence" | "digests") {
            continue;
        }
        let Some(handle_table) = handle_table_for_vector_table(&binding.table_name) else {
            continue;
        };
        let enforce_space_filter = vector_column_has_foreign_space_bindings(
            store,
            &binding.table_name,
            &binding.vector_column,
            embedding_space_id,
        )?;
        let active_target_ids = if enforce_space_filter {
            active_target_ids_for_vector_binding(
                store,
                handle_table,
                &binding.table_name,
                &binding.vector_column,
                embedding_space_id,
            )?
        } else {
            None
        };
        let should_count_candidates = filters.include_planner_trace
            && (candidate_count_mode == CandidateCountMode::Full
                || (candidate_count_mode == CandidateCountMode::SkipEmptyAdditionalBindings
                    && binding.table_name != "evidence"));
        let (candidate_count_before_filter, candidate_count_after_filter) =
            if should_count_candidates {
                candidate_counts_for_diagnostics(
                    store,
                    &binding.table_name,
                    handle_table,
                    embedding_space_id,
                    allowed_contexts.as_deref(),
                    enforce_space_filter,
                    active_target_ids.as_deref(),
                )?
            } else if allowed_contexts
                .as_ref()
                .is_some_and(|contexts| contexts.is_empty())
            {
                (0, 0)
            } else {
                (0, 1)
            };
        if filters.include_planner_trace && candidate_count_after_filter == 0 {
            continue;
        }
        let searched = execute_ranked_vector_queries(
            store,
            RankedVectorQuery {
                table: &binding.table_name,
                vector_column: &binding.vector_column,
                handle_table,
                embedding_space_id,
                query_vector,
                contexts: allowed_contexts.as_deref(),
                limit,
                include_planner_trace: filters.include_planner_trace,
                enforce_space_filter,
                active_target_ids: active_target_ids.as_deref(),
            },
        )?;
        for row in searched {
            let values = &row.values;
            let Some(target) = target_for_row(&binding.table_name, values) else {
                continue;
            };
            let context_id = row_context_id(values.get("context_id"));
            hits.push(VectorSearchResult {
                embedding_space_id: embedding_space_id.to_string(),
                target,
                context_id,
                score: row.score,
                per_clause_scores: Vec::new(),
                index_ref: format!("{}.{}", binding.table_name, binding.vector_column),
                planner_trace: row.planner_trace,
                projected_context_labels: context_id
                    .and_then(|id| context_labels.get(&id).cloned())
                    .unwrap_or_default(),
                candidate_count_before_filter,
                candidate_count_after_filter,
            });
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(limit);
    Ok(hits)
}

pub fn evidence_search(
    store: &Store,
    evidence: &EvidenceRef,
    embedding_space_id: Option<&str>,
    filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    let space_id = match embedding_space_id {
        Some(space_id) => space_id.to_string(),
        None => store
            .default_embedding_spaces
            .get(&evidence.kind)
            .cloned()
            .ok_or(CgError::UnsupportedEvidenceKind {
                kind: evidence.kind,
            })?,
    };
    let space =
        store
            .embedding_space_for_id(&space_id)
            .map_err(|_| CgError::UnknownEmbeddingSpace {
                id: space_id.clone(),
            })?;
    let binding = vector_binding_for_target(&space, "evidence")?;

    if evidence_has_persisted_vector(store, evidence.id, &space_id, &binding.vector_column)? {
        return row_vector_search_from_evidence(
            store,
            evidence.id,
            &binding.vector_column,
            &space,
            filters,
        );
    }

    let embedder = store
        .embedders
        .get(&space_id)
        .ok_or_else(|| embedder_missing_error(evidence.kind))?;
    let output = embedder.embed(embedding_input_for_evidence(evidence)?)?;
    if output.embedding_space_id != space_id {
        return Err(CgError::EmbeddingSpaceIdMismatch {
            embedder_declared: output.embedding_space_id,
            registry_declared: space_id,
        });
    }
    if output.dimension != space.dimension || output.vector.len() != space.dimension {
        return Err(CgError::EmbeddingDimensionMismatch {
            space_id: space.embedding_space_id,
            expected: space.dimension as u32,
            found: output.vector.len() as u32,
        });
    }
    vector_search_with_count_mode(
        store,
        &output.vector,
        &output.embedding_space_id,
        filters,
        CandidateCountMode::Skip,
        Some("evidence"),
    )
}

pub fn semantic_search(
    store: &Store,
    text: &str,
    filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    let space_id = store
        .default_embedding_spaces
        .get(&EvidenceKind::Text)
        .cloned()
        .ok_or(CgError::TextEmbedderDisabled)?;
    let space = store.embedding_space_for_id(&space_id)?;
    let embedder = store
        .embedders
        .get(&space_id)
        .ok_or(CgError::TextEmbedderDisabled)?;
    let output = embedder.embed(EmbeddingInput::Text(text.to_string()))?;
    if output.embedding_space_id != space_id {
        return Err(CgError::EmbeddingSpaceIdMismatch {
            embedder_declared: output.embedding_space_id,
            registry_declared: space_id,
        });
    }
    if output.dimension != space.dimension || output.vector.len() != space.dimension {
        return Err(CgError::EmbeddingDimensionMismatch {
            space_id: space.embedding_space_id,
            expected: space.dimension as u32,
            found: output.vector.len() as u32,
        });
    }
    vector_search_with_count_mode(
        store,
        &output.vector,
        &output.embedding_space_id,
        filters,
        CandidateCountMode::SkipEmptyAdditionalBindings,
        None,
    )
}

pub fn composite_vector_search_with_policy(
    store: &Store,
    clauses: Vec<CompositeVectorClause>,
    policy: FusionPolicy,
    filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    let mut seen = HashSet::new();
    for clause in &clauses {
        if !seen.insert(clause.embedding_space_id.clone()) {
            return Err(CgError::CompositeQueryClausesShareSpace {
                space_id: clause.embedding_space_id.clone(),
            });
        }
    }
    let FusionPolicy::WeightedSum(weights) = policy;
    let weight_spaces = weights
        .iter()
        .map(|(space, _)| space.clone())
        .collect::<Vec<_>>();
    let weights = weights.into_iter().collect::<HashMap<_, _>>();
    let mut fused = HashMap::<EmbeddingTarget, VectorSearchResult>::new();
    for clause in clauses {
        let clause_hits = vector_search(
            store,
            &clause.query_vector,
            &clause.embedding_space_id,
            VectorSearchFilters {
                top_k: filters.top_k.or(Some(100)),
                ..filters.clone()
            },
        )?;
        let weight = weights
            .get(&clause.embedding_space_id)
            .copied()
            .unwrap_or(1.0);
        for hit in clause_hits {
            let entry = fused
                .entry(hit.target.clone())
                .or_insert_with(|| VectorSearchResult {
                    score: 0.0,
                    per_clause_scores: Vec::new(),
                    ..hit.clone()
                });
            entry.score += hit.score * weight;
            entry
                .per_clause_scores
                .push((clause.embedding_space_id.clone(), hit.score));
        }
    }
    let mut hits = fused.into_values().collect::<Vec<_>>();
    for hit in &mut hits {
        for space in &weight_spaces {
            if !hit
                .per_clause_scores
                .iter()
                .any(|(existing, _)| existing == space)
            {
                hit.per_clause_scores.push((space.clone(), 0.0));
            }
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(filters.top_k.unwrap_or(10));
    Ok(hits)
}

pub fn dedup_compare_digest_spaces(
    store: &Store,
    query_digest_id: DigestId,
    query_space: &str,
    candidate_digest_id: DigestId,
    candidate_space: &str,
) -> Result<Vec<VectorSearchResult>> {
    if query_space != candidate_space {
        return dedup_refuse_cross_space(store, query_space, candidate_space);
    }
    let Some(query_actual_space) = latest_digest_embedding_space(store, query_digest_id)? else {
        return Ok(vec![]);
    };
    let Some(candidate_actual_space) = latest_digest_embedding_space(store, candidate_digest_id)?
    else {
        return Ok(vec![]);
    };
    if query_actual_space != query_space
        || candidate_actual_space != candidate_space
        || query_actual_space != candidate_actual_space
    {
        if query_actual_space != candidate_actual_space {
            return dedup_refuse_cross_space(store, &query_actual_space, &candidate_actual_space);
        }
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "embedding_space_id",
            expected: query_space.to_string(),
            actual: query_actual_space,
        });
    }
    let dimension = dedup_embedding_space_dimension(store, query_space)?.ok_or_else(|| {
        CgError::UnknownEmbeddingSpace {
            id: query_space.to_string(),
        }
    })?;
    if !dedup_digest_space_has_vector_binding(store, query_space)? {
        return Ok(vec![]);
    }
    if dimension != 768 {
        return Err(CgError::EmbeddingDimensionMismatch {
            space_id: candidate_space.to_string(),
            expected: 768,
            found: dimension as u32,
        });
    }
    let params = HashMap::from([
        (
            "candidate_id".to_string(),
            Value::Uuid(candidate_digest_id.as_uuid()),
        ),
        (
            "query_id".to_string(),
            Value::Uuid(query_digest_id.as_uuid()),
        ),
    ]);
    let result = store
        .db
        .execute(
            "SELECT id, context_id, score FROM digests WHERE id = $candidate_id ORDER BY vector_text <=> ROW_VECTOR('digests','vector_text',$query_id) LIMIT 1",
            &params,
        )
        .map_err(dedup_engine_err)?;
    let Some(row) = result.rows.first() else {
        return Ok(vec![]);
    };
    let score_index = column_index(&result.columns, "score")?;
    let context_index = column_index(&result.columns, "context_id")?;
    let score = match row.get(score_index) {
        Some(Value::Float64(score)) => *score as f32,
        _ => 0.0,
    };
    let candidate_context = match row.get(context_index) {
        Some(Value::Uuid(id)) => ContextId::from(*id),
        _ => return Ok(vec![]),
    };
    Ok(vec![VectorSearchResult {
        embedding_space_id: query_space.to_string(),
        target: EmbeddingTarget::Digest(candidate_digest_id),
        context_id: Some(candidate_context),
        score,
        per_clause_scores: vec![],
        index_ref: "digests.vector_text".to_string(),
        planner_trace: format!("{:?}", result.trace),
        projected_context_labels: vec![],
        candidate_count_before_filter: 1,
        candidate_count_after_filter: 1,
    }])
}

fn dedup_refuse_cross_space(
    store: &Store,
    query_space: &str,
    candidate_space: &str,
) -> Result<Vec<VectorSearchResult>> {
    let query_dim = dedup_embedding_space_dimension(store, query_space)?.ok_or_else(|| {
        CgError::UnknownEmbeddingSpace {
            id: query_space.to_string(),
        }
    })?;
    let candidate_dim =
        dedup_embedding_space_dimension(store, candidate_space)?.ok_or_else(|| {
            CgError::UnknownEmbeddingSpace {
                id: candidate_space.to_string(),
            }
        })?;
    store.push_reindex_event(super::reindex::ReindexEvent::TriggerSuggested {
        from_space: candidate_space.to_string(),
        to_space: query_space.to_string(),
    });
    Err(CgError::CrossSpaceQueryRefused {
        query_space: query_space.to_string(),
        candidate_space: candidate_space.to_string(),
        query_dim: Some(query_dim),
        candidate_dim: Some(candidate_dim),
    })
}

fn latest_digest_embedding_space(store: &Store, digest_id: DigestId) -> Result<Option<String>> {
    let result = store
        .db
        .execute(
            "SELECT embedding_space_id FROM digest_embedding_handles WHERE target_id = $target_id ORDER BY created_at DESC LIMIT 1",
            &HashMap::from([(
                "target_id".to_string(),
                Value::Uuid(digest_id.as_uuid()),
            )]),
        )
        .map_err(dedup_engine_err)?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    let space_index = column_index(&result.columns, "embedding_space_id")?;
    Ok(match row.get(space_index) {
        Some(Value::Text(space)) => Some(space.clone()),
        _ => None,
    })
}

fn dedup_digest_space_has_vector_binding(store: &Store, id: &str) -> Result<bool> {
    let result = store
        .db
        .execute(
            "SELECT bindings FROM embedding_spaces WHERE embedding_space_id = $embedding_space_id LIMIT 1",
            &HashMap::from([(
                "embedding_space_id".to_string(),
                Value::Text(id.to_string()),
            )]),
        )
        .map_err(dedup_engine_err)?;
    let Some(row) = result.rows.first() else {
        return Ok(false);
    };
    let bindings_index = column_index(&result.columns, "bindings")?;
    let Some(Value::Json(serde_json::Value::Array(bindings))) = row.get(bindings_index) else {
        return Ok(false);
    };
    for binding in bindings {
        let table_name = binding.get("table_name").and_then(|value| value.as_str());
        let vector_column = binding
            .get("vector_column")
            .and_then(|value| value.as_str());
        if table_name == Some("digests") && vector_column == Some("vector_text") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn dedup_embedding_space_dimension(store: &Store, id: &str) -> Result<Option<usize>> {
    let result = store
        .db
        .execute(
            "SELECT dimension FROM embedding_spaces WHERE embedding_space_id = $embedding_space_id LIMIT 1",
            &HashMap::from([(
                "embedding_space_id".to_string(),
                Value::Text(id.to_string()),
            )]),
        )
        .map_err(dedup_engine_err)?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    let dimension_index = column_index(&result.columns, "dimension")?;
    Ok(match row.get(dimension_index) {
        Some(Value::Int64(value)) => Some(*value as usize),
        _ => Some(0),
    })
}

fn dedup_engine_err(err: contextdb_core::Error) -> CgError {
    CgError::Engine(format!("unexpected engine error: {err}"))
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
    Err(CgError::CrossSpaceQueryRefused {
        query_space: query_space.to_string(),
        candidate_space: candidate_space.to_string(),
        query_dim: Some(query_dim),
        candidate_dim: Some(candidate_dim),
    })
}

fn context_label_map(store: &Store) -> Result<HashMap<ContextId, Vec<String>>> {
    let result = store
        .db
        .execute("SELECT id, labels FROM contexts", &HashMap::new())
        .map_err(map_engine_err)?;
    let mut out = HashMap::new();
    for row in &result.rows {
        let id = ContextId::from(get_uuid(row, &result.columns, "id")?);
        let labels = tags_from_json(&get_json(row, &result.columns, "labels")?);
        out.insert(id, labels);
    }
    Ok(out)
}

fn allowed_contexts(
    labels: &HashMap<ContextId, Vec<String>>,
    filters: &VectorSearchFilters,
) -> Option<Vec<ContextId>> {
    let mut contexts = filters.context_ids.clone();
    if !filters.label_filters.is_empty() {
        let matching = labels
            .iter()
            .filter_map(|(context_id, labels)| {
                filters
                    .label_filters
                    .iter()
                    .all(|(key, value)| {
                        labels
                            .iter()
                            .any(|label| label == &format!("{key}:{value}"))
                    })
                    .then_some(*context_id)
            })
            .collect::<Vec<_>>();
        contexts = match contexts {
            Some(existing) => Some(
                existing
                    .into_iter()
                    .filter(|context| matching.contains(context))
                    .collect(),
            ),
            None => Some(matching),
        };
    }
    contexts
}

fn row_context_id(value: Option<&Value>) -> Option<ContextId> {
    match value {
        Some(Value::Uuid(id)) => Some(ContextId::from(*id)),
        _ => None,
    }
}

fn target_for_row(table: &str, values: &HashMap<String, Value>) -> Option<EmbeddingTarget> {
    match (table, values.get("id")) {
        ("evidence", Some(Value::Uuid(id))) => {
            Some(EmbeddingTarget::Evidence(EvidenceId::from(*id)))
        }
        ("digests", Some(Value::Uuid(id))) => Some(EmbeddingTarget::Digest(DigestId::from(*id))),
        _ => None,
    }
}

fn evidence_has_persisted_vector(
    store: &Store,
    evidence_id: EvidenceId,
    embedding_space_id: &str,
    vector_column: &str,
) -> Result<bool> {
    if vector_column_has_foreign_space_bindings(
        store,
        "evidence",
        vector_column,
        embedding_space_id,
    )? {
        return Ok(active_target_ids_for_vector_binding(
            store,
            "evidence_embedding_handles",
            "evidence",
            vector_column,
            embedding_space_id,
        )?
        .is_some_and(|ids| ids.contains(&evidence_id.as_uuid())));
    }
    let result = store
        .db
        .execute(
            "SELECT id FROM evidence WHERE id = $id AND id IN (SELECT target_id FROM evidence_embedding_handles WHERE embedding_space_id = $embedding_space_id) LIMIT 1",
            &HashMap::from([
                ("id".to_string(), Value::Uuid(evidence_id.as_uuid())),
                (
                    "embedding_space_id".to_string(),
                    Value::Text(embedding_space_id.to_string()),
                ),
            ]),
        )
        .map_err(map_engine_err)?;
    Ok(!result.rows.is_empty())
}

fn row_vector_search_from_evidence(
    store: &Store,
    evidence_id: EvidenceId,
    source_vector_column: &str,
    space: &crate::EmbeddingSpace,
    filters: VectorSearchFilters,
) -> Result<Vec<VectorSearchResult>> {
    if matches!(filters.context_ids.as_ref(), Some(contexts) if contexts.is_empty()) {
        return Ok(Vec::new());
    }
    let context_labels = if filters.label_filters.is_empty() {
        HashMap::new()
    } else {
        context_label_map(store)?
    };
    let allowed_contexts = allowed_contexts(&context_labels, &filters);
    let limit = filters.top_k.unwrap_or(10);
    let mut hits = Vec::new();
    for binding in &space.bindings {
        if binding.table_name != "evidence" {
            continue;
        }
        let Some(handle_table) = handle_table_for_vector_table(&binding.table_name) else {
            continue;
        };
        let enforce_space_filter = vector_column_has_foreign_space_bindings(
            store,
            &binding.table_name,
            &binding.vector_column,
            &space.embedding_space_id,
        )?;
        let active_target_ids = if enforce_space_filter {
            active_target_ids_for_vector_binding(
                store,
                handle_table,
                &binding.table_name,
                &binding.vector_column,
                &space.embedding_space_id,
            )?
        } else {
            None
        };
        let (candidate_count_before_filter, candidate_count_after_filter) = if allowed_contexts
            .as_ref()
            .is_some_and(|contexts| contexts.is_empty())
        {
            (0, 0)
        } else {
            (0, 1)
        };
        if filters.include_planner_trace && candidate_count_after_filter == 0 {
            continue;
        }
        let searched = execute_ranked_row_vector_queries(
            store,
            RankedRowVectorQuery {
                table: &binding.table_name,
                vector_column: &binding.vector_column,
                handle_table,
                embedding_space_id: &space.embedding_space_id,
                source_table: "evidence",
                source_vector_column,
                source_id: evidence_id,
                contexts: allowed_contexts.as_deref(),
                limit,
                include_planner_trace: filters.include_planner_trace,
                enforce_space_filter,
                active_target_ids: active_target_ids.as_deref(),
            },
        )?;
        for row in searched {
            let values = &row.values;
            let Some(target) = target_for_row(&binding.table_name, values) else {
                continue;
            };
            let context_id = row_context_id(values.get("context_id"));
            hits.push(VectorSearchResult {
                embedding_space_id: space.embedding_space_id.clone(),
                target,
                context_id,
                score: row.score,
                per_clause_scores: Vec::new(),
                index_ref: format!("{}.{}", binding.table_name, binding.vector_column),
                planner_trace: row.planner_trace,
                projected_context_labels: context_id
                    .and_then(|id| context_labels.get(&id).cloned())
                    .unwrap_or_default(),
                candidate_count_before_filter,
                candidate_count_after_filter,
            });
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(limit);
    Ok(hits)
}

fn candidate_counts_for_diagnostics(
    store: &Store,
    table: &str,
    handle_table: &str,
    embedding_space_id: &str,
    contexts: Option<&[ContextId]>,
    enforce_space_filter: bool,
    active_target_ids: Option<&[uuid::Uuid]>,
) -> Result<(usize, usize)> {
    let before = count_rankable_rows(
        store,
        table,
        handle_table,
        embedding_space_id,
        None,
        enforce_space_filter,
        active_target_ids,
    )?;
    let after = match contexts {
        Some([]) => 0,
        Some(contexts) => {
            let mut total = 0usize;
            for context in contexts {
                total += count_rankable_rows(
                    store,
                    table,
                    handle_table,
                    embedding_space_id,
                    Some(*context),
                    enforce_space_filter,
                    active_target_ids,
                )?;
            }
            total
        }
        None => before,
    };
    Ok((before, after))
}

fn count_rankable_rows(
    store: &Store,
    table: &str,
    handle_table: &str,
    embedding_space_id: &str,
    context: Option<ContextId>,
    enforce_space_filter: bool,
    active_target_ids: Option<&[uuid::Uuid]>,
) -> Result<usize> {
    let mut params = HashMap::from([(
        "embedding_space_id".to_string(),
        Value::Text(embedding_space_id.to_string()),
    )]);
    let mut clauses = Vec::new();
    if let Some(context) = context {
        params.insert("context_id".to_string(), Value::Uuid(context.as_uuid()));
        clauses.push("context_id = $context_id".to_string());
    }
    if let Some(active_target_ids) = active_target_ids {
        if active_target_ids.is_empty() {
            return Ok(0);
        }
        clauses.push(active_target_filter_sql(&mut params, active_target_ids));
    } else if enforce_space_filter {
        clauses.push(format!(
            "id IN (SELECT target_id FROM {handle_table} WHERE embedding_space_id = $embedding_space_id)"
        ));
    }
    let where_clause = if clauses.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", clauses.join(" AND "))
    };
    let sql = format!("SELECT COUNT(*) AS n FROM {table}{where_clause}");
    let result = store.db.execute(&sql, &params).map_err(map_engine_err)?;
    let Some(row) = result.rows.first() else {
        return Ok(0);
    };
    let count_index = column_index(&result.columns, "n")?;
    Ok(match row.get(count_index) {
        Some(Value::Int64(value)) => *value as usize,
        _ => 0,
    })
}

fn embedding_input_for_evidence(evidence: &EvidenceRef) -> Result<EmbeddingInput> {
    match evidence.kind {
        EvidenceKind::Text => evidence
            .text_projection
            .clone()
            .map(EmbeddingInput::Text)
            .ok_or(CgError::EvidenceUnavailable),
        EvidenceKind::ImageFrame => Ok(EmbeddingInput::ImageFrame {
            uri: evidence.source_ref.clone(),
        }),
        EvidenceKind::AudioSegment => Ok(EmbeddingInput::AudioSegment {
            uri: evidence.source_ref.clone(),
        }),
        EvidenceKind::VideoSegment => Ok(EmbeddingInput::VideoSegment {
            uri: evidence.source_ref.clone(),
        }),
        EvidenceKind::StructuredSignal => Ok(EmbeddingInput::StructuredSignal(serde_json::json!({
            "source_ref": evidence.source_ref,
            "properties": evidence.properties,
        }))),
    }
}

fn embedder_missing_error(kind: EvidenceKind) -> CgError {
    if kind == EvidenceKind::Text {
        CgError::TextEmbedderDisabled
    } else {
        CgError::UnsupportedEvidenceKind { kind }
    }
}

#[derive(Debug, Clone)]
struct RankedVectorRow {
    values: HashMap<String, Value>,
    score: f32,
    planner_trace: String,
}

enum VectorQueryScope {
    Unrestricted,
    Context(ContextId),
}

struct RankedVectorQuery<'a> {
    table: &'a str,
    vector_column: &'a str,
    handle_table: &'a str,
    embedding_space_id: &'a str,
    query_vector: &'a [f32],
    contexts: Option<&'a [ContextId]>,
    limit: usize,
    include_planner_trace: bool,
    enforce_space_filter: bool,
    active_target_ids: Option<&'a [uuid::Uuid]>,
}

struct RankedRowVectorQuery<'a> {
    table: &'a str,
    vector_column: &'a str,
    handle_table: &'a str,
    embedding_space_id: &'a str,
    source_table: &'a str,
    source_vector_column: &'a str,
    source_id: EvidenceId,
    contexts: Option<&'a [ContextId]>,
    limit: usize,
    include_planner_trace: bool,
    enforce_space_filter: bool,
    active_target_ids: Option<&'a [uuid::Uuid]>,
}

fn execute_ranked_vector_queries(
    store: &Store,
    query: RankedVectorQuery<'_>,
) -> Result<Vec<RankedVectorRow>> {
    if matches!(query.active_target_ids, Some(ids) if ids.is_empty()) {
        return Ok(Vec::new());
    }
    let scopes = query.contexts.map_or_else(
        || vec![VectorQueryScope::Unrestricted],
        |contexts| {
            contexts
                .iter()
                .copied()
                .map(VectorQueryScope::Context)
                .collect()
        },
    );
    let mut rows = Vec::new();
    for scope in scopes {
        let mut params = HashMap::from([
            (
                "query".to_string(),
                Value::Vector(query.query_vector.to_vec()),
            ),
            (
                "embedding_space_id".to_string(),
                Value::Text(query.embedding_space_id.to_string()),
            ),
        ]);
        let unrestricted = matches!(scope, VectorQueryScope::Unrestricted);
        let active_filter = query
            .active_target_ids
            .map(|ids| active_target_filter_sql(&mut params, ids));
        let sql = match scope {
            VectorQueryScope::Unrestricted if query.enforce_space_filter => {
                let active_filter = active_filter.unwrap_or_else(|| {
                    format!(
                        "id IN (SELECT target_id FROM {} WHERE embedding_space_id = $embedding_space_id)",
                        query.handle_table
                    )
                });
                format!(
                    "SELECT id, context_id, score FROM {table} WHERE {active_filter} ORDER BY {vector_column} <=> $query LIMIT {limit}",
                    table = query.table,
                    active_filter = active_filter,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
            VectorQueryScope::Unrestricted => {
                format!(
                    "SELECT id, context_id, score FROM {table} ORDER BY {vector_column} <=> $query LIMIT {limit}",
                    table = query.table,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
            VectorQueryScope::Context(context) if query.enforce_space_filter => {
                params.insert("context_id".to_string(), Value::Uuid(context.as_uuid()));
                let active_filter = active_filter.unwrap_or_else(|| {
                    format!(
                        "id IN (SELECT target_id FROM {} WHERE embedding_space_id = $embedding_space_id)",
                        query.handle_table
                    )
                });
                format!(
                    "SELECT id, context_id, score FROM {table} WHERE context_id = $context_id AND {active_filter} ORDER BY {vector_column} <=> $query LIMIT {limit}",
                    table = query.table,
                    active_filter = active_filter,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
            VectorQueryScope::Context(context) => {
                params.insert("context_id".to_string(), Value::Uuid(context.as_uuid()));
                format!(
                    "SELECT id, context_id, score FROM {table} WHERE context_id = $context_id ORDER BY {vector_column} <=> $query LIMIT {limit}",
                    table = query.table,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
        };
        let result = store.db.execute(&sql, &params).map_err(map_engine_err)?;
        let trace = if query.include_planner_trace {
            format!("{:?}", result.trace)
        } else {
            String::new()
        };
        let id_column = result
            .columns
            .iter()
            .position(|column| column == "id")
            .ok_or_else(|| CgError::Engine("vector search row lookup missing id".to_string()))?;
        let score_column = result.columns.iter().position(|column| column == "score");
        for row in result.rows {
            let id = match row.get(id_column) {
                Some(Value::Uuid(id)) => *id,
                Some(_) => uuid::Uuid::nil(),
                None => uuid::Uuid::nil(),
            };
            if id.is_nil() {
                continue;
            }
            let values = row_values_by_column(&result.columns, &row);
            let score = score_column
                .and_then(|column| row.get(column))
                .and_then(|value| match value {
                    Value::Float64(score) => Some(*score as f32),
                    _ => None,
                })
                .unwrap_or(0.0);
            rows.push(RankedVectorRow {
                values,
                score,
                planner_trace: trace.clone(),
            });
        }
        if unrestricted {
            break;
        }
    }
    rows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(query.limit);
    Ok(rows)
}

fn execute_ranked_row_vector_queries(
    store: &Store,
    query: RankedRowVectorQuery<'_>,
) -> Result<Vec<RankedVectorRow>> {
    if matches!(query.active_target_ids, Some(ids) if ids.is_empty()) {
        return Ok(Vec::new());
    }
    let scopes = query.contexts.map_or_else(
        || vec![VectorQueryScope::Unrestricted],
        |contexts| {
            contexts
                .iter()
                .copied()
                .map(VectorQueryScope::Context)
                .collect()
        },
    );
    let mut rows = Vec::new();
    for scope in scopes {
        let mut params = HashMap::from([
            (
                "embedding_space_id".to_string(),
                Value::Text(query.embedding_space_id.to_string()),
            ),
            (
                "query_id".to_string(),
                Value::Uuid(query.source_id.as_uuid()),
            ),
        ]);
        let unrestricted = matches!(scope, VectorQueryScope::Unrestricted);
        let row_vector = format!(
            "ROW_VECTOR('{source_table}','{source_vector_column}',$query_id)",
            source_table = query.source_table,
            source_vector_column = query.source_vector_column,
        );
        let active_filter = query
            .active_target_ids
            .map(|ids| active_target_filter_sql(&mut params, ids));
        let sql = match scope {
            VectorQueryScope::Unrestricted if query.enforce_space_filter => {
                let active_filter = active_filter.unwrap_or_else(|| {
                    format!(
                        "id IN (SELECT target_id FROM {} WHERE embedding_space_id = $embedding_space_id)",
                        query.handle_table
                    )
                });
                format!(
                    "SELECT id, context_id, score FROM {table} WHERE {active_filter} ORDER BY {vector_column} <=> {row_vector} LIMIT {limit}",
                    table = query.table,
                    active_filter = active_filter,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
            VectorQueryScope::Unrestricted => {
                format!(
                    "SELECT id, context_id, score FROM {table} ORDER BY {vector_column} <=> {row_vector} LIMIT {limit}",
                    table = query.table,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
            VectorQueryScope::Context(context) if query.enforce_space_filter => {
                params.insert("context_id".to_string(), Value::Uuid(context.as_uuid()));
                let active_filter = active_filter.unwrap_or_else(|| {
                    format!(
                        "id IN (SELECT target_id FROM {} WHERE embedding_space_id = $embedding_space_id)",
                        query.handle_table
                    )
                });
                format!(
                    "SELECT id, context_id, score FROM {table} WHERE context_id = $context_id AND {active_filter} ORDER BY {vector_column} <=> {row_vector} LIMIT {limit}",
                    table = query.table,
                    active_filter = active_filter,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
            VectorQueryScope::Context(context) => {
                params.insert("context_id".to_string(), Value::Uuid(context.as_uuid()));
                format!(
                    "SELECT id, context_id, score FROM {table} WHERE context_id = $context_id ORDER BY {vector_column} <=> {row_vector} LIMIT {limit}",
                    table = query.table,
                    vector_column = query.vector_column,
                    limit = query.limit,
                )
            }
        };
        let result = store.db.execute(&sql, &params).map_err(map_engine_err)?;
        let trace = if query.include_planner_trace {
            format!("{:?}", result.trace)
        } else {
            String::new()
        };
        let id_column = result
            .columns
            .iter()
            .position(|column| column == "id")
            .ok_or_else(|| {
                CgError::Engine("row vector search row lookup missing id".to_string())
            })?;
        let score_column = result.columns.iter().position(|column| column == "score");
        for row in result.rows {
            let id = match row.get(id_column) {
                Some(Value::Uuid(id)) => *id,
                Some(_) => uuid::Uuid::nil(),
                None => uuid::Uuid::nil(),
            };
            if id.is_nil() {
                continue;
            }
            let values = row_values_by_column(&result.columns, &row);
            let score = score_column
                .and_then(|column| row.get(column))
                .and_then(|value| match value {
                    Value::Float64(score) => Some(*score as f32),
                    _ => None,
                })
                .unwrap_or(0.0);
            rows.push(RankedVectorRow {
                values,
                score,
                planner_trace: trace.clone(),
            });
        }
        if unrestricted {
            break;
        }
    }
    rows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(query.limit);
    Ok(rows)
}

fn handle_table_for_vector_table(table: &str) -> Option<&'static str> {
    match table {
        "evidence" => Some("evidence_embedding_handles"),
        "digests" => Some("digest_embedding_handles"),
        _ => None,
    }
}

fn active_target_filter_sql(
    params: &mut HashMap<String, Value>,
    active_target_ids: &[uuid::Uuid],
) -> String {
    let mut placeholders = Vec::with_capacity(active_target_ids.len());
    for (index, id) in std::iter::zip(0usize.., active_target_ids) {
        let param = format!("active_target_id_{index}");
        params.insert(param.clone(), Value::Uuid(*id));
        placeholders.push(format!("${param}"));
    }
    format!("id IN ({})", placeholders.join(", "))
}

fn active_target_ids_for_vector_binding(
    store: &Store,
    handle_table: &str,
    table_name: &str,
    vector_column: &str,
    embedding_space_id: &str,
) -> Result<Option<Vec<uuid::Uuid>>> {
    let binding_space_ids = binding_space_ids_for_vector_column(store, table_name, vector_column)?;
    if binding_space_ids.len() <= 1 {
        return Ok(None);
    }
    let mut params = HashMap::new();
    let mut space_filter_parts = Vec::with_capacity(binding_space_ids.len());
    for (index, space_id) in std::iter::zip(0usize.., &binding_space_ids) {
        let param = format!("binding_space_id_{index}");
        params.insert(param.clone(), Value::Text(space_id.clone()));
        space_filter_parts.push(format!("embedding_space_id = ${param}"));
    }
    let space_filter = space_filter_parts.join(" OR ");
    let result = store
        .db
        .execute(
            &format!(
                "SELECT target_id, embedding_space_id, created_at FROM {handle_table} WHERE {space_filter}"
            ),
            &params,
        )
        .map_err(map_engine_err)?;
    let mut latest_by_target = HashMap::<uuid::Uuid, (TxId, String)>::new();
    let space_column = column_index(&result.columns, "embedding_space_id")?;
    for row in &result.rows {
        let target_id = get_uuid(row, &result.columns, "target_id")?;
        let created_at = get_txid(row, &result.columns, "created_at")?;
        let space_id = match row.get(space_column) {
            Some(Value::Text(space_id)) => space_id.clone(),
            other => {
                return Err(CgError::Engine(format!(
                    "expected embedding_space_id in handle row, got {other:?}"
                )));
            }
        };
        latest_by_target
            .entry(target_id)
            .and_modify(|existing| {
                if created_at >= existing.0 {
                    *existing = (created_at, space_id.clone());
                }
            })
            .or_insert((created_at, space_id));
    }
    let mut active = latest_by_target
        .into_iter()
        .filter_map(|(target_id, (_, space_id))| {
            (space_id == embedding_space_id).then_some(target_id)
        })
        .collect::<Vec<_>>();
    active.sort();
    Ok(Some(active))
}

fn binding_space_ids_for_vector_column(
    store: &Store,
    table_name: &str,
    vector_column: &str,
) -> Result<Vec<String>> {
    let result = store
        .db
        .execute(
            "SELECT embedding_space_id FROM embedding_space_bindings WHERE table_name = $table_name AND vector_column = $vector_column",
            &HashMap::from([
                ("table_name".to_string(), Value::Text(table_name.to_string())),
                ("vector_column".to_string(), Value::Text(vector_column.to_string())),
            ]),
        )
        .map_err(map_engine_err)?;
    let space_column = column_index(&result.columns, "embedding_space_id")?;
    let mut space_ids = result
        .rows
        .iter()
        .filter_map(|row| match row.get(space_column) {
            Some(Value::Text(space_id)) => Some(space_id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    space_ids.sort();
    space_ids.dedup();
    Ok(space_ids)
}

fn vector_column_has_foreign_space_bindings(
    store: &Store,
    table_name: &str,
    vector_column: &str,
    embedding_space_id: &str,
) -> Result<bool> {
    Ok(store.list_embedding_spaces()?.into_iter().any(|space| {
        if space.embedding_space_id == embedding_space_id {
            return false;
        }
        space.bindings.iter().any(|binding| {
            binding.table_name == table_name && binding.vector_column == vector_column
        })
    }))
}

fn vector_column_has_foreign_space_bindings_scanned(
    store: &Store,
    table_name: &str,
    vector_column: &str,
    embedding_space_id: &str,
) -> Result<bool> {
    let current_space = Value::Text(embedding_space_id.to_string());
    Ok(store
        .db
        .scan("embedding_spaces", store.db.snapshot())
        .map_err(map_engine_err)?
        .into_iter()
        .any(|row| {
            if row.values.get("embedding_space_id") == Some(&current_space) {
                return false;
            }
            let Some(Value::Json(serde_json::Value::Array(bindings))) = row.values.get("bindings")
            else {
                return false;
            };
            bindings.iter().any(|binding| {
                binding.get("table_name").and_then(|value| value.as_str()) == Some(table_name)
                    && binding
                        .get("vector_column")
                        .and_then(|value| value.as_str())
                        == Some(vector_column)
            })
        }))
}

fn candidate_counts_for_diagnostics_scanned(
    store: &Store,
    table: &str,
    handle_table: &str,
    embedding_space_id: &str,
    contexts: Option<&[ContextId]>,
    enforce_space_filter: bool,
    active_target_ids: Option<&[uuid::Uuid]>,
) -> Result<(usize, usize)> {
    let rows = store
        .db
        .scan(table, store.db.snapshot())
        .map_err(map_engine_err)?;
    let before = rows.len();
    let handles = if let Some(active_target_ids) = active_target_ids {
        active_target_ids.iter().copied().collect::<HashSet<_>>()
    } else if enforce_space_filter {
        store
            .db
            .scan(handle_table, store.db.snapshot())
            .map_err(map_engine_err)?
            .into_iter()
            .filter_map(|row| {
                (row.values.get("embedding_space_id")
                    == Some(&Value::Text(embedding_space_id.to_string())))
                .then(|| match row.values.get("target_id") {
                    Some(Value::Uuid(id)) => Some(*id),
                    _ => None,
                })?
            })
            .collect::<std::collections::HashSet<_>>()
    } else {
        std::collections::HashSet::new()
    };
    let after = rows
        .iter()
        .filter(|row| {
            let Some(Value::Uuid(id)) = row.values.get("id") else {
                return false;
            };
            if enforce_space_filter && !handles.contains(id) {
                return false;
            }
            contexts.is_none_or(|contexts| {
                row_context_id(row.values.get("context_id"))
                    .is_some_and(|context_id| contexts.contains(&context_id))
            })
        })
        .count();
    Ok((before, after))
}

fn scanned_embedding_space(store: &Store, id: &str) -> Result<Option<SearchEmbeddingSpace>> {
    let spaces = store
        .db
        .scan("embedding_spaces", store.db.snapshot())
        .map_err(map_engine_err)?;
    let Some(row) = spaces
        .into_iter()
        .find(|row| row.values.get("embedding_space_id") == Some(&Value::Text(id.to_string())))
    else {
        return Ok(None);
    };
    let dimension = match row.values.get("dimension") {
        Some(Value::Int64(value)) => *value as usize,
        _ => 0,
    };
    let mut bindings = store
        .db
        .scan("embedding_space_bindings", store.db.snapshot())
        .map_err(map_engine_err)?
        .into_iter()
        .filter_map(|binding| {
            (binding.values.get("embedding_space_id") == Some(&Value::Text(id.to_string()))).then(
                || SearchBinding {
                    table_name: match binding.values.get("table_name") {
                        Some(Value::Text(value)) => value.clone(),
                        _ => String::new(),
                    },
                    vector_column: match binding.values.get("vector_column") {
                        Some(Value::Text(value)) => value.clone(),
                        _ => String::new(),
                    },
                },
            )
        })
        .collect::<Vec<_>>();
    if bindings.is_empty()
        && let Some(Value::Json(serde_json::Value::Array(json_bindings))) =
            row.values.get("bindings")
    {
        bindings = json_bindings
            .iter()
            .filter_map(|binding| {
                Some(SearchBinding {
                    table_name: binding.get("table_name")?.as_str()?.to_string(),
                    vector_column: binding.get("vector_column")?.as_str()?.to_string(),
                })
            })
            .collect();
    }
    Ok(Some(SearchEmbeddingSpace {
        dimension,
        bindings,
    }))
}

#[derive(Debug, Clone)]
struct SearchEmbeddingSpace {
    dimension: usize,
    bindings: Vec<SearchBinding>,
}

#[derive(Debug, Clone)]
struct SearchBinding {
    table_name: String,
    vector_column: String,
}

fn row_values_by_column(columns: &[String], row: &[Value]) -> HashMap<String, Value> {
    let mut values = HashMap::new();
    let mut index = 0usize;
    while index < columns.len() && index < row.len() {
        values.insert(columns[index].clone(), row[index].clone());
        index += 1;
    }
    values
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use chrono::Utc;

    use super::*;
    use crate::{
        ContextId, DistanceMetric, Embedder, EmbedderConfig, EmbedderRegistration, EmbeddingId,
        EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding, EntityId, EvidenceId,
        EvidenceProducer, ObservationId, PrecomputedEmbedding, RecordObservation, ReindexEvent,
        ReindexOptions, RetentionStatus, StoreConfig,
    };

    const NOMIC_TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
    const SIBLING_TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";

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
            "text_default_nomic_v1_5_768",
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

    #[test]
    fn semantic_and_evidence_search_execute_real_nomic_default_recall_path() {
        let dir = tempfile::tempdir().expect("store dir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("cg.db"),
            ..Default::default()
        })
        .expect("open");
        let context_id = ContextId::new_v7();
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let evidence = text_evidence(
            observation_id,
            evidence_id,
            context_id,
            "auth module chooses argon2id password hashing",
        );
        store
            .record_observation(text_observation(
                observation_id,
                EntityId::new_v7(),
                context_id,
                evidence.clone(),
                Vec::new(),
            ))
            .expect("record observation");

        let filters = VectorSearchFilters {
            context_ids: Some(vec![context_id]),
            top_k: Some(1),
            ..Default::default()
        };
        let semantic_hits = semantic_search(&store, "auth password hashing", filters.clone())
            .expect("semantic search");
        assert!(
            matches!(
                semantic_hits.first().map(|hit| &hit.target),
                Some(EmbeddingTarget::Evidence(id)) if *id == evidence_id
            ),
            "semantic_search must return the recorded Nomic text evidence, hits={semantic_hits:?}"
        );
        assert!(
            semantic_hits
                .iter()
                .all(|hit| hit.embedding_space_id == NOMIC_TEXT_SPACE),
            "semantic_search must route through the Nomic bundled default, hits={semantic_hits:?}"
        );

        let evidence_hits = evidence_search(&store, &evidence, Some(NOMIC_TEXT_SPACE), filters)
            .expect("evidence search");
        assert!(
            matches!(
                evidence_hits.first().map(|hit| &hit.target),
                Some(EmbeddingTarget::Evidence(id)) if *id == evidence_id
            ),
            "evidence_search must find the persisted Nomic evidence vector, hits={evidence_hits:?}"
        );
        assert!(
            evidence_hits
                .iter()
                .all(|hit| hit.embedding_space_id == NOMIC_TEXT_SPACE),
            "evidence_search must keep Nomic space identity on every hit, hits={evidence_hits:?}"
        );
    }

    #[test]
    fn registered_embeddinggemma_space_searches_text_evidence_and_refuses_cross_space_vector_query()
    {
        let dir = tempfile::tempdir().expect("store dir");
        let mut sibling_vector = vec![0.0; 768];
        sibling_vector[0] = 1.0;
        let sibling = Arc::new(RegisteredTextEmbedder {
            space_id: SIBLING_TEXT_SPACE.to_string(),
            vector: sibling_vector.clone(),
        });
        let space = sibling.embedding_space();
        let store = Store::open_with_embedder_registrations(
            StoreConfig {
                db_path: dir.path().join("cg.db"),
                default_text_embedder: None,
                extra_embedding_spaces: vec![space.clone()],
                ..Default::default()
            },
            vec![EmbedderRegistration {
                space,
                embedder: sibling.clone(),
            }],
        )
        .expect("open with sibling embedder");
        let context_id = ContextId::new_v7();
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let text = "billing worker retries idempotently after queue restart";
        let evidence = text_evidence(observation_id, evidence_id, context_id, text);
        store
            .record_observation(text_observation(
                observation_id,
                EntityId::new_v7(),
                context_id,
                evidence.clone(),
                vec![PrecomputedEmbedding {
                    id: EmbeddingId::new_v7(),
                    embedding_space_id: SIBLING_TEXT_SPACE.to_string(),
                    vector: sibling_vector.clone(),
                    target: EmbeddingTarget::Evidence(evidence_id),
                    produced_by: producer(),
                }],
            ))
            .expect("record observation");

        let filters = VectorSearchFilters {
            context_ids: Some(vec![context_id]),
            top_k: Some(1),
            ..Default::default()
        };
        let hits = evidence_search(&store, &evidence, Some(SIBLING_TEXT_SPACE), filters)
            .expect("registered sibling evidence search");
        assert!(
            matches!(
                hits.first().map(|hit| &hit.target),
                Some(EmbeddingTarget::Evidence(id)) if *id == evidence_id
            ),
            "registered sibling text space must be usable for text recall, hits={hits:?}"
        );
        assert!(
            hits.iter()
                .all(|hit| hit.embedding_space_id == SIBLING_TEXT_SPACE),
            "registered sibling recall must keep the sibling space id, hits={hits:?}"
        );

        let cross_space = vector_search(
            &store,
            &sibling_vector,
            NOMIC_TEXT_SPACE,
            VectorSearchFilters {
                candidate_embedding_space_ids: vec![SIBLING_TEXT_SPACE.to_string()],
                candidate_dimensions: vec![(SIBLING_TEXT_SPACE.to_string(), 768)],
                top_k: Some(1),
                ..Default::default()
            },
        );
        assert!(
            matches!(
                cross_space,
                Err(CgError::CrossSpaceQueryRefused {
                    ref query_space,
                    ref candidate_space,
                    query_dim: Some(768),
                    candidate_dim: Some(768),
                }) if query_space == NOMIC_TEXT_SPACE
                    && candidate_space == SIBLING_TEXT_SPACE
            ),
            "cross-space Nomic-vs-sibling vector comparison must be refused, result={cross_space:?}"
        );
    }

    #[test]
    fn runtime_cross_embedder_switch_from_text_default_nomic_v1_5_768_to_text_default_embeddinggemma_300m_768_via_existing_reindex_no_new_api_surface()
     {
        let dir = tempfile::tempdir().expect("store dir");
        let mut old_vector = vec![0.0; 768];
        old_vector[1] = 1.0;
        let mut reindexed_vector = vec![0.0; 768];
        reindexed_vector[2] = 1.0;
        let sibling = Arc::new(RegisteredTextEmbedder {
            space_id: SIBLING_TEXT_SPACE.to_string(),
            vector: reindexed_vector.clone(),
        });
        let sibling_space = sibling.embedding_space();
        let store = Store::open_with_embedder_registrations(
            StoreConfig {
                db_path: dir.path().join("cg.db"),
                extra_embedding_spaces: vec![sibling_space.clone()],
                ..Default::default()
            },
            vec![EmbedderRegistration {
                space: sibling_space,
                embedder: sibling,
            }],
        )
        .expect("open with Nomic default and registered sibling text space");
        let spaces = store.list_embedding_spaces().expect("embedding spaces");
        let context_id = ContextId::new_v7();
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let evidence = text_evidence(
            observation_id,
            evidence_id,
            context_id,
            "embeddinggemma reindex keeps prior text searchable",
        );

        assert!(
            spaces
                .iter()
                .any(|space| space.embedding_space_id == NOMIC_TEXT_SPACE)
        );
        assert!(
            spaces
                .iter()
                .any(|space| space.embedding_space_id == SIBLING_TEXT_SPACE)
        );
        store
            .record_observation(text_observation(
                observation_id,
                EntityId::new_v7(),
                context_id,
                evidence,
                vec![PrecomputedEmbedding {
                    id: EmbeddingId::new_v7(),
                    embedding_space_id: SIBLING_TEXT_SPACE.to_string(),
                    vector: old_vector,
                    target: EmbeddingTarget::Evidence(evidence_id),
                    produced_by: producer(),
                }],
            ))
            .expect("seed sibling target-space handle");
        store
            .reindex_space(SIBLING_TEXT_SPACE, ReindexOptions::default())
            .expect("reindex sibling target-space handle");
        let events = store.drain_reindex_events();
        assert!(
            events.iter().any(|event| matches!(
                event,
                ReindexEvent::Completed {
                    embedding_space_id,
                    rows_processed: 1,
                    ..
                } if embedding_space_id == SIBLING_TEXT_SPACE
            )),
            "expected sibling-space reindex completion, events={events:?}"
        );
        let hits = vector_search(
            &store,
            &reindexed_vector,
            SIBLING_TEXT_SPACE,
            VectorSearchFilters {
                top_k: Some(1),
                ..Default::default()
            },
        )
        .expect("query refreshed sibling-space vector");
        assert!(
            matches!(
                hits.first(),
                Some(hit)
                    if hit.embedding_space_id == SIBLING_TEXT_SPACE
                        && hit.target == EmbeddingTarget::Evidence(evidence_id)
                        && hit.score > 0.99
            ),
            "vector_search through sibling space must return the reindexed text evidence, hits={hits:?}"
        );
    }

    #[test]
    fn vector_search_with_embedding_space_id_text_default_nomic_v1_5_768_refuses_query_vectors_of_mismatched_dimension()
     {
        let dir = tempfile::tempdir().expect("store dir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("cg.db"),
            default_text_embedder: Some(EmbedderConfig::disabled()),
            ..Default::default()
        })
        .expect("open with disabled text embedder still seeds Nomic space");
        let query = vec![0.0; 384];

        let err = vector_search(
            &store,
            &query,
            NOMIC_TEXT_SPACE,
            VectorSearchFilters::default(),
        )
        .expect_err("384-dim query must be refused against the 768-dim Nomic space");

        assert!(
            matches!(
                err,
                CgError::EmbeddingDimensionMismatch {
                    ref space_id,
                    expected: 768,
                    found: 384
                } if space_id == NOMIC_TEXT_SPACE
            ),
            "expected typed Nomic dimension mismatch, got {err:?}"
        );
    }

    struct RegisteredTextEmbedder {
        space_id: String,
        vector: Vec<f32>,
    }

    impl Embedder for RegisteredTextEmbedder {
        fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
            match input {
                EmbeddingInput::Text(_) => Ok(EmbeddingOutput {
                    embedding_space_id: self.space_id.clone(),
                    vector: self.vector.clone(),
                    dimension: self.vector.len(),
                    producer: producer(),
                }),
                EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(..) => {
                    Err(CgError::UnsupportedEvidenceKind {
                        kind: EvidenceKind::ImageFrame,
                    })
                }
                EmbeddingInput::AudioSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::AudioSegment,
                }),
                EmbeddingInput::VideoSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::VideoSegment,
                }),
                EmbeddingInput::StructuredSignal(_) => Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::StructuredSignal,
                }),
            }
        }

        fn supported_inputs(&self) -> Vec<EvidenceKind> {
            vec![EvidenceKind::Text]
        }

        fn embedding_space(&self) -> EmbeddingSpace {
            EmbeddingSpace {
                embedding_space_id: self.space_id.clone(),
                dimension: self.vector.len(),
                metric: DistanceMetric::Cosine,
                model_family: "text".to_string(),
                model_name: "registered-text".to_string(),
                model_version: "1".to_string(),
                supported_evidence_kinds: vec![EvidenceKind::Text],
                default_for_evidence_kind: Some(EvidenceKind::Text),
                bindings: vec![EmbeddingSpaceBinding {
                    embedding_space_id: self.space_id.clone(),
                    table_name: "evidence".to_string(),
                    vector_column: "vector_text".to_string(),
                }],
            }
        }
    }

    fn text_evidence(
        observation_id: ObservationId,
        evidence_id: EvidenceId,
        context_id: ContextId,
        text: &str,
    ) -> EvidenceRef {
        EvidenceRef {
            id: evidence_id,
            observation_id,
            context_id,
            kind: EvidenceKind::Text,
            source_ref:
                "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            content_hash: Some(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            ),
            captured_at: Some(Utc::now()),
            producer: producer(),
            text_projection: Some(text.to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            ..Default::default()
        }
    }

    fn text_observation(
        id: ObservationId,
        entity_id: EntityId,
        context_id: ContextId,
        evidence: EvidenceRef,
        embeddings: Vec<PrecomputedEmbedding>,
    ) -> RecordObservation {
        RecordObservation {
            id,
            entity_id,
            context_id,
            observation_type: "unit".to_string(),
            source: "unit".to_string(),
            observed_at: Utc::now(),
            evidence: vec![evidence],
            observed_properties: BTreeMap::from([(
                "component".to_string(),
                serde_json::json!("auth"),
            )]),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings,
        }
    }

    fn producer() -> EvidenceProducer {
        EvidenceProducer {
            system: "unit".to_string(),
            model_name: "fixture".to_string(),
            model_version: "1".to_string(),
            pipeline_version: "1".to_string(),
        }
    }
}

/// The dedicated event-history space and the existing text-default space are both registered,
/// 768-dimensional, and never comparable: a cross-space query between them is refused with the
/// exact declared space ids, even at equal dimension. Reaches `vector_search` and
/// `VectorSearchFilters` directly as crate-internal items, so this runs in the default `cargo
/// test -p context-graph --lib` tier rather than behind a cfg gate.
///
/// The event-history query vector is obtained by actually preparing and embedding a query
/// through the real cached Nomic embedder's role-aware surface (`prepare_query` +
/// `embed_prepared_query`), never a hand-built vector, so this proves the capability that feeds
/// the space, not a same-dimension coincidence.
#[cfg(test)]
mod event_search_space_isolation_tests {
    use super::*;
    use crate::embedder::nomic::NomicTextEmbedder;
    use crate::embedding_space::{
        BuiltinSpaceRegistry, default_text_embedding_space, event_history_embedding_space,
    };
    use crate::{
        DistanceMetric, EmbedderConfig, EmbeddingSpace, EmbeddingSpaceBinding, EventSearchEmbedder,
        StoreConfig,
    };

    const EVENT_HISTORY_SPACE: &str = BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID;
    const TEXT_DEFAULT_SPACE: &str = BuiltinSpaceRegistry::TEXT_DEFAULT_ID;
    const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";

    /// The canonical, already-installed model directory this crate's production code resolves a
    /// cached Nomic model under: the one cache-root resolver, joined with the model id — called
    /// here directly rather than re-derived (mirrors
    /// `tests/common/nomic_real_model.rs::canonical_model_dir`, which an in-src `#[cfg(test)]`
    /// module cannot reach — it lives in a separate integration test binary, not this crate's own
    /// compiled unit).
    fn canonical_nomic_model_dir() -> std::path::PathBuf {
        crate::embedder::nomic::default_cache_root().join(NOMIC_MODEL_ID)
    }

    /// A local-source, offline, download-denying config over the canonical installed model
    /// directory, pinned to the exact revision the reserved event-history identity names — never
    /// `EmbedderConfig::default_nomic()`, which sources from Hugging Face and allows a network
    /// download. Mirrors `tests/common/nomic_real_model.rs::local_offline_config`.
    fn local_offline_nomic_config() -> EmbedderConfig {
        EmbedderConfig {
            source: crate::ModelSource::LocalPath(canonical_nomic_model_dir()),
            offline: true,
            cache_dir: None,
            pinned_revision: Some(crate::embedder::NOMIC_DEFAULT_REVISION.to_string()),
            network_policy: "deny-download".to_string(),
        }
    }

    /// A store carrying BOTH spaces and a real text-default embedder.
    ///
    /// The embedder matters: `semantic_search` embeds the caller's text through whatever the store
    /// registered for [`EvidenceKind::Text`] and short-circuits on `TextEmbedderDisabled` when
    /// there is none, so a store without one can never reach the cross-space refusal the arms
    /// below drive. It is the same offline, download-denying configuration over the canonical
    /// installed model the rest of this module uses — never `default_nomic()`, which could fetch.
    fn store_with_both_spaces() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().expect("store dir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("cg.db"),
            extra_embedding_spaces: vec![event_history_embedding_space()],
            default_text_embedder: Some(local_offline_nomic_config()),
            ..Default::default()
        })
        .expect("open store with both spaces registered");
        (dir, store)
    }

    #[test]
    fn semantic_search_preserves_a_space_lookup_engine_error() {
        let (_dir, mut store) = store_with_both_spaces();
        store.read_memory_limit_bytes = std::num::NonZeroUsize::new(1).unwrap();

        let result = semantic_search(&store, "query", VectorSearchFilters::default());
        assert!(
            matches!(&result, Err(CgError::Engine(_))),
            "a failed space lookup must retain its engine error, got {result:?}"
        );
    }

    /// The canonical production declaration ([`event_history_embedding_space`]) binds only to the
    /// dedicated event-history passage vector, never the `evidence`/`digests` population the
    /// text-default space owns.
    #[test]
    fn canonical_event_history_declaration_has_only_the_passage_binding() {
        let declaration = event_history_embedding_space();
        assert_eq!(
            declaration.bindings,
            vec![EmbeddingSpaceBinding {
                embedding_space_id: EVENT_HISTORY_SPACE.into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }]
        );
        assert_eq!(declaration.default_for_evidence_kind, None);
        assert_eq!(
            declaration.embedding_space_id,
            "event_history_nomic_v1_5_document_query_v1_768"
        );
    }

    #[test]
    fn dedicated_event_search_space_refuses_old_unlabelled_space_comparison() {
        assert_eq!(
            EVENT_HISTORY_SPACE,
            "event_history_nomic_v1_5_document_query_v1_768"
        );
        assert_eq!(
            default_text_embedding_space().dimension,
            event_history_embedding_space().dimension
        );
        assert_eq!(
            default_text_embedding_space().embedding_space_id,
            TEXT_DEFAULT_SPACE
        );

        let embedder = NomicTextEmbedder::new(local_offline_nomic_config());
        // `prepare_query`/`embed_prepared_query` come from the sealed `EventSearchEmbedder`
        // extension trait, blanket-implemented for every embedder, so they are reached here on
        // the concrete handle exactly as a caller holding one would reach them.
        // The event-history query vector must come from the real role-aware preparation surface —
        // never a hand-built same-dimension vector.
        let prepared = embedder
            .prepare_query("find the incident review from last quarter")
            .unwrap_or_else(|err| {
                panic!(
                    "event-search query preparation must be supported over the bundled Nomic embedder, got {err:?}"
                )
            });
        let query_output = embedder
            .embed_prepared_query(&prepared)
            .expect("embedding a prepared event-search query must succeed");
        assert_eq!(
            query_output.vector.len(),
            BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION
        );

        // The event-history space registers with no table binding at all (proved above), so
        // isolation is proved through the Store-independent comparability seam
        // (`compare_embedding_targets_across_spaces`) as well as through `vector_search` against
        // a store that has both spaces registered — both must refuse by exact id and dimension,
        // never by coincidence of a shared table.
        let (_dir, store) = store_with_both_spaces();
        let result = vector_search(
            &store,
            &query_output.vector,
            EVENT_HISTORY_SPACE,
            VectorSearchFilters {
                candidate_embedding_space_ids: vec![TEXT_DEFAULT_SPACE.to_string()],
                ..Default::default()
            },
        );

        assert!(
            matches!(
                &result,
                Err(CgError::CrossSpaceQueryRefused {
                    query_space,
                    candidate_space,
                    query_dim: Some(768),
                    candidate_dim: Some(768),
                }) if query_space == EVENT_HISTORY_SPACE && candidate_space == TEXT_DEFAULT_SPACE
            ),
            "a query against the dedicated event-history space must refuse comparison with the \
             unlabelled text-default space by exact id, got {result:?}"
        );

        // The other direction, and the one an application actually takes: `semantic_search` is
        // the store's own text-query entrypoint. It embeds the caller's plain text through the
        // store's text-default embedder and then compares — so a caller who names the
        // event-history space as a candidate is asking for exactly the comparison this space
        // exists to forbid, and gets it refused by exact id, not by dimension. Nothing else in the
        // estate reaches the refusal site through `semantic_search`; a same-model, same-dimension
        // shortcut added there would leave every other arm green.
        let through_the_text_query_entrypoint = semantic_search(
            &store,
            "find the incident review from last quarter",
            VectorSearchFilters {
                candidate_embedding_space_ids: vec![EVENT_HISTORY_SPACE.to_string()],
                ..Default::default()
            },
        );
        assert!(
            matches!(
                &through_the_text_query_entrypoint,
                Err(CgError::CrossSpaceQueryRefused {
                    query_space,
                    candidate_space,
                    query_dim: Some(768),
                    candidate_dim: Some(768),
                }) if query_space == TEXT_DEFAULT_SPACE && candidate_space == EVENT_HISTORY_SPACE
            ),
            "the store's own text-query entrypoint must refuse a candidate in the dedicated \
             event-history space by exact id — the two spaces are the same width, so a check that \
             compares only dimensions would let this through, got \
             {through_the_text_query_entrypoint:?}"
        );

        // SMOKE, not a proof: `compare_embedding_targets_across_spaces` returns the refusal
        // unconditionally for every input it is given, so this call cannot distinguish a working
        // isolation rule from a broken one. It stays only to keep the seam's error shape wired up
        // until it computes an answer; the two arms above are what prove the promise.
        let store_independent_result = compare_embedding_targets_across_spaces(
            &store,
            EmbeddingTarget::Evidence(EvidenceId::from(uuid::Uuid::nil())),
            EVENT_HISTORY_SPACE,
            EmbeddingTarget::Evidence(EvidenceId::from(uuid::Uuid::nil())),
            TEXT_DEFAULT_SPACE,
            query_output.vector.len(),
            BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION,
        );
        assert!(
            matches!(
                &store_independent_result,
                Err(CgError::CrossSpaceQueryRefused { .. })
            ),
            "the not-yet-computing comparability seam must still answer with the cross-space \
             refusal shape, got {store_independent_result:?}"
        );
    }

    /// A second, non-default owner of `evidence.vector_text` is refused at store open — the
    /// existing owner guard the canonical event-history declaration above deliberately never
    /// triggers because it carries no bindings at all.
    #[test]
    fn second_owner_of_evidence_vector_text_is_refused() {
        let dir = tempfile::tempdir().expect("store dir");
        let conflicting_space = EmbeddingSpace {
            embedding_space_id: "conflicting_evidence_vector_text_owner".to_string(),
            dimension: BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION,
            metric: DistanceMetric::Cosine,
            model_family: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_FAMILY.to_string(),
            model_name: "conflicting-owner".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: "conflicting_evidence_vector_text_owner".to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            }],
        };
        let result = Store::open(StoreConfig {
            db_path: dir.path().join("cg.db"),
            extra_embedding_spaces: vec![conflicting_space],
            ..Default::default()
        });
        assert!(
            matches!(
                &result,
                Err(CgError::EmbeddingSpaceConfigMismatch {
                    field: "bindings",
                    ..
                })
            ),
            "a second, non-default owner of evidence.vector_text must be refused at store open, \
             got {result:?}"
        );
    }
}
