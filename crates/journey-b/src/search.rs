//! Plan 6 cg-search stub for journey M3.
//!
//! Plan 6 is unshipped today. Acceptance test
//! `semantic_search_effective_confidence_ranking` requires
//! `effective_confidence = confidence × success_rate` ranking via
//! ComputedRankColumn (released in contextdb v1.0.0 per `plan-manifest.md`).
//! The stub:
//!   1. Issues a vector top-k via `Store::vector_top_k` (engine NamedVectorIndexes
//!      shipped in v1.0.0).
//!   2. Joins with the `outcomes` table to look up each Decision's
//!      `success_rate` (success → 1.0; failure → 0.0; no outcome → 1.0
//!      per `effective_confidence_formula_no_outcome`).
//!   3. Computes `effective_confidence = coalesce(confidence, 1.0) × coalesce(success_rate, 1.0)`
//!      per `effective_confidence_null_confidence_defaults_to_success_rate`.
//!   4. Filters or deprioritises Decisions whose status is `invalidated`
//!      (per `semantic_search_effective_confidence_ranking`'s "the rank-policy
//!      formula deprioritises it OR is filtered out" branch — we filter, the
//!      stricter branch).
//!   5. Sorts by effective_confidence desc, ties broken by similarity desc.
//!
//! Stub-vs-test divergence flagged in findings: the engine's rank-policy
//! re-ranks BEFORE the cosine top-k cutoff (per probe
//! `engine-probes/group_a/plan_6.rs::rank_policy_orders_by_effective_confidence_within_top_k`);
//! a post-rerank stub can drop a decision the engine would have surfaced.

use crate::store::Store;
use crate::types::{DecisionStatusSnapshot, ScoredResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionStatusFilter {
    /// Per `semantic_search_effective_confidence_ranking`: invalidated decisions
    /// are filtered out (the stricter of the two contract branches).
    ExcludeInvalidated,
    /// Allow invalidated decisions through (developer asks for "everything").
    All,
}

pub fn semantic_search(
    store: &Store,
    query: &str,
    k: usize,
    filter: DecisionStatusFilter,
) -> Vec<ScoredResult> {
    let raw = store.vector_top_k(query, k * 2); // overshoot for the rerank
    let mut out: Vec<ScoredResult> = raw
        .into_iter()
        .map(|(d_id, similarity, _row_id)| {
            let confidence = store.get_decision_confidence(d_id);
            let outcome = store.outcome_success(d_id);
            let success_rate = match outcome {
                None => 1.0,
                Some(true) => 1.0,
                Some(false) => 0.0,
            };
            let conf_mul = confidence.unwrap_or(1.0);
            let eff = Some(conf_mul * success_rate);
            let status = store.get_decision_status(d_id);
            ScoredResult {
                decision_id: d_id,
                similarity,
                effective_confidence: eff,
                status_at_query_time: status,
            }
        })
        .collect();
    match filter {
        DecisionStatusFilter::ExcludeInvalidated => {
            out.retain(|r| r.status_at_query_time != DecisionStatusSnapshot::Invalidated)
        }
        DecisionStatusFilter::All => {}
    }
    out.sort_by(|a, b| {
        b.effective_confidence
            .unwrap_or(0.0)
            .partial_cmp(&a.effective_confidence.unwrap_or(0.0))
            .unwrap()
            .then(b.similarity.partial_cmp(&a.similarity).unwrap())
    });
    out.truncate(k);
    out
}
