//! P0 — intention durability. No model at all: over a chronologically ordered corpus, what
//! fraction of gold decisions attach to an intention already established by an EARLIER session?
//! High fraction → per-session intention work is almost always attachment to a known intention
//! and the generative slot is a rare cold-start event; low fraction → the generative slot is
//! load-bearing (the pre-registered owner stop at < 0.50 on the dev corpus).

use crate::corpus::SessionRecord;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct P0Numbers {
    /// pre_existing / attached; 0.0 when nothing is attached.
    pub fraction: f32,
    /// Decisions with a serving intention.
    pub attached: usize,
    /// Attached decisions whose serving intention was established by an earlier session.
    pub pre_existing: usize,
    /// Decisions with no serving intention (reported, excluded from the fraction).
    pub unattached: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccretionRow {
    pub session_id: String,
    pub decision: String,
    pub intention: String,
    pub pre_existing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct P0Report {
    pub numbers: P0Numbers,
    pub rows: Vec<AccretionRow>,
}

/// P0 over the authored corpus, where cross-session intention identity is authored directly
/// (shared keys + `established_here`). Sessions must already be in corpus order.
pub fn p0_authored(sessions: &[SessionRecord]) -> P0Report {
    let _ = sessions;
    todo!("probe impl commit")
}

/// Greedy first-seen clustering for the dev corpus (where cross-session identity is NOT
/// authored): item i joins the cluster of its best-matching earlier item with
/// similarity >= threshold, else starts a new cluster. Returns one cluster id per item,
/// cluster ids dense in first-seen order.
pub fn cluster_by_threshold(similarity: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let _ = (similarity, threshold);
    todo!("probe impl commit")
}

/// First session index in which each cluster appears. Input: (session_index, cluster) per
/// intention, in corpus order.
pub fn first_session_of_cluster(ordered: &[(usize, usize)]) -> HashMap<usize, usize> {
    let _ = ordered;
    todo!("probe impl commit")
}

/// The P0 numbers from clustered intentions: decisions are (session_index, serving cluster or
/// None); pre-existing iff the serving cluster first appeared in a STRICTLY earlier session.
pub fn pre_existing_fraction(
    decisions: &[(usize, Option<usize>)],
    first_seen: &HashMap<usize, usize>,
) -> P0Numbers {
    let _ = (decisions, first_seen);
    todo!("probe impl commit")
}
