//! P0 — intention durability. No model at all: over a chronologically ordered corpus, what
//! fraction of gold decisions attach to an intention already established by an EARLIER session?
//! High fraction → per-session intention work is almost always attachment to a known intention
//! and the generative slot is a rare cold-start event; low fraction → the generative slot is
//! load-bearing (the pre-registered owner stop at < 0.50 on the dev corpus).

use crate::corpus::SessionRecord;
use std::collections::{HashMap, HashSet};

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

fn numbers(attached: usize, pre_existing: usize, unattached: usize) -> P0Numbers {
    P0Numbers {
        fraction: if attached == 0 {
            0.0
        } else {
            pre_existing as f32 / attached as f32
        },
        attached,
        pre_existing,
        unattached,
    }
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
    let mut established: HashSet<String> = HashSet::new();
    let mut rows = Vec::new();
    let (mut attached, mut pre_existing, mut unattached) = (0usize, 0usize, 0usize);
    for session in sessions {
        for decision in &session.gold.decisions {
            let Some(serves) = &decision.serves else {
                unattached += 1;
                continue;
            };
            attached += 1;
            // Pre-existing means established by a STRICTLY earlier session: keys established in
            // this session are only inserted after its decisions are scored.
            let pre = established.contains(serves);
            if pre {
                pre_existing += 1;
            }
            rows.push(AccretionRow {
                session_id: session.id.clone(),
                decision: decision.statement.clone(),
                intention: serves.clone(),
                pre_existing: pre,
            });
        }
        for intention in &session.gold.intentions {
            if intention.established_here {
                established.insert(intention.key.clone());
            }
        }
    }
    P0Report {
        numbers: numbers(attached, pre_existing, unattached),
        rows,
    }
}

/// Greedy first-seen clustering for the dev corpus (where cross-session identity is NOT
/// authored): item i joins the cluster of its best-matching earlier item with
/// similarity >= threshold, else starts a new cluster. Returns one cluster id per item,
/// cluster ids dense in first-seen order.
pub fn cluster_by_threshold(similarity: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let mut clusters: Vec<usize> = Vec::with_capacity(similarity.len());
    let mut next_id = 0usize;
    for (i, row) in similarity.iter().enumerate() {
        let best_earlier = (0..i).max_by(|&a, &b| {
            row[a]
                .partial_cmp(&row[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        match best_earlier {
            Some(j) if row[j] >= threshold => clusters.push(clusters[j]),
            _ => {
                clusters.push(next_id);
                next_id += 1;
            }
        }
    }
    clusters
}

/// First session index in which each cluster appears. Input: (session_index, cluster) per
/// intention, in corpus order.
pub fn first_session_of_cluster(ordered: &[(usize, usize)]) -> HashMap<usize, usize> {
    let mut first: HashMap<usize, usize> = HashMap::new();
    for &(session, cluster) in ordered {
        first
            .entry(cluster)
            .and_modify(|s| *s = (*s).min(session))
            .or_insert(session);
    }
    first
}

/// The P0 numbers from clustered intentions: decisions are (session_index, serving cluster or
/// None); pre-existing iff the serving cluster first appeared in a STRICTLY earlier session.
pub fn pre_existing_fraction(
    decisions: &[(usize, Option<usize>)],
    first_seen: &HashMap<usize, usize>,
) -> P0Numbers {
    let (mut attached, mut pre_existing, mut unattached) = (0usize, 0usize, 0usize);
    for &(session, cluster) in decisions {
        let Some(cluster) = cluster else {
            unattached += 1;
            continue;
        };
        attached += 1;
        if first_seen
            .get(&cluster)
            .is_some_and(|&first| first < session)
        {
            pre_existing += 1;
        }
    }
    numbers(attached, pre_existing, unattached)
}
