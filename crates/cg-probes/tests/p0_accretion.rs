//! RED tests — P0 intention durability: authored-corpus accretion (shared keys +
//! established_here), and the dev-corpus clustering helpers (greedy first-seen clustering +
//! pre-existing fraction over clusters).

use cg_probes::accretion::{
    cluster_by_threshold, first_session_of_cluster, p0_authored, pre_existing_fraction,
};
use cg_probes::corpus::{ProbeDecision, ProbeGold, ProbeIntention, SessionRecord};
use std::collections::HashMap;

fn intention(key: &str, established_here: bool) -> ProbeIntention {
    ProbeIntention {
        key: key.to_string(),
        statement: format!("statement for {key}"),
        established_here,
        evidence_quote: None,
        notes: None,
    }
}

fn decision(statement: &str, serves: Option<&str>) -> ProbeDecision {
    ProbeDecision {
        statement: statement.to_string(),
        serves: serves.map(str::to_string),
        evidence_quote: None,
        made_in: None,
        notes: None,
    }
}

fn session(
    id: &str,
    order: u32,
    intentions: Vec<ProbeIntention>,
    decisions: Vec<ProbeDecision>,
) -> SessionRecord {
    SessionRecord {
        id: id.to_string(),
        domain: Some("d".to_string()),
        order,
        calibration: false,
        turns: Vec::new(),
        gold: ProbeGold {
            session: format!("{id}.txt"),
            domain: Some("d".to_string()),
            order,
            calibration: false,
            drafted_by: None,
            approved_by: None,
            intentions,
            decisions,
            entities: Vec::new(),
            non_decisions: Vec::new(),
            unsupported_claims: Vec::new(),
        },
    }
}

#[test]
fn authored_accretion_counts_pre_existing_only_across_sessions() {
    let sessions = vec![
        session(
            "s1",
            1,
            vec![intention("k1", true)],
            vec![decision("d1", Some("k1"))],
        ),
        session(
            "s2",
            2,
            vec![intention("k1", false), intention("k2", true)],
            vec![decision("d2", Some("k1")), decision("d3", Some("k2"))],
        ),
        session(
            "s3",
            3,
            vec![intention("k1", false)],
            vec![decision("d4", Some("k1")), decision("d5", None)],
        ),
    ];
    let report = p0_authored(&sessions);
    // d1 (k1 established same session) NOT pre-existing; d2 pre-existing; d3 (k2 same session)
    // not; d4 pre-existing; d5 unattached.
    assert_eq!(report.numbers.attached, 4);
    assert_eq!(report.numbers.pre_existing, 2);
    assert_eq!(report.numbers.unattached, 1);
    assert!((report.numbers.fraction - 0.5).abs() < 1e-6);
    assert_eq!(report.rows.len(), 4);
    assert!(!report.rows[0].pre_existing);
    assert!(report.rows[1].pre_existing);
    assert_eq!(report.rows[1].session_id, "s2");
}

#[test]
fn greedy_clustering_joins_best_matching_earlier_item() {
    // sim matrix (symmetric): 0-1: 0.8, 0-2: 0.3, 1-2: 0.6
    let sim = vec![
        vec![1.0, 0.8, 0.3],
        vec![0.8, 1.0, 0.6],
        vec![0.3, 0.6, 1.0],
    ];
    // threshold 0.7: item1 joins item0; item2's best earlier is item1 at 0.6 < 0.7 → new cluster.
    assert_eq!(cluster_by_threshold(&sim, 0.7), vec![0, 0, 1]);
    // threshold 0.5: item2 joins item1's cluster (0).
    assert_eq!(cluster_by_threshold(&sim, 0.5), vec![0, 0, 0]);
    // threshold above everything: all singletons, dense ids.
    assert_eq!(cluster_by_threshold(&sim, 0.95), vec![0, 1, 2]);
}

#[test]
fn pre_existing_fraction_uses_strictly_earlier_sessions() {
    // Intentions in corpus order: (session, cluster)
    let ordered = vec![(0usize, 0usize), (1, 0), (1, 1), (2, 0)];
    let first_seen = first_session_of_cluster(&ordered);
    assert_eq!(first_seen.get(&0), Some(&0));
    assert_eq!(first_seen.get(&1), Some(&1));

    let decisions = vec![
        (0usize, Some(0usize)), // cluster 0 first seen in this session → not pre-existing
        (1, Some(0)),           // pre-existing
        (1, Some(1)),           // first seen here → not
        (2, Some(0)),           // pre-existing
        (2, None),              // unattached
    ];
    let numbers = pre_existing_fraction(&decisions, &first_seen);
    assert_eq!(numbers.attached, 4);
    assert_eq!(numbers.pre_existing, 2);
    assert_eq!(numbers.unattached, 1);
    assert!((numbers.fraction - 0.5).abs() < 1e-6);
}

#[test]
fn empty_corpus_reports_zero_without_panicking() {
    let report = p0_authored(&[]);
    assert_eq!(report.numbers.attached, 0);
    assert!((report.numbers.fraction - 0.0).abs() < 1e-6);
    let numbers = pre_existing_fraction(&[], &HashMap::new());
    assert_eq!(numbers.attached, 0);
}
