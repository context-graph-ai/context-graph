//! RED tests — exhaustive attachment pair generation (every decision × every roster intention;
//! distractors are never sampled) and P3-known scoring with abstention correctness.

use cg_probes::corpus::{ProbeDecision, ProbeGold, ProbeIntention, SessionRecord};
use cg_probes::pairs::{attachment_pairs, intention_roster, score_known_attachment};

fn record(
    id: &str,
    intentions: &[(&str, &str)],
    decisions: &[(&str, Option<&str>)],
) -> SessionRecord {
    SessionRecord {
        id: id.to_string(),
        domain: None,
        order: 0,
        calibration: false,
        turns: Vec::new(),
        gold: ProbeGold {
            session: format!("{id}.txt"),
            domain: None,
            order: 0,
            calibration: false,
            drafted_by: None,
            approved_by: None,
            intentions: intentions
                .iter()
                .map(|(k, s)| ProbeIntention {
                    key: k.to_string(),
                    statement: s.to_string(),
                    established_here: true,
                    evidence_quote: None,
                    notes: None,
                })
                .collect(),
            decisions: decisions
                .iter()
                .map(|(d, serves)| ProbeDecision {
                    statement: d.to_string(),
                    serves: serves.map(str::to_string),
                    evidence_quote: Some(format!("quote for {d}")),
                    made_in: None,
                    notes: None,
                })
                .collect(),
            entities: Vec::new(),
            non_decisions: Vec::new(),
            unsupported_claims: Vec::new(),
        },
    }
}

#[test]
fn roster_dedups_shared_keys_in_first_occurrence_order() {
    let sessions = vec![
        record("s1", &[("k1", "goal one")], &[]),
        record(
            "s2",
            &[("k1", "goal one restated"), ("k2", "goal two")],
            &[],
        ),
    ];
    let roster = intention_roster(&sessions);
    assert_eq!(roster.len(), 2);
    assert_eq!(roster[0], ("k1".to_string(), "goal one".to_string()));
    assert_eq!(roster[1].0, "k2");
}

#[test]
fn pairs_are_exhaustive_with_exactly_one_gold_per_attached_decision() {
    let sessions = vec![record(
        "s1",
        &[("k1", "goal one"), ("k2", "goal two"), ("k3", "goal three")],
        &[("attached decision", Some("k1")), ("orphan decision", None)],
    )];
    let pairs = attachment_pairs(&sessions);
    assert_eq!(pairs.len(), 6); // 2 decisions × 3 roster intentions
    let attached: Vec<_> = pairs.iter().filter(|p| p.decision_index == 0).collect();
    assert_eq!(attached.iter().filter(|p| p.is_gold).count(), 1);
    assert!(
        attached
            .iter()
            .any(|p| p.intention_key == "k3" && !p.is_gold)
    );
    let orphan: Vec<_> = pairs.iter().filter(|p| p.decision_index == 1).collect();
    assert_eq!(orphan.len(), 3);
    assert!(orphan.iter().all(|p| !p.is_gold));
    // Pairs carry the evidence span — P1-b judges spans, not just statements.
    assert_eq!(
        attached[0].decision_evidence.as_deref(),
        Some("quote for attached decision")
    );
}

#[test]
fn known_attachment_scoring_counts_abstention_correctness() {
    let choices = vec![
        (Some("k1".to_string()), Some("k1".to_string())), // correct
        (Some("k2".to_string()), Some("k1".to_string())), // wrong intention
        (None, None),                                     // correct abstention
        (None, Some("k1".to_string())),                   // missed attachment
        (Some("k1".to_string()), None),                   // attached a none-fits case
    ];
    let score = score_known_attachment(&choices);
    assert!((score - 0.4).abs() < 1e-6);
    assert!((score_known_attachment(&[]) - 0.0).abs() < 1e-6);
}
