//! RED tests — P1 item construction: oracle packets, decision-vs-discussion judgment items
//! (positives from gold spans with structural turn context; negatives from authored
//! non-decisions AND every turn without a gold span — exhaustive, never sampled), and
//! basis-support items (supported claims vs the oracle packet; unsupported from authored
//! negatives, or cross-session claims when a session has none).

use cg_probes::corpus::{
    NonDecision, ProbeDecision, ProbeGold, ProbeIntention, SessionRecord, UnsupportedClaim,
    parse_plain_transcript,
};
use cg_probes::items::{basis_items, decision_judgment_items, judgment_premise, oracle_packet};

fn decision(statement: &str, serves: Option<&str>, quote: &str) -> ProbeDecision {
    ProbeDecision {
        statement: statement.to_string(),
        serves: serves.map(str::to_string),
        evidence_quote: Some(quote.to_string()),
        made_in: None,
        notes: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn session(
    id: &str,
    order: u32,
    calibration: bool,
    transcript: &str,
    decisions: Vec<ProbeDecision>,
    non_decisions: Vec<NonDecision>,
    unsupported: Vec<UnsupportedClaim>,
) -> SessionRecord {
    SessionRecord {
        id: id.to_string(),
        domain: Some("d".to_string()),
        order,
        calibration,
        turns: parse_plain_transcript(transcript),
        gold: ProbeGold {
            session: format!("{id}.txt"),
            domain: Some("d".to_string()),
            order,
            calibration,
            drafted_by: None,
            approved_by: None,
            intentions: vec![ProbeIntention {
                key: "k1".to_string(),
                statement: "the goal".to_string(),
                established_here: true,
                evidence_quote: None,
                notes: None,
            }],
            decisions,
            entities: Vec::new(),
            non_decisions,
            unsupported_claims: unsupported,
        },
    }
}

const TRANSCRIPT: &str = "User: opening chatter about the weather\nAssistant: what should we do about the gate rule\nUser: lock the gate every night from now on\nAssistant: noted\n";

#[test]
fn judgment_premise_is_turn_plus_predecessor() {
    let turns = parse_plain_transcript(TRANSCRIPT);
    // Turn 2 carries the decision; its premise includes turn 1 (the antecedent) — a structural,
    // content-blind rule, uniform for positives and negatives.
    let premise = judgment_premise(&turns, 2);
    assert_eq!(
        premise,
        "Assistant: what should we do about the gate rule\nUser: lock the gate every night from now on"
    );
    // The first turn has no predecessor.
    assert_eq!(
        judgment_premise(&turns, 0),
        "User: opening chatter about the weather"
    );
}

#[test]
fn decision_items_are_exhaustive_and_labelled() {
    let s = session(
        "s1",
        1,
        false,
        TRANSCRIPT,
        vec![decision(
            "Lock the gate nightly.",
            Some("k1"),
            "lock the gate every night",
        )],
        vec![NonDecision {
            quote: "what should we do about the gate rule".to_string(),
            why: None,
        }],
        Vec::new(),
    );
    let items = decision_judgment_items(&[s]);
    // 1 positive (the gold span, premised on its turn + predecessor)…
    let positives: Vec<_> = items.iter().filter(|i| i.positive).collect();
    assert_eq!(positives.len(), 1);
    assert!(positives[0].premise.contains("lock the gate every night"));
    assert!(
        positives[0]
            .premise
            .contains("what should we do about the gate rule")
    );
    // …negatives: the authored non-decision + every turn with no gold span (turns 0, 1, 3 —
    // turn 1 is also the authored negative's turn and is not double-counted).
    let negatives: Vec<_> = items.iter().filter(|i| !i.positive).collect();
    assert_eq!(negatives.len(), 3);
    assert!(
        negatives
            .iter()
            .any(|n| n.premise.contains("opening chatter"))
    );
    assert!(negatives.iter().any(|n| n.premise.contains("noted")));
    // Every item carries the session + calibration tag for the split.
    assert!(items.iter().all(|i| i.session_id == "s1" && !i.calibration));
}

#[test]
fn oracle_packet_joins_gold_spans() {
    let s = session(
        "s1",
        1,
        false,
        TRANSCRIPT,
        vec![
            decision("A.", Some("k1"), "lock the gate every night"),
            decision("B.", None, "noted"),
        ],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(oracle_packet(&s), "lock the gate every night\nnoted");
}

#[test]
fn basis_items_use_authored_negatives_or_cross_session_claims() {
    let s1 = session(
        "s1",
        1,
        false,
        TRANSCRIPT,
        vec![decision(
            "Lock the gate nightly.",
            Some("k1"),
            "lock the gate every night",
        )],
        Vec::new(),
        vec![UnsupportedClaim {
            claim: "The gate was replaced.".to_string(),
            why: None,
        }],
    );
    let s2 = session(
        "s2",
        2,
        false,
        "User: paint the fence before winter\n",
        vec![decision(
            "Paint the fence before winter.",
            Some("k1"),
            "paint the fence before winter",
        )],
        Vec::new(),
        Vec::new(), // no authored negatives → next session's claims stand in
    );
    let items = basis_items(&[s1, s2]);
    // s1 supported: its decision statement vs its packet; s1 unsupported: the authored claim.
    assert!(
        items.iter().any(|i| {
            i.session_id == "s1" && i.supported && i.claim == "Lock the gate nightly."
        })
    );
    assert!(
        items
            .iter()
            .any(|i| i.session_id == "s1" && !i.supported && i.claim == "The gate was replaced.")
    );
    // s2 has no authored unsupported claims: the chronologically NEXT session's (wrapping to
    // s1) decision statements are its unsupported claims, judged against s2's packet.
    let s2_unsupported: Vec<_> = items
        .iter()
        .filter(|i| i.session_id == "s2" && !i.supported)
        .collect();
    assert_eq!(s2_unsupported.len(), 1);
    assert_eq!(s2_unsupported[0].claim, "Lock the gate nightly.");
    assert_eq!(s2_unsupported[0].doc, "paint the fence before winter");
}
