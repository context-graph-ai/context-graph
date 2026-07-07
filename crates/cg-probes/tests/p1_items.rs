//! RED tests — P1 item construction: oracle packets (the distinct turns CONTAINING the gold
//! spans), decision-vs-discussion judgment items (positives from gold spans with structural
//! turn context; negatives from authored non-decisions AND every turn without a gold span,
//! EXCLUDING turns that statement-match a gold decision — the restatement artifact: an
//! assistant turn restating the user's decision is a dedup event, not a false positive), and
//! basis-support items. Changed 2026-07-07 (own commit, per the test wall): round-1 P1 scoring
//! exposed the restatement artifact and the context-stripped packet; see working-doc D8/D10.

use cg_probes::corpus::{
    NonDecision, ProbeDecision, ProbeGold, ProbeIntention, SessionRecord, UnsupportedClaim,
    parse_plain_transcript,
};
use cg_probes::items::{
    attachment_premise, basis_items, decision_judgment_items, judgment_premise, oracle_packet,
};

/// Content-faithful lexical similarity double (dice coefficient over lowercase alphanumeric
/// tokens): returns what a real embedder-cosine would qualitatively return for restatements vs
/// unrelated text — never keyed to fixture content.
struct LexicalSim;

impl cg_bench::scoring::SimilarityProvider for LexicalSim {
    fn similarity(&self, a: &str, b: &str) -> f32 {
        let tokens = |s: &str| -> Vec<String> {
            s.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        };
        let (ta, tb) = (tokens(a), tokens(b));
        if ta.is_empty() || tb.is_empty() {
            return 0.0;
        }
        let mut pool = tb.clone();
        let mut intersection = 0usize;
        for token in &ta {
            if let Some(pos) = pool.iter().position(|t| t == token) {
                pool.remove(pos);
                intersection += 1;
            }
        }
        (2.0 * intersection as f32) / (ta.len() + tb.len()) as f32
    }
}

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

const TRANSCRIPT: &str = "User: opening chatter about the weather\nAssistant: what should we do about the gate rule\nUser: lock the gate every night from now on\nAssistant: the rule is lock the gate every night\n";

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
            "Lock the gate every night.",
            Some("k1"),
            "lock the gate every night",
        )],
        vec![NonDecision {
            quote: "what should we do about the gate rule".to_string(),
            why: None,
        }],
        Vec::new(),
    );
    let report = decision_judgment_items(&[s], &LexicalSim);
    // 1 positive (the gold span, premised on its turn + predecessor)…
    let positives: Vec<_> = report.items.iter().filter(|i| i.positive).collect();
    assert_eq!(positives.len(), 1);
    assert!(
        positives[0]
            .premise
            .contains("lock the gate every night from now on")
    );
    assert!(
        positives[0]
            .premise
            .contains("what should we do about the gate rule")
    );
    // …negatives: the authored non-decision + turns with no gold span — but turn 3 RESTATES the
    // gold decision (statement-match on the frozen band) and is EXCLUDED, not a negative.
    let negatives: Vec<_> = report.items.iter().filter(|i| !i.positive).collect();
    assert_eq!(negatives.len(), 2);
    assert!(
        negatives
            .iter()
            .any(|n| n.premise.contains("opening chatter"))
    );
    assert!(
        !negatives
            .iter()
            .any(|n| n.premise.contains("the rule is lock"))
    );
    assert_eq!(report.restatement_negatives_excluded, 1);
    // Every item carries the session + calibration tag for the split.
    assert!(
        report
            .items
            .iter()
            .all(|i| i.session_id == "s1" && !i.calibration)
    );
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
            decision("B.", None, "the rule is lock the gate every night"),
        ],
        Vec::new(),
        Vec::new(),
    );
    // The packet is the distinct CONTAINING turns (gold-anchored context), not bare quotes.
    assert_eq!(
        oracle_packet(&s),
        "User: lock the gate every night from now on\nAssistant: the rule is lock the gate every night"
    );
    // A quote found in no turn falls back to the quote itself.
    let mut orphan = s.clone();
    orphan.gold.decisions[0].evidence_quote = Some("a span from nowhere".to_string());
    assert!(oracle_packet(&orphan).contains("a span from nowhere"));
}

#[test]
fn attachment_premise_uses_the_structural_context_rule() {
    let s = session(
        "s1",
        1,
        false,
        TRANSCRIPT,
        vec![decision(
            "Lock the gate nightly.",
            Some("k1"),
            "lock the gate every night from now on",
        )],
        Vec::new(),
        Vec::new(),
    );
    let premise = attachment_premise(&s, 0);
    assert!(premise.contains("lock the gate every night from now on"));
    assert!(premise.contains("what should we do about the gate rule"));
    // Unlocatable evidence falls back to the quote.
    let mut orphan = s.clone();
    orphan.gold.decisions[0].evidence_quote = Some("a span from nowhere".to_string());
    assert_eq!(attachment_premise(&orphan, 0), "a span from nowhere");
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
    assert_eq!(s2_unsupported[0].doc, "User: paint the fence before winter");
}
