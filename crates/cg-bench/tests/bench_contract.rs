//! RED contract tests for criteria C1/C8 (criteria commit 99d99ee — the criteria→test mapping
//! in the run working doc names these). Authored and committed BEFORE implementation; the
//! implementation commit may not touch this file.

use cg_bench::gold::{
    MadeIn, OmissionTrainingPair, load_gold_str, omission_pair_from_json, omission_pair_to_json,
    to_yaml,
};
use cg_bench::scoring::{
    AUTO_MATCH_THRESHOLD, AUTO_NON_MATCH_THRESHOLD, ExtractedDecision, ExtractedEntity,
    ExtractedSession, MatchBand, SimilarityProvider, band, score_session,
};

/// Deterministic similarity double: identical strings are 1.0; listed pairs get their listed
/// score; everything else 0.0. This tests the banding/aggregation logic (the deterministic part
/// of the bench); it never stands in for model quality.
struct TableSim {
    pairs: Vec<(&'static str, &'static str, f32)>,
}

impl SimilarityProvider for TableSim {
    fn similarity(&self, a: &str, b: &str) -> f32 {
        if a == b {
            return 1.0;
        }
        self.pairs
            .iter()
            .find(|(x, y, _)| (*x == a && *y == b) || (*x == b && *y == a))
            .map(|(_, _, s)| *s)
            .unwrap_or(0.0)
    }
}

const GOLD_YAML: &str = r#"
session: test-session.jsonl
drafted_by: fable-agent 2026-07-03
approved_by: null
intentions:
  - key: i1
    statement: "Route skill promotion through the held-out gate so bad rules never auto-apply"
decisions:
  - statement: "Production promotion must route through the held-out gate, not a count threshold"
    serves: i1
    made_in: tool_output
    evidence_quote: "never calls try_promote_skill"
  - statement: "Green unit tests on orphaned code paths do not prove completion"
    serves: i1
    made_in: text
entities:
  - name: "skill_store.rs"
    type: FILE
    aliases: ["crates/context-graph/src/skill_store.rs"]
  - name: "context-graph"
    type: REPOSITORY
observations:
  - statement: "The held-out gate has zero production callers"
    about: "skill_store.rs"
outcomes:
  - decision: "Production promotion must route through the held-out gate, not a count threshold"
    success: false
    statement: "As of the session the gate remains orphaned"
"#;

#[test]
fn gold_format_roundtrips_and_records_omissions() {
    let gold = load_gold_str(GOLD_YAML).expect("gold YAML parses");
    assert_eq!(gold.session, "test-session.jsonl");
    assert_eq!(gold.intentions.len(), 1);
    assert_eq!(gold.decisions.len(), 2);
    assert_eq!(gold.decisions[0].made_in, MadeIn::ToolOutput);
    assert_eq!(gold.decisions[0].serves, "i1");
    assert_eq!(gold.entities[0].aliases.len(), 1);
    assert_eq!(gold.observations.len(), 1);
    assert_eq!(gold.outcomes.len(), 1);

    // Roundtrip: render → parse → same content.
    let rendered = to_yaml(&gold).expect("gold renders");
    let reparsed = load_gold_str(&rendered).expect("rendered gold reparses");
    assert_eq!(gold, reparsed);

    // Omission recording: an extraction that misses a gold decision lists it by statement —
    // the record `cg fix` structurally cannot produce.
    let extracted = ExtractedSession {
        intentions: vec![
            "Route skill promotion through the held-out gate so bad rules never auto-apply".into(),
        ],
        decisions: vec![ExtractedDecision {
            statement:
                "Production promotion must route through the held-out gate, not a count threshold"
                    .into(),
            goal: Some(
                "Route skill promotion through the held-out gate so bad rules never auto-apply"
                    .into(),
            ),
        }],
        entities: vec![],
        observations: vec![],
    };
    let score = score_session(&gold, &extracted, &TableSim { pairs: vec![] });
    assert_eq!(score.decisions.matched, 1);
    assert_eq!(score.decisions.gold_total, 2);
    assert_eq!(
        score.decisions.missed,
        vec!["Green unit tests on orphaned code paths do not prove completion".to_string()]
    );
    // Entities missed entirely → both recorded.
    assert_eq!(score.entities.matched, 0);
    assert_eq!(score.entities.missed.len(), 2);
}

#[test]
fn scoring_matcher_applies_frozen_thresholds() {
    // The registration itself: these values are frozen in docs/cg-correctness-criteria.md.
    assert_eq!(AUTO_MATCH_THRESHOLD, 0.75);
    assert_eq!(AUTO_NON_MATCH_THRESHOLD, 0.55);

    assert_eq!(band(0.80), MatchBand::Match);
    assert_eq!(band(0.75), MatchBand::Match); // inclusive: "≥ 0.75 auto-match"
    assert_eq!(band(0.74), MatchBand::GrayZone);
    assert_eq!(band(0.55), MatchBand::GrayZone); // "0.55–0.75 gray zone"
    assert_eq!(band(0.54), MatchBand::NonMatch); // "< 0.55 auto-non-match"
    assert_eq!(band(0.0), MatchBand::NonMatch);
}

#[test]
fn scoring_routes_gray_zone_out_instead_of_deciding() {
    let gold = load_gold_str(GOLD_YAML).expect("gold YAML parses");
    // One extracted decision sits in the gray zone against gold decision 1: neither matched
    // nor missed silently — routed to adjudication.
    let extracted = ExtractedSession {
        intentions: vec![],
        decisions: vec![ExtractedDecision {
            statement: "Promotion should use the held-out gate".into(),
            goal: None,
        }],
        entities: vec![],
        observations: vec![],
    };
    let sim = TableSim {
        pairs: vec![(
            "Promotion should use the held-out gate",
            "Production promotion must route through the held-out gate, not a count threshold",
            0.65,
        )],
    };
    let score = score_session(&gold, &extracted, &sim);
    assert_eq!(score.decisions.matched, 0, "gray zone is not an auto-match");
    assert!(
        !score.decisions.missed.contains(
            &"Production promotion must route through the held-out gate, not a count threshold"
                .to_string()
        ),
        "gray zone is not an auto-miss either"
    );
    assert_eq!(score.gray_zone.len(), 1);
    assert_eq!(
        score.gray_zone[0],
        (
            "Promotion should use the held-out gate".to_string(),
            "Production promotion must route through the held-out gate, not a count threshold"
                .to_string()
        )
    );
}

#[test]
fn scoring_dedupes_cross_chunk_duplicates_before_recall() {
    let gold = load_gold_str(GOLD_YAML).expect("gold YAML parses");
    // The same decision extracted from two chunks in variant phrasings (the probe-1 observed
    // failure): recall counts ONE hit, precision counts the group ONCE, duplication reported.
    let extracted = ExtractedSession {
        intentions: vec![],
        decisions: vec![
            ExtractedDecision {
                statement:
                    "Production promotion must route through the held-out gate, not a count threshold"
                        .into(),
                goal: None,
            },
            ExtractedDecision {
                statement: "Promotion routes through the held-out gate rather than a threshold"
                    .into(),
                goal: None,
            },
        ],
        entities: vec![],
        observations: vec![],
    };
    let sim = TableSim {
        pairs: vec![(
            "Promotion routes through the held-out gate rather than a threshold",
            "Production promotion must route through the held-out gate, not a count threshold",
            0.90,
        )],
    };
    let score = score_session(&gold, &extracted, &sim);
    assert_eq!(score.decisions.matched, 1, "one gold decision hit, not two");
    assert_eq!(
        score.decision_groups, 1,
        "two variants collapse to one group"
    );
    assert_eq!(score.decision_noise_groups, 0);
    assert!(
        (score.duplication_rate - 0.5).abs() < f32::EPSILON,
        "1 of 2 items was a duplicate"
    );
}

#[test]
fn attachment_scored_only_over_matched_decisions() {
    let gold = load_gold_str(GOLD_YAML).expect("gold YAML parses");
    let extracted = ExtractedSession {
        intentions: vec![
            "Route skill promotion through the held-out gate so bad rules never auto-apply".into(),
        ],
        decisions: vec![
            // Matched decision attached to the WRONG goal → counted, wrong.
            ExtractedDecision {
                statement:
                    "Production promotion must route through the held-out gate, not a count threshold"
                        .into(),
                goal: Some("Ship the release notes".into()),
            },
            // Unmatched (noise) decision with a goal → NOT counted in attachment at all.
            ExtractedDecision {
                statement: "The sky is blue".into(),
                goal: Some("irrelevant".into()),
            },
        ],
        entities: vec![],
        observations: vec![],
    };
    let score = score_session(&gold, &extracted, &TableSim { pairs: vec![] });
    assert_eq!(
        score.attachment_total, 1,
        "only the matched decision is attachment-eligible"
    );
    assert_eq!(
        score.attachment_correct, 0,
        "wrong-goal attachment is wrong"
    );
    assert_eq!(
        score.decision_noise_groups, 1,
        "the unmatched decision is noise"
    );
}

#[test]
fn intention_scored_at_session_level() {
    let gold = load_gold_str(GOLD_YAML).expect("gold YAML parses");
    // A procedural restatement fails by failing to match the gold intention.
    let procedural = ExtractedSession {
        intentions: vec!["Satisfy the requirements of the brief".into()],
        decisions: vec![],
        entities: vec![],
        observations: vec![],
    };
    let sim = TableSim {
        pairs: vec![(
            "Satisfy the requirements of the brief",
            "Route skill promotion through the held-out gate so bad rules never auto-apply",
            0.30,
        )],
    };
    let score = score_session(&gold, &procedural, &sim);
    assert!(
        !score.intention_captured,
        "a procedural restatement is not the intention"
    );

    let real = ExtractedSession {
        intentions: vec![
            "Route skill promotion through the held-out gate so bad rules never auto-apply".into(),
        ],
        decisions: vec![],
        entities: vec![],
        observations: vec![],
    };
    let score = score_session(&gold, &real, &TableSim { pairs: vec![] });
    assert!(score.intention_captured);
}

#[test]
fn entity_matching_uses_aliases_and_scores_types_over_matched() {
    let gold = load_gold_str(GOLD_YAML).expect("gold YAML parses");
    let extracted = ExtractedSession {
        intentions: vec![],
        decisions: vec![],
        entities: vec![
            // Matches gold "skill_store.rs" via its alias, but with the WRONG type.
            ExtractedEntity {
                name: "crates/context-graph/src/skill_store.rs".into(),
                entity_type: "MODULE".into(),
            },
            // Matches gold "context-graph" exactly, right type.
            ExtractedEntity {
                name: "context-graph".into(),
                entity_type: "REPOSITORY".into(),
            },
        ],
        observations: vec![],
    };
    let score = score_session(&gold, &extracted, &TableSim { pairs: vec![] });
    assert_eq!(score.entities.matched, 2);
    assert_eq!(score.entities.missed.len(), 0);
    assert_eq!(
        score.entity_type_correct, 1,
        "type accuracy is over matched entities"
    );
}

#[test]
fn omission_training_pair_format_roundtrip() {
    let pair = OmissionTrainingPair {
        chunk_input: "the raw chunk text the model saw".into(),
        corrected_harvest: serde_json::json!({
            "goals": [{"statement": "the real goal"}],
            "decisions": [{"statement": "the missed decision", "goal_key": "g1"}],
            "entities": [{"name": "skill_store.rs", "type": "FILE"}],
        }),
    };
    let json = omission_pair_to_json(&pair).expect("pair renders");
    let back = omission_pair_from_json(&json).expect("pair reparses");
    assert_eq!(pair, back);
}
