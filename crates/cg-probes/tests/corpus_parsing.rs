//! RED tests — corpus loading contracts: plain-transcript parsing (the authored corpus and the
//! production import rendering share one format), probe gold parsing (superset of dev gold),
//! dev gold conversion, and corpus directory loading.

use cg_probes::corpus::{
    Speaker, TurnKind, dev_session_earliest_timestamp, dev_session_turns, load_plain_corpus,
    parse_plain_transcript, parse_probe_gold, probe_gold_from_dev,
};

#[test]
fn plain_transcript_parses_speakers_and_continuation_lines() {
    let text = "User: line one\ncontinues here\nAssistant: reply\nUser: again\n";
    let turns = parse_plain_transcript(text);
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[0].speaker, Speaker::User);
    assert_eq!(turns[0].kind, TurnKind::Dialogue);
    assert_eq!(turns[0].text, "line one\ncontinues here");
    assert_eq!(turns[1].speaker, Speaker::Assistant);
    assert_eq!(turns[1].text, "reply");
    assert_eq!(turns[2].speaker, Speaker::User);
    assert_eq!(turns[2].text, "again");
}

#[test]
fn production_tool_lines_become_tool_turns_and_render_verbatim() {
    let text = "User: check the parser\nAssistant used Read on /tmp/x.rs. It returned: fn main() {}\nAssistant: it looks fine\n";
    let turns = parse_plain_transcript(text);
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[1].speaker, Speaker::Assistant);
    assert_eq!(turns[1].kind, TurnKind::Tool);
    assert_eq!(
        turns[1].text,
        "Assistant used Read on /tmp/x.rs. It returned: fn main() {}"
    );
    // Rendering round-trips: dialogue re-gains its marker, tool lines stay verbatim.
    assert_eq!(turns[0].rendered(), "User: check the parser");
    assert_eq!(
        turns[1].rendered(),
        "Assistant used Read on /tmp/x.rs. It returned: fn main() {}"
    );
    assert_eq!(turns[2].rendered(), "Assistant: it looks fine");
}

#[test]
fn text_before_the_first_marker_is_ignored() {
    let turns = parse_plain_transcript("stray preamble\nUser: hello\n");
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].text, "hello");
}

const PROBE_GOLD: &str = r#"
session: x.txt
domain: farm-security
order: 2
calibration: false
drafted_by: fable
approved_by: "owner"
shapes: [some-shape-tag]
intentions:
  - key: farm.pickup-safety
    statement: "Keep the children safe."
    established_here: false
decisions:
  - statement: "Do the thing."
    serves: farm.pickup-safety
    evidence_quote: "do it"
  - statement: "Unattached operational call."
    serves: null
    evidence_quote: "op"
entities:
  - name: "Kumar"
    type: PERSON
non_decisions:
  - quote: "maybe we could"
    why: "discussion"
unsupported_claims:
  - claim: "floodlights installed"
"#;

#[test]
fn probe_gold_parses_superset_fields() {
    let gold = parse_probe_gold(PROBE_GOLD).expect("probe gold parses");
    assert_eq!(gold.domain.as_deref(), Some("farm-security"));
    assert_eq!(gold.order, 2);
    assert!(!gold.calibration);
    assert_eq!(gold.intentions.len(), 1);
    assert!(!gold.intentions[0].established_here);
    assert_eq!(gold.decisions.len(), 2);
    assert_eq!(
        gold.decisions[0].serves.as_deref(),
        Some("farm.pickup-safety")
    );
    assert_eq!(gold.decisions[1].serves, None);
    assert_eq!(gold.non_decisions.len(), 1);
    assert_eq!(gold.non_decisions[0].quote, "maybe we could");
    assert_eq!(gold.unsupported_claims.len(), 1);
    assert_eq!(gold.unsupported_claims[0].claim, "floodlights installed");
}

const DEV_SHAPED_GOLD: &str = r#"
session: 640408a7.jsonl
drafted_by: fable-agent
approved_by: "owner"
intentions:
  - key: i1
    statement: "Verify the loop closes."
decisions:
  - statement: "The gate is a blocker."
    serves: i1
    made_in: tool_output
    evidence_quote: "never calls try_promote_skill"
entities:
  - name: "context-graph"
    type: REPOSITORY
    aliases: ["cg"]
observations:
  - statement: "an observation"
    about: "context-graph"
outcomes:
  - decision: "The gate is a blocker."
    success: true
    statement: "it landed"
"#;

#[test]
fn dev_shaped_gold_parses_with_defaults_and_converts_with_qualified_keys() {
    // The dev format (no established_here, no negatives, observations/outcomes present) parses
    // as probe gold with defaults…
    let direct = parse_probe_gold(DEV_SHAPED_GOLD).expect("dev-shaped gold parses");
    assert!(direct.intentions[0].established_here);
    assert!(direct.non_decisions.is_empty());
    assert!(direct.unsupported_claims.is_empty());

    // …and the cg-bench loader path qualifies session-local keys so a corpus roster is
    // collision-free across the 18 dev sessions.
    let bench = cg_bench::gold::load_gold_str(DEV_SHAPED_GOLD).expect("cg-bench gold parses");
    let converted = probe_gold_from_dev("640408a7", &bench);
    assert_eq!(converted.intentions[0].key, "640408a7::i1");
    assert_eq!(
        converted.decisions[0].serves.as_deref(),
        Some("640408a7::i1")
    );
    assert_eq!(
        converted.decisions[0].evidence_quote.as_deref(),
        Some("never calls try_promote_skill")
    );
}

#[test]
fn plain_corpus_loads_pairs_and_sorts_by_domain_then_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let domain = dir.path().join("zeta-domain");
    std::fs::create_dir_all(&domain).expect("mkdir");
    std::fs::write(domain.join("s2-later.txt"), "User: second session\n").expect("write");
    std::fs::write(
        domain.join("s2-later.gold.yaml"),
        "session: s2-later.txt\ndomain: zeta-domain\norder: 2\nintentions: []\ndecisions: []\n",
    )
    .expect("write");
    std::fs::write(domain.join("s1-first.txt"), "User: first session\n").expect("write");
    std::fs::write(
        domain.join("s1-first.gold.yaml"),
        "session: s1-first.txt\ndomain: zeta-domain\norder: 1\ncalibration: true\nintentions: []\ndecisions: []\n",
    )
    .expect("write");
    let alpha = dir.path().join("alpha-domain");
    std::fs::create_dir_all(&alpha).expect("mkdir");
    std::fs::write(alpha.join("s1-only.txt"), "User: alpha session\n").expect("write");
    std::fs::write(
        alpha.join("s1-only.gold.yaml"),
        "session: s1-only.txt\ndomain: alpha-domain\norder: 1\nintentions: []\ndecisions: []\n",
    )
    .expect("write");

    let corpus = load_plain_corpus(dir.path()).expect("corpus loads");
    assert_eq!(corpus.len(), 3);
    assert_eq!(corpus[0].domain.as_deref(), Some("alpha-domain"));
    assert_eq!(corpus[1].id, "s1-first");
    assert!(corpus[1].calibration);
    assert_eq!(corpus[2].id, "s2-later");
    assert_eq!(corpus[2].order, 2);
    assert_eq!(corpus[0].turns.len(), 1);
    assert_eq!(corpus[0].turns[0].text, "alpha session");
}

#[test]
fn plain_corpus_refuses_a_transcript_without_gold() {
    let dir = tempfile::tempdir().expect("tempdir");
    let domain = dir.path().join("d");
    std::fs::create_dir_all(&domain).expect("mkdir");
    std::fs::write(domain.join("orphan.txt"), "User: no gold\n").expect("write");
    assert!(load_plain_corpus(dir.path()).is_err());
}

#[test]
fn dev_session_turns_render_through_the_production_parser() {
    let raw = concat!(
        r#"{"type":"user","timestamp":"2026-06-02T09:00:00Z","message":{"role":"user","content":"we should use the relay hub"}}"#,
        "\n",
        r#"{"type":"assistant","timestamp":"2026-06-02T09:00:05Z","message":{"role":"assistant","content":[{"type":"text","text":"agreed, the relay hub it is"}]}}"#,
        "\n"
    );
    let turns = dev_session_turns(raw.as_bytes());
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].speaker, Speaker::User);
    assert_eq!(turns[0].text, "we should use the relay hub");
    assert_eq!(turns[1].speaker, Speaker::Assistant);
    assert_eq!(turns[1].text, "agreed, the relay hub it is");
}

#[test]
fn dev_session_chronology_key_is_the_earliest_timestamp() {
    let raw = concat!(
        r#"{"leafUuid":"x","sessionId":"y","type":"summary"}"#,
        "\n",
        r#"{"type":"user","timestamp":"2026-06-02T10:30:00Z","message":{"role":"user","content":"later line"}}"#,
        "\n",
        r#"{"type":"assistant","timestamp":"2026-06-02T09:15:00Z","message":{"role":"assistant","content":"earlier line"}}"#,
        "\n"
    );
    assert_eq!(
        dev_session_earliest_timestamp(raw.as_bytes()).as_deref(),
        Some("2026-06-02T09:15:00Z")
    );
    assert_eq!(dev_session_earliest_timestamp(b"not json\n"), None);
}
