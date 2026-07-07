//! Corpus loading: the authored non-developer fixture corpus (plain `User:`/`Assistant:`
//! transcripts + probe gold keys carrying authored negatives) and the dev gold corpus (Claude
//! session jsonl rendered through the PRODUCTION import parser, gold via cg-bench's loader).
//!
//! Probe gold is a superset of the dev gold shape: `serves` is optional (a decision with no
//! serving intention in the roster is an abstention case), and two authored-negative sections
//! (`non_decisions`, `unsupported_claims`) make precision and basis-support graded against
//! explicit gold. Dev gold keys are session-local (`i1`, `i2`, …), so the dev conversion
//! qualifies them with the session id; authored-corpus keys are corpus-global by design.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    User,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnKind {
    Dialogue,
    /// A production-rendered tool-event line ("Assistant used X on Y. It returned: …").
    Tool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub speaker: Speaker,
    pub kind: TurnKind,
    /// Dialogue: the text WITHOUT the `User: `/`Assistant: ` marker. Tool: the full line(s).
    pub text: String,
}

impl Turn {
    /// The turn as transcript text: dialogue re-gains its speaker marker, tool lines are verbatim.
    pub fn rendered(&self) -> String {
        todo!("probe impl commit")
    }
}

/// Parse a plain-format transcript: `User: `/`Assistant: ` start dialogue turns,
/// `Assistant used ` starts a tool turn (the production import rendering), any other line
/// continues the current turn. Text before the first marker is ignored.
pub fn parse_plain_transcript(text: &str) -> Vec<Turn> {
    let _ = text;
    todo!("probe impl commit")
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProbeIntention {
    pub key: String,
    pub statement: String,
    /// False when the intention pre-exists this session (established by an earlier session).
    /// Dev gold has no such field: a listed intention defaults to established-in-session.
    #[serde(default = "default_true")]
    pub established_here: bool,
    #[serde(default)]
    pub evidence_quote: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProbeDecision {
    pub statement: String,
    /// None = no serving intention in the roster (correct attachment behavior is abstention).
    #[serde(default)]
    pub serves: Option<String>,
    #[serde(default)]
    pub evidence_quote: Option<String>,
    #[serde(default)]
    pub made_in: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProbeEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

/// A verbatim span that must NOT be judged a durable decision (rejected option, deferral,
/// observation, plausible discussion).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NonDecision {
    pub quote: String,
    #[serde(default)]
    pub why: Option<String>,
}

/// A plausible statement with no supporting span — basis-support must reject it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct UnsupportedClaim {
    pub claim: String,
    #[serde(default)]
    pub why: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProbeGold {
    pub session: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub order: u32,
    #[serde(default)]
    pub calibration: bool,
    #[serde(default)]
    pub drafted_by: Option<String>,
    #[serde(default)]
    pub approved_by: Option<String>,
    pub intentions: Vec<ProbeIntention>,
    pub decisions: Vec<ProbeDecision>,
    #[serde(default)]
    pub entities: Vec<ProbeEntity>,
    #[serde(default)]
    pub non_decisions: Vec<NonDecision>,
    #[serde(default)]
    pub unsupported_claims: Vec<UnsupportedClaim>,
}

pub fn parse_probe_gold(yaml: &str) -> Result<ProbeGold, String> {
    let _ = yaml;
    todo!("probe impl commit")
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionRecord {
    pub id: String,
    pub domain: Option<String>,
    pub order: u32,
    pub calibration: bool,
    pub turns: Vec<Turn>,
    pub gold: ProbeGold,
}

/// Load the authored corpus: walk `dir`'s domain subdirectories, pair every `*.txt` transcript
/// with its `*.gold.yaml`, sort by (domain, order). A transcript without a gold key (or the
/// reverse) is an error — the corpus is gold-complete by construction.
pub fn load_plain_corpus(dir: &Path) -> Result<Vec<SessionRecord>, String> {
    let _ = dir;
    todo!("probe impl commit")
}

/// Turns of a dev session, rendered through the PRODUCTION import parser
/// (`ClaudeJsonlParser::parse_items` + `render_import_transcript`) then split back into turns —
/// exactly the text production capture sees.
pub fn dev_session_turns(raw: &[u8]) -> Vec<Turn> {
    let _ = raw;
    todo!("probe impl commit")
}

/// Earliest `timestamp` field value across the session's jsonl lines (RFC3339 strings order
/// lexicographically) — the dev corpus chronology key.
pub fn dev_session_earliest_timestamp(raw: &[u8]) -> Option<String> {
    let _ = raw;
    todo!("probe impl commit")
}

/// Convert a cg-bench dev gold session into probe gold, qualifying the session-local intention
/// keys (`i1` → `<session_id>::i1`) so a corpus-wide roster stays collision-free.
pub fn probe_gold_from_dev(session_id: &str, gold: &cg_bench::gold::GoldSession) -> ProbeGold {
    let _ = (session_id, gold);
    todo!("probe impl commit")
}

/// Load the dev corpus: every `<stem>.yaml` in `gold_dir` pairs with the unique
/// `<stem>-*.jsonl` in `transcripts_dir`; sessions are ordered chronologically by
/// `dev_session_earliest_timestamp` and `order` is the chronological rank.
pub fn load_dev_corpus(
    gold_dir: &Path,
    transcripts_dir: &Path,
) -> Result<Vec<SessionRecord>, String> {
    let _ = (gold_dir, transcripts_dir);
    todo!("probe impl commit")
}
