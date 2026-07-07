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
        match self.kind {
            TurnKind::Tool => self.text.clone(),
            TurnKind::Dialogue => {
                let marker = match self.speaker {
                    Speaker::User => "User",
                    Speaker::Assistant => "Assistant",
                };
                format!("{marker}: {}", self.text)
            }
        }
    }
}

/// Parse a plain-format transcript: `User: `/`Assistant: ` start dialogue turns,
/// `Assistant used ` starts a tool turn (the production import rendering), any other line
/// continues the current turn. Text before the first marker is ignored.
pub fn parse_plain_transcript(text: &str) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("User:") {
            turns.push(Turn {
                speaker: Speaker::User,
                kind: TurnKind::Dialogue,
                text: rest.trim_start().to_string(),
            });
        } else if line.starts_with("Assistant used ") {
            turns.push(Turn {
                speaker: Speaker::Assistant,
                kind: TurnKind::Tool,
                text: line.to_string(),
            });
        } else if let Some(rest) = line.strip_prefix("Assistant:") {
            turns.push(Turn {
                speaker: Speaker::Assistant,
                kind: TurnKind::Dialogue,
                text: rest.trim_start().to_string(),
            });
        } else if let Some(current) = turns.last_mut() {
            current.text.push('\n');
            current.text.push_str(line);
        }
        // Text before the first marker is ignored.
    }
    turns
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
    serde_yaml_ng::from_str(yaml).map_err(|e| format!("probe gold YAML parse error: {e}"))
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
    let mut records = Vec::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(current) = dirs.pop() {
        let entries =
            std::fs::read_dir(&current).map_err(|e| format!("read_dir {current:?}: {e}"))?;
        for entry in entries {
            let path = entry.map_err(|e| format!("read_dir entry: {e}"))?.path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with(".gold.yaml") {
                let transcript = path.with_file_name(name.replace(".gold.yaml", ".txt"));
                if !transcript.exists() {
                    return Err(format!("gold key without a transcript: {path:?}"));
                }
                continue;
            }
            if !name.ends_with(".txt") {
                continue;
            }
            let gold_path = path.with_file_name(name.replace(".txt", ".gold.yaml"));
            if !gold_path.exists() {
                return Err(format!("transcript without a gold key: {path:?}"));
            }
            let transcript = std::fs::read_to_string(&path)
                .map_err(|e| format!("read transcript {path:?}: {e}"))?;
            let gold_text = std::fs::read_to_string(&gold_path)
                .map_err(|e| format!("read gold {gold_path:?}: {e}"))?;
            let gold = parse_probe_gold(&gold_text).map_err(|e| format!("{gold_path:?}: {e}"))?;
            let id = name.trim_end_matches(".txt").to_string();
            let domain = gold.domain.clone().or_else(|| {
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .map(str::to_string)
            });
            records.push(SessionRecord {
                id,
                domain,
                order: gold.order,
                calibration: gold.calibration,
                turns: parse_plain_transcript(&transcript),
                gold,
            });
        }
    }
    records.sort_by(|a, b| {
        (a.domain.as_deref().unwrap_or(""), a.order, a.id.as_str()).cmp(&(
            b.domain.as_deref().unwrap_or(""),
            b.order,
            b.id.as_str(),
        ))
    });
    Ok(records)
}

/// Turns of a dev session, rendered through the PRODUCTION import parser
/// (`ClaudeJsonlParser::parse_items` + `render_import_transcript`) then split back into turns —
/// exactly the text production capture sees.
pub fn dev_session_turns(raw: &[u8]) -> Vec<Turn> {
    let parser = context_graph::ingest::ClaudeJsonlParser;
    let items = parser.parse_items(raw);
    let rendered = context_graph::ingest::render_import_transcript(&items);
    parse_plain_transcript(&rendered)
}

/// Earliest `timestamp` field value across the session's jsonl lines (RFC3339 strings order
/// lexicographically) — the dev corpus chronology key.
pub fn dev_session_earliest_timestamp(raw: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(raw);
    text.lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()?
                .get("timestamp")?
                .as_str()
                .map(str::to_string)
        })
        .min()
}

/// Convert a cg-bench dev gold session into probe gold, qualifying the session-local intention
/// keys (`i1` → `<session_id>::i1`) so a corpus-wide roster stays collision-free.
pub fn probe_gold_from_dev(session_id: &str, gold: &cg_bench::gold::GoldSession) -> ProbeGold {
    let qualify = |key: &str| format!("{session_id}::{key}");
    ProbeGold {
        session: gold.session.clone(),
        domain: None,
        order: 0,
        calibration: false,
        drafted_by: Some(gold.drafted_by.clone()),
        approved_by: gold.approved_by.clone(),
        intentions: gold
            .intentions
            .iter()
            .map(|i| ProbeIntention {
                key: qualify(&i.key),
                statement: i.statement.clone(),
                established_here: true,
                evidence_quote: None,
                notes: None,
            })
            .collect(),
        decisions: gold
            .decisions
            .iter()
            .map(|d| ProbeDecision {
                statement: d.statement.clone(),
                serves: Some(qualify(&d.serves)),
                evidence_quote: d.evidence_quote.clone(),
                made_in: Some(
                    match d.made_in {
                        cg_bench::gold::MadeIn::Text => "text",
                        cg_bench::gold::MadeIn::ToolOutput => "tool_output",
                    }
                    .to_string(),
                ),
                notes: d.notes.clone(),
            })
            .collect(),
        entities: gold
            .entities
            .iter()
            .map(|e| ProbeEntity {
                name: e.name.clone(),
                entity_type: e.entity_type.clone(),
                aliases: e.aliases.clone(),
                notes: None,
            })
            .collect(),
        non_decisions: Vec::new(),
        unsupported_claims: Vec::new(),
    }
}

/// Load the dev corpus: every `<stem>.yaml` in `gold_dir` pairs with the unique
/// `<stem>-*.jsonl` in `transcripts_dir`; sessions are ordered chronologically by
/// `dev_session_earliest_timestamp` and `order` is the chronological rank.
pub fn load_dev_corpus(
    gold_dir: &Path,
    transcripts_dir: &Path,
) -> Result<Vec<SessionRecord>, String> {
    let mut loaded: Vec<(Option<String>, SessionRecord)> = Vec::new();
    let entries = std::fs::read_dir(gold_dir).map_err(|e| format!("read_dir {gold_dir:?}: {e}"))?;
    for entry in entries {
        let path = entry.map_err(|e| format!("read_dir entry: {e}"))?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.ends_with(".yaml") || name.ends_with(".gold.yaml") {
            continue;
        }
        let stem = name.trim_end_matches(".yaml").to_string();
        let prefix = format!("{stem}-");
        let mut matches: Vec<_> = std::fs::read_dir(transcripts_dir)
            .map_err(|e| format!("read_dir {transcripts_dir:?}: {e}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".jsonl"))
            })
            .collect();
        if matches.len() != 1 {
            return Err(format!(
                "gold {stem}: expected exactly one transcript {prefix}*.jsonl in \
                 {transcripts_dir:?}, found {}",
                matches.len()
            ));
        }
        let transcript_path = matches.remove(0);
        let raw = std::fs::read(&transcript_path)
            .map_err(|e| format!("read transcript {transcript_path:?}: {e}"))?;
        let gold_text =
            std::fs::read_to_string(&path).map_err(|e| format!("read gold {path:?}: {e}"))?;
        let bench_gold =
            cg_bench::gold::load_gold_str(&gold_text).map_err(|e| format!("{path:?}: {e}"))?;
        loaded.push((
            dev_session_earliest_timestamp(&raw),
            SessionRecord {
                id: stem.clone(),
                domain: None,
                order: 0,
                calibration: false,
                turns: dev_session_turns(&raw),
                gold: probe_gold_from_dev(&stem, &bench_gold),
            },
        ));
    }
    // Chronological: earliest in-file timestamp, id as a stable tiebreak; sessions with no
    // timestamp sort last.
    loaded.sort_by(|a, b| match (&a.0, &b.0) {
        (Some(ta), Some(tb)) => ta.cmp(tb).then_with(|| a.1.id.cmp(&b.1.id)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.1.id.cmp(&b.1.id),
    });
    Ok(loaded
        .into_iter()
        .enumerate()
        .map(|(rank, (_, mut record))| {
            record.order = rank as u32 + 1;
            record
        })
        .collect())
}
