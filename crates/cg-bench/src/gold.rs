//! Gold answer-key format: session-level labels drafted from the ORIGINAL session files (never
//! through cg) and owner-approved. Session-level by design — chunk-aligned labels are forbidden
//! (the importer fix changes chunk boundaries and would invalidate them).
//!
//! The format is omission-capable by construction: gold lists what SHOULD have been extracted,
//! independent of anything a model emitted — the surface `cg fix` does not have.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldSession {
    pub session: String,
    pub drafted_by: String,
    pub approved_by: Option<String>,
    pub intentions: Vec<GoldIntention>,
    pub decisions: Vec<GoldDecision>,
    pub entities: Vec<GoldEntity>,
    #[serde(default)]
    pub observations: Vec<GoldObservation>,
    #[serde(default)]
    pub outcomes: Vec<GoldOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldIntention {
    pub key: String,
    pub statement: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldDecision {
    pub statement: String,
    /// Key of the intention this decision serves (the trunk attachment).
    pub serves: String,
    /// Where the decisive evidence lives in the source transcript.
    pub made_in: MadeIn,
    #[serde(default)]
    pub evidence_quote: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MadeIn {
    Text,
    ToolOutput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldObservation {
    pub statement: String,
    pub about: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldOutcome {
    pub decision: String,
    pub success: bool,
    pub statement: String,
}

/// Parse a gold session from its YAML form.
pub fn load_gold_str(_yaml: &str) -> Result<GoldSession, String> {
    todo!("cg-correctness C1: gold format parsing not implemented yet")
}

/// Render a gold session back to YAML (roundtrip surface for draft → review → approve).
pub fn to_yaml(_gold: &GoldSession) -> Result<String, String> {
    todo!("cg-correctness C1: gold format rendering not implemented yet")
}

/// An omission training pair: the chunk input plus the COMPLETE corrected harvest — the artifact
/// the correction data engine cannot produce from `cg fix` value-edits alone (criterion C8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OmissionTrainingPair {
    pub chunk_input: String,
    pub corrected_harvest: serde_json::Value,
}

pub fn omission_pair_to_json(_pair: &OmissionTrainingPair) -> Result<String, String> {
    todo!("cg-correctness C8: omission training pair rendering not implemented yet")
}

pub fn omission_pair_from_json(_json: &str) -> Result<OmissionTrainingPair, String> {
    todo!("cg-correctness C8: omission training pair parsing not implemented yet")
}
