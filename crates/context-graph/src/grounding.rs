//! Placeholder grounding predicate. Flags on low model confidence rather than span absence.
//!
//! Correct behavior: flag if cited_span is None OR span text not in chunk OR value not supported
//! by span OR value out-of-vocab OR ambiguous span. The model's own confidence field is NEVER used.

#[derive(Debug, Clone)]
pub struct GroundingInput {
    pub value: String,
    pub cited_span: Option<String>,
    pub chunk: String,
    pub model_confidence: f32,
    pub vocab: Option<Vec<String>>,
    /// Optional: the label the span is meant to support (for mislabel / ambiguity checks).
    pub label: Option<String>,
    /// Optional: competing labels a span might equally support (for ambiguity check).
    pub competing_labels: Vec<String>,
}

impl Default for GroundingInput {
    fn default() -> Self {
        Self {
            value: String::new(),
            cited_span: None,
            chunk: String::new(),
            model_confidence: 1.0,
            vocab: None,
            label: None,
            competing_labels: vec![],
        }
    }
}

/// Placeholder: uses model confidence instead of checking span presence in transcript.
/// Correct behavior: flag if cited_span is None OR cited_span text is not found in chunk,
/// or if value is out-of-vocab when vocab is given.
pub fn is_grounded(input: &GroundingInput) -> bool {
    // Placeholder: uses confidence threshold instead of span-presence check.
    input.model_confidence > 0.5
}

pub fn should_flag(input: &GroundingInput) -> bool {
    !is_grounded(input)
}

/// Check whether a span ambiguously supports multiple competing labels.
/// Placeholder: always returns false (never flags ambiguous spans).
pub fn is_ambiguous_span(input: &GroundingInput) -> bool {
    let _ = &input.competing_labels;
    false // Placeholder: should return true when competing_labels.len() > 1 and no disambiguator.
}

/// Full grounding check including span presence, vocab check, and ambiguity.
/// Placeholder: only checks confidence (same as is_grounded) — misses span, vocab, and ambiguity.
pub fn full_grounding_check(input: &GroundingInput) -> GroundingVerdict {
    if input.model_confidence <= 0.5 {
        GroundingVerdict::Flagged(FlagReason::LowConfidence) // Placeholder reason
    } else if input.competing_labels.len() > 1 {
        // Placeholder: should flag as ambiguous
        GroundingVerdict::Grounded
    } else {
        GroundingVerdict::Grounded
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroundingVerdict {
    Grounded,
    Flagged(FlagReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlagReason {
    MissingSpan,
    SpanNotInChunk,
    SpanDoesNotSupportLabel,
    OutOfVocab,
    AmbiguousSpan,
    EmptyCandidateSet,
    /// Placeholder: uses low confidence as a reason — correct impl never uses this.
    LowConfidence,
}

/// Placeholder: returns empty instead of the legal EntityType variant names.
pub fn legal_entity_types() -> Vec<String> {
    // Placeholder: returns empty. Correct: return all 13 EntityType names.
    vec![]
}

/// Return the legal entity type names for display when a closed-vocab item is flagged.
/// Placeholder: returns empty list — test asserts the legal list is enumerated.
pub fn legal_entity_type_names() -> Vec<String> {
    // Placeholder: should return all 13 names from EntityType.
    // Correct impl: EntityType::ALL.iter().map(|t| t.to_string()).collect()
    vec![]
}

/// Check whether a flagged closed-vocab item can expose its legal value set.
/// Placeholder: returns empty — test asserts the legal set is non-empty.
pub fn why_flagged_closed_vocab(_flagged_value: &str) -> Vec<String> {
    // Placeholder: should return the full legal vocabulary for the field.
    vec![]
}

/// Placeholder: returns None even for empty candidates (should return Some with a reason).
pub fn flag_if_empty_candidates(_candidates: &[String]) -> Option<String> {
    // Placeholder: should return Some("empty candidate set at resolution") when empty.
    None
}

/// Check the grounding of a value whose resolution requires a candidate set.
/// Placeholder: always returns Grounded even with an empty candidate set.
pub fn check_candidate_grounding(_candidates: &[String]) -> GroundingVerdict {
    // Placeholder: should return Flagged(EmptyCandidateSet) when candidates is empty.
    GroundingVerdict::Grounded
}
