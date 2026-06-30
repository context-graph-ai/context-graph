//! Deterministic grounding predicates. A value is flagged based on span presence and alignment
//! in the source transcript — the model's confidence score is never consulted.

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

/// Returns true when the value is fully grounded: cited_span is present and found in the
/// chunk. The model's confidence score is never consulted.
pub fn is_grounded(input: &GroundingInput) -> bool {
    let Some(span) = &input.cited_span else {
        return false;
    };
    if span.is_empty() {
        return false;
    }
    // Span must appear in the chunk (case-sensitive substring match).
    input.chunk.contains(span.as_str())
}

/// Returns true when the value must be flagged for human review.
/// The flag is a pure function of (span, chunk, value, vocab, competing_labels).
/// The model's confidence score is never consulted.
pub fn should_flag(input: &GroundingInput) -> bool {
    !is_grounded(input)
}

/// Check whether a span ambiguously supports two or more competing labels equally.
/// Returns true when there are two or more competing labels (a tie with no disambiguator).
pub fn is_ambiguous_span(input: &GroundingInput) -> bool {
    input.competing_labels.len() >= 2
}

/// Full grounding check: span presence, span-in-chunk, out-of-vocab, mislabel, ambiguity.
/// Returns Grounded only when all checks pass. Never consults model_confidence.
pub fn full_grounding_check(input: &GroundingInput) -> GroundingVerdict {
    // 1. Span must be present.
    let Some(span) = &input.cited_span else {
        return GroundingVerdict::Flagged(FlagReason::MissingSpan);
    };
    if span.is_empty() {
        return GroundingVerdict::Flagged(FlagReason::MissingSpan);
    }

    // 2. Span text must appear in the chunk.
    if !input.chunk.contains(span.as_str()) {
        return GroundingVerdict::Flagged(FlagReason::SpanNotInChunk);
    }

    // 3. If a closed vocab is supplied, the label must be in it.
    if let Some(vocab) = &input.vocab
        && let Some(label) = &input.label
    {
        let in_vocab = vocab.iter().any(|v| v.eq_ignore_ascii_case(label));
        if !in_vocab {
            return GroundingVerdict::Flagged(FlagReason::OutOfVocab);
        }
    }

    // 4. Ambiguity: two or more equally-supported competing labels.
    if is_ambiguous_span(input) {
        return GroundingVerdict::Flagged(FlagReason::AmbiguousSpan);
    }

    // 5. The span must support the chosen label (not an obviously-wrong label).
    // When competing_labels is exactly one and differs from the chosen label, the span
    // supports a different label than the one selected — flag as mislabeled.
    if let Some(label) = &input.label
        && input.competing_labels.len() == 1
        && &input.competing_labels[0] != label
    {
        return GroundingVerdict::Flagged(FlagReason::SpanDoesNotSupportLabel);
    }

    GroundingVerdict::Grounded
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
}

/// Returns the canonical set of legal entity type names, derived from the single
/// vocabulary source on `CaptureHarnessConfig::allowed_entity_types()` default.
/// All three consumers — the schema descriptor, the grounding closed-vocab check, and
/// the `why_flagged_closed_vocab` response — read from this one function.
pub fn legal_entity_type_names() -> Vec<String> {
    crate::harness::CaptureHarnessConfig::default().allowed_entity_type_names()
}

/// Returns the canonical legal entity type values (same source as `legal_entity_type_names`).
pub fn legal_entity_types() -> Vec<String> {
    legal_entity_type_names()
}

/// Returns the legal entity type names when a closed-vocab item is flagged, so the caller
/// can present a named correction set. Always returns all 13 names from the canonical source.
pub fn why_flagged_closed_vocab(_flagged_value: &str) -> Vec<String> {
    legal_entity_type_names()
}

/// `cg why @<item>` legal-vocabulary path: resolve a flagged item BY ITS ID against the persisted
/// pending-review-item row it refers to, and return the legal value set the correction must come from
/// when it is a genuinely-flagged CLOSED-VOCAB item (the row carries a `legal_vocab` — e.g. the 13
/// `EntityType` values for an entity-type flag). A non-flagged id (no pending row by that exact id) —
/// or a free-text flag with no constrained vocabulary — returns `None` (the caller renders free text /
/// provenance instead). Resolution is BY EXACT ID, never token overlap against entity names: a stored
/// entity whose NAME merely shares a token with the id does not make a non-flagged probe resolve.
pub fn legal_types_for_flagged_item(store: &crate::Store, item_id: &str) -> Option<Vec<String>> {
    crate::review_surface::legal_vocab_for_pending_item(store, item_id)
}

/// Returns `Some(reason)` when the candidate set is empty, indicating the resolution
/// could not be grounded against any existing candidate.
pub fn flag_if_empty_candidates(candidates: &[String]) -> Option<String> {
    if candidates.is_empty() {
        Some("empty candidate set at resolution".to_string())
    } else {
        None
    }
}

/// Returns Flagged(EmptyCandidateSet) when the candidate slice is empty.
pub fn check_candidate_grounding(candidates: &[String]) -> GroundingVerdict {
    if candidates.is_empty() {
        GroundingVerdict::Flagged(FlagReason::EmptyCandidateSet)
    } else {
        GroundingVerdict::Grounded
    }
}

/// Check an entity name and type against the transcript for grounding.
///
/// Returns `Some(reason)` when the entity must be flagged, or `None` when it is grounded.
///
/// An entity is grounded when:
/// - Its name appears (case-insensitive) in the transcript, AND
/// - Its type name (e.g. "Module") appears (case-insensitive) somewhere in the transcript.
///
/// When the type name is absent from the transcript, the span does not support the chosen
/// label — it is flagged as `SpanDoesNotSupportLabel`.  When the name itself is absent,
/// it is flagged as `SpanNotInChunk`.
pub fn check_entity_grounding(
    entity_name: &str,
    entity_type_name: &str,
    transcript: &str,
) -> Option<FlagReason> {
    let transcript_lower = transcript.to_ascii_lowercase();
    // 1. Entity name must appear in the transcript.
    if !transcript_lower.contains(&entity_name.to_ascii_lowercase()) {
        return Some(FlagReason::SpanNotInChunk);
    }
    // 2. The chosen entity type name must appear somewhere in the transcript.
    // When it does not, the transcript's language does not support the chosen label.
    if !transcript_lower.contains(&entity_type_name.to_ascii_lowercase()) {
        return Some(FlagReason::SpanDoesNotSupportLabel);
    }
    None
}
