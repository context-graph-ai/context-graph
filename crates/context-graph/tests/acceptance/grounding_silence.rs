use context_graph::grounding::{
    GroundingInput, GroundingVerdict, check_candidate_grounding, full_grounding_check,
    legal_entity_type_names, should_flag, why_flagged_closed_vocab,
};
use context_graph::ingest::CaptureRequest;
use context_graph::{EntityType, Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::{
    AmbiguousSpanExtractor, HighConfidenceFabricatedSpanExtractor, MislabeledSpanExtractor,
    OutOfVocabExtractor, SpanFaithfulExtractor,
};

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn capture_with<E: context_graph::ingest::ConversationExtractor>(
    store: &Store,
    context_label: &str,
    transcript: &str,
    extractor: &E,
) -> context_graph::ingest::CaptureReport {
    context_graph::ingest::record_conversation_with_extractor(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        extractor,
    )
    .expect("capture succeeded")
}

// A fully span-faithful capture produces an empty flagged-items list.
// Silence holds for BOTH high- and low-confidence values (span presence decides, not confidence).
// The placeholder full_grounding_check flags on low confidence, so low-confidence span-faithful
// values incorrectly appear in flagged_items.
#[test]
fn every_extracted_value_with_a_supporting_span_present_and_aligned_is_not_flagged() {
    let (_dir, store) = fresh_store();

    // Transcript that contains spans for all emitted values.
    let transcript = "Goal G1: keep deployment reliable. \
        Decision D1: use Kubernetes for orchestration serves G1. \
        Entity E1: Kubernetes type=infrastructure.";

    let report = capture_with(
        &store,
        "span-faithful-context",
        transcript,
        &SpanFaithfulExtractor,
    );

    // Placeholder: flags on model_confidence <= 0.5, so low-confidence span-faithful values get flagged.
    assert!(
        report.flagged_items.is_empty(),
        "flagged_items must be empty for a fully span-faithful capture, got: {:?}",
        report.flagged_items
    );

    // Run the grounding predicate directly on a span-faithful input with HIGH confidence.
    let grounded_high = GroundingInput {
        value: "Kubernetes".to_string(),
        cited_span: Some("Kubernetes".to_string()),
        chunk: "use Kubernetes for orchestration".to_string(),
        model_confidence: 0.99,
        vocab: None,
        label: None,
        competing_labels: vec![],
    };
    // Run again with LOW confidence, same faithful span.
    let grounded_low = GroundingInput {
        model_confidence: 0.01,
        ..grounded_high.clone()
    };

    assert!(
        !should_flag(&grounded_high),
        "span-faithful high-confidence value must not be flagged"
    );
    assert!(
        !should_flag(&grounded_low),
        "span-faithful LOW-confidence value must also not be flagged (span decides)"
    );
}

// A value whose cited span is absent from the chunk is flagged.
// The placeholder uses a confidence threshold — this value with missing span and high
// confidence incorrectly passes through.
#[test]
fn value_with_missing_span_is_flagged() {
    let (_dir, store) = fresh_store();

    // Transcript that does NOT contain "kubernetes cluster".
    let transcript = "Goal G1: test fabricated span detection.";

    let report = capture_with(
        &store,
        "missing-span-context",
        transcript,
        &HighConfidenceFabricatedSpanExtractor,
    );

    // Assert via the report's flagged_items.
    assert!(
        !report.flagged_items.is_empty(),
        "flagged_items must be non-empty when span is absent from transcript"
    );

    // Also assert directly via the grounding predicate.
    let missing_span = GroundingInput {
        value: "kubernetes cluster".to_string(),
        cited_span: None, // absent
        chunk: "Goal G1: test fabricated span detection.".to_string(),
        model_confidence: 0.99,
        vocab: None,
        label: None,
        competing_labels: vec![],
    };
    assert!(
        should_flag(&missing_span),
        "missing cited_span must cause should_flag=true regardless of confidence"
    );
}

// A model-confident, in-vocab value whose span is fabricated (not in chunk) is still flagged.
// The placeholder flags on low confidence: this high-confidence input incorrectly passes through.
// Second arm: same real span, low confidence — must NOT be flagged.
#[test]
fn confident_value_with_fabricated_span_is_still_flagged() {
    // Arm 1: high confidence + span NOT in chunk → must be flagged.
    let fabricated = GroundingInput {
        value: "kubernetes cluster".to_string(),
        cited_span: Some("kubernetes cluster".to_string()), // span text given
        chunk: "we use a container runtime".to_string(),    // span NOT in chunk
        model_confidence: 0.99,
        vocab: None,
        label: None,
        competing_labels: vec![],
    };
    assert!(
        should_flag(&fabricated),
        "high-confidence value with span absent from chunk must be flagged"
    );

    // Arm 2: low confidence + span IS in chunk → must NOT be flagged.
    // The placeholder would flag this (because confidence <= 0.5 triggers flag).
    let real_span_low_confidence = GroundingInput {
        value: "container runtime".to_string(),
        cited_span: Some("container runtime".to_string()), // span is in chunk
        chunk: "we use a container runtime".to_string(),
        model_confidence: 0.10,
        vocab: None,
        label: None,
        competing_labels: vec![],
    };
    assert!(
        !should_flag(&real_span_low_confidence),
        "low-confidence value with span IN chunk must NOT be flagged (span decides)"
    );
}

// A span present in the chunk but not supporting the chosen label is flagged.
// The placeholder uses confidence, not label-support — passes mislabeled spans through.
#[test]
fn span_present_but_not_supporting_chosen_label_is_flagged() {
    // "the context-graph repo" is a Repository span, but chosen type is File.
    let mislabeled = GroundingInput {
        value: "the context-graph repo".to_string(),
        cited_span: Some("the context-graph repo".to_string()),
        chunk: "the context-graph repo is a rust workspace".to_string(),
        model_confidence: 0.95,
        vocab: Some(EntityType::ALL.iter().map(|t| format!("{:?}", t)).collect()),
        label: Some("File".to_string()), // wrong label for this span
        competing_labels: vec!["Repository".to_string()], // span actually supports Repository
    };

    // full_grounding_check must flag this as SpanDoesNotSupportLabel.
    let verdict = full_grounding_check(&mislabeled);
    assert_ne!(
        verdict,
        GroundingVerdict::Grounded,
        "mislabeled span (span supports Repository, chosen label File) must be flagged"
    );

    // Also test via the capture path with MislabeledSpanExtractor.
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "mislabel-context",
        "the context-graph repo is a rust workspace",
        &MislabeledSpanExtractor,
    );
    assert!(
        !report.flagged_items.is_empty(),
        "mislabeled entity span must appear in flagged_items"
    );
}

// An out-of-vocab entity_type ("unknown" or non-vocab string) is flagged, not silently stored.
// The placeholder flags on confidence, so an out-of-vocab high-confidence value passes through.
// The capture-path arm is omitted: OutOfVocabExtractor's entity_type deserializes to the
// in-vocab File default (the closed enum cannot carry an out-of-vocab token), so asserting
// that a capture with it is flagged would force flagging a valid entity — the opposite of
// the contract. The direct-predicate arm is load-bearing: it drives the real out-of-vocab
// raw string through the grounding predicate and asserts it is flagged.
#[test]
fn out_of_vocab_or_unknown_closed_field_is_flagged() {
    // Direct predicate check: out-of-vocab value with span present → still flagged.
    let out_of_vocab = GroundingInput {
        value: "unknown".to_string(),
        cited_span: Some("unknown".to_string()),
        chunk: "the unknown component handles routing".to_string(),
        model_confidence: 0.88,
        vocab: Some(EntityType::ALL.iter().map(|t| format!("{:?}", t)).collect()),
        label: Some("unknown".to_string()),
        competing_labels: vec![],
    };

    let verdict = full_grounding_check(&out_of_vocab);
    assert_ne!(
        verdict,
        GroundingVerdict::Grounded,
        "out-of-vocab entity_type='unknown' must be flagged even at high confidence; placeholder returns Grounded"
    );
}

// A span supporting two closed-vocab labels equally is flagged, not silently resolved to one.
// The placeholder never flags ambiguous spans (is_ambiguous_span always returns false).
#[test]
fn ambiguous_span_supporting_two_labels_equally_is_flagged() {
    // "auth component" supports both Service and Module equally with no tie-breaker.
    let ambiguous = GroundingInput {
        value: "auth component".to_string(),
        cited_span: Some("auth component".to_string()),
        chunk: "the auth component handles routing".to_string(),
        model_confidence: 0.85,
        vocab: None,
        label: Some("Service".to_string()),
        competing_labels: vec!["Service".to_string(), "Module".to_string()],
    };

    let verdict = full_grounding_check(&ambiguous);
    assert_ne!(
        verdict,
        GroundingVerdict::Grounded,
        "ambiguous span must be flagged when two competing labels score equally"
    );

    // Also via capture path.
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "ambiguous-context",
        "the auth component handles routing",
        &AmbiguousSpanExtractor,
    );
    assert!(
        !report.flagged_items.is_empty(),
        "ambiguous entity span must appear in flagged_items"
    );
}

// A flagged closed-vocab item exposes the legal type list via legal_entity_type_names().
// The placeholder returns empty — the real implementation returns all 13 names.
#[test]
fn flagged_closed_vocab_item_exposes_legal_type_list_via_cg_why() {
    // why_flagged_closed_vocab must return the full legal vocab for a flagged entity_type.
    let legal_types = why_flagged_closed_vocab("unknown");

    assert!(
        !legal_types.is_empty(),
        "why_flagged_closed_vocab must return the legal entity type list, not empty"
    );
    assert_eq!(
        legal_types.len(),
        13,
        "legal entity type list must contain all 13 variants, got: {:?}",
        legal_types
    );

    // Also directly via legal_entity_type_names().
    let names = legal_entity_type_names();
    assert_eq!(
        names.len(),
        13,
        "legal_entity_type_names must return 13 names, got: {:?}",
        names
    );

    // Each EntityType variant must appear in the list.
    for variant in EntityType::ALL.iter() {
        let variant_name = format!("{:?}", variant);
        assert!(
            names.iter().any(|n| n.eq_ignore_ascii_case(&variant_name)),
            "EntityType::{} must appear in legal_entity_type_names()",
            variant_name
        );
    }
}

// When Stage-3 resolution has an empty candidate set, the item is flagged.
// The placeholder check_candidate_grounding returns Grounded even on empty candidates.
#[test]
fn empty_candidate_set_at_resolution_is_flagged() {
    // Direct predicate check: empty candidates → must be flagged.
    let verdict = check_candidate_grounding(&[]);
    assert_ne!(
        verdict,
        GroundingVerdict::Grounded,
        "check_candidate_grounding([]) must return Flagged, not Grounded"
    );

    // Also via capture path with a fresh store so resolution candidates are empty.
    let (_dir, store) = fresh_store();
    let transcript = "Goal G1: test empty candidate resolution. \
        Entity E1: mystery component type=unknown.";
    let report = capture_with(
        &store,
        "empty-candidates-context",
        transcript,
        &OutOfVocabExtractor,
    );
    // With an empty/unresolvable value, the item must appear in flagged_items.
    assert!(
        !report.flagged_items.is_empty(),
        "empty candidate set at resolution must produce a flagged item"
    );
}

// The flag predicate is a pure function of (span, chunk, value, vocab, candidates).
// It must ignore model_confidence entirely.
// The placeholder should_flag uses a confidence threshold — identical inputs with different
// confidence produce different results. The correct implementation produces the same result.
// This is a unit test — no store.
#[test]
fn flag_signal_is_deterministic_independent_of_model_confidence_score() {
    // Two inputs identical except model_confidence, with a REAL supporting span.
    let grounded_high_conf = GroundingInput {
        value: "repo".to_string(),
        cited_span: Some("repo".to_string()),
        chunk: "the repo is on GitHub".to_string(),
        model_confidence: 0.99,
        vocab: None,
        label: None,
        competing_labels: vec![],
    };
    let grounded_low_conf = GroundingInput {
        model_confidence: 0.01,
        ..grounded_high_conf.clone()
    };

    // Both must produce the same verdict (both not-flagged when span is present and supported).
    // The placeholder produces different results: 0.99 not-flagged, 0.01 flagged.
    assert_eq!(
        should_flag(&grounded_high_conf),
        should_flag(&grounded_low_conf),
        "should_flag must be identical for same span regardless of model_confidence"
    );

    // Also: with a FABRICATED span (absent from chunk), both confidence levels must flag.
    let fabricated_high = GroundingInput {
        value: "kubernetes cluster".to_string(),
        cited_span: Some("kubernetes cluster".to_string()), // not in chunk
        chunk: "the repo is on GitHub".to_string(),
        model_confidence: 0.99,
        vocab: None,
        label: None,
        competing_labels: vec![],
    };
    let fabricated_low = GroundingInput {
        model_confidence: 0.01,
        ..fabricated_high.clone()
    };

    assert_eq!(
        should_flag(&fabricated_high),
        should_flag(&fabricated_low),
        "should_flag must be identical for fabricated span regardless of model_confidence"
    );

    // Discriminator: grounded verdict must differ from fabricated verdict.
    // (One is false, one is true — proving the predicate responds to span, not confidence.)
    assert_ne!(
        should_flag(&grounded_high_conf),
        should_flag(&fabricated_high),
        "span-faithful and fabricated-span inputs must produce different flag verdicts"
    );
}
