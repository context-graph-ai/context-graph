use context_graph::grounding::{
    FlagReason, GroundingInput, GroundingVerdict, check_candidate_grounding, full_grounding_check,
    legal_entity_type_names, should_flag, why_flagged_closed_vocab,
};
use std::collections::BTreeMap;

use context_graph::ingest::CaptureRequest;
use context_graph::{CreateEntity, EntityType, Store, StoreConfig};

use super::action_test_extractors::{
    AmbiguousDecisionTypeExtractor, AmbiguousSpanExtractor, FiveFabricatedValuesExtractor,
    MislabeledDecisionTypeExtractor, MislabeledSpanExtractor, OutOfVocabDecisionTypeExtractor,
    OutOfVocabExtractor, SpanFaithfulExtractor,
};
use super::fake_extractor::open_store_with_lexical_embedder;

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

// Unsupported spans are flagged across entities, decision types, agent ids, observation values
// and outcome values. The double emits one fabricated value per class, absent from the transcript.
#[test]
fn value_with_missing_span_is_flagged() {
    let (_dir, store) = fresh_store();

    // A transcript that contains NONE of the five fabricated phrases.
    let transcript = "the team discussed routine maintenance and nothing else of note today";

    let report = capture_with(
        &store,
        "missing-span-all-classes-context",
        transcript,
        &FiveFabricatedValuesExtractor,
    );

    // Each fabricated value appears in the flagged block under its own value text.
    let flagged = |needle: &str| {
        report
            .flagged_items
            .iter()
            .any(|f| f.reason.contains(needle))
    };

    assert!(
        flagged(FiveFabricatedValuesExtractor::FAB_ENTITY),
        "the fabricated ENTITY must be flagged (span absent); flagged_items: {:?}",
        report.flagged_items
    );
    assert!(
        flagged(FiveFabricatedValuesExtractor::FAB_DECISION_TYPE),
        "the fabricated DECISION_TYPE must be flagged (span absent); grounding must reach the \
         decision_type class, not entities only; flagged_items: {:?}",
        report.flagged_items
    );
    assert!(
        flagged(FiveFabricatedValuesExtractor::FAB_AGENT_ID),
        "the fabricated AGENT_ID must be flagged (span absent); grounding must reach the agent_id \
         class; flagged_items: {:?}",
        report.flagged_items
    );
    assert!(
        flagged(FiveFabricatedValuesExtractor::FAB_OBSERVATION),
        "the fabricated OBSERVATION value must be flagged (span absent); grounding must reach the \
         observation class; flagged_items: {:?}",
        report.flagged_items
    );
    assert!(
        flagged(FiveFabricatedValuesExtractor::FAB_OUTCOME),
        "the fabricated OUTCOME value must be flagged (span absent); grounding must reach the \
         outcome class; flagged_items: {:?}",
        report.flagged_items
    );

    // Direct predicate check stays: missing cited_span flags regardless of confidence.
    let missing_span = GroundingInput {
        value: "kubernetes cluster".to_string(),
        cited_span: None, // absent
        chunk: transcript.to_string(),
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

    // full_grounding_check must flag this with the SPECIFIC mislabel reason — not merely "not
    // Grounded". A stub that flags for the wrong reason (e.g. MissingSpan) fails this.
    let verdict = full_grounding_check(&mislabeled);
    assert_eq!(
        verdict,
        GroundingVerdict::Flagged(FlagReason::SpanDoesNotSupportLabel),
        "a present span that supports Repository but is labeled File must flag specifically as \
         SpanDoesNotSupportLabel; got {verdict:?}"
    );

    // Also test via the capture path with MislabeledSpanExtractor (entity arm).
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

    // CAPTURE-PATH arm : grounding must reach the closed-vocab DECISION_TYPE, not entities
    // only. A decision whose decision_type span is present but does not support the chosen label
    // must be flagged with the SPECIFIC SpanDoesNotSupportLabel reason. an incorrect implementation's capture grounding is
    // entity-only, so the decision_type flag is absent → this fails.
    let (_dir2, store2) = fresh_store();
    let report2 = capture_with(
        &store2,
        "mislabel-decision-context",
        "we will rewrite the authentication module to ship the auth rewrite",
        &MislabeledDecisionTypeExtractor,
    );
    assert!(
        report2.flagged_items.iter().any(|f| {
            f.reason.contains("SpanDoesNotSupportLabel")
                && (f.reason.contains("security_policy")
                    || f.reason.contains("authentication module"))
        }),
        "the mislabeled DECISION_TYPE must be flagged with the SpanDoesNotSupportLabel reason via \
         capture; an incorrect implementation's entity-only grounding lets it through. flagged_items: {:?}",
        report2.flagged_items
    );
}

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
    assert_eq!(
        verdict,
        GroundingVerdict::Flagged(FlagReason::OutOfVocab),
        "an out-of-vocab closed field ('unknown') must flag specifically as OutOfVocab even at high \
         confidence; got {verdict:?}"
    );

    // CAPTURE-PATH arm : an out-of-vocab closed-vocab DECISION_TYPE with a present span must be
    // flagged with the SPECIFIC OutOfVocab reason via capture. an incorrect implementation's entity-only grounding lets the
    // decision_type through → this fails.
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "oov-decision-context",
        "we use the wibblefangle approach to decide the storage engine",
        &OutOfVocabDecisionTypeExtractor,
    );
    assert!(
        report.flagged_items.iter().any(|f| {
            f.reason.contains("OutOfVocab")
                && (f.reason.contains("wibblefangle") || f.reason.contains("storage engine"))
        }),
        "the out-of-vocab DECISION_TYPE must be flagged with the OutOfVocab reason via capture; \
         an incorrect implementation's entity-only grounding lets it through. flagged_items: {:?}",
        report.flagged_items
    );
}

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
    assert_eq!(
        verdict,
        GroundingVerdict::Flagged(FlagReason::AmbiguousSpan),
        "a span supporting two competing labels equally must flag specifically as AmbiguousSpan; \
         got {verdict:?}"
    );

    // Also via capture path (entity arm).
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

    // CAPTURE-PATH arm : an ambiguous closed-vocab DECISION_TYPE (span supports two labels
    // equally) must be flagged with the SPECIFIC AmbiguousSpan reason via capture. an incorrect implementation's entity-only
    // grounding lets the decision_type through → this fails.
    let (_dir2, store2) = fresh_store();
    let report2 = capture_with(
        &store2,
        "ambiguous-decision-context",
        "we will adjust the policy to tune the gateway",
        &AmbiguousDecisionTypeExtractor,
    );
    assert!(
        report2.flagged_items.iter().any(|f| {
            f.reason.contains("AmbiguousSpan")
                && (f.reason.contains("policy") || f.reason.contains("adjust the policy"))
        }),
        "the ambiguous DECISION_TYPE must be flagged with the AmbiguousSpan reason via capture; \
         an incorrect implementation's entity-only grounding lets it through. flagged_items: {:?}",
        report2.flagged_items
    );
}

// A flagged closed-vocab item surfaces its LEGAL value set THROUGH the flagged item — the path
// resolves the item, finds its closed field, and reads the legal set from the single vocabulary
// source — NOT a direct constant lookup that returns EntityType::ALL regardless of any item. The
// load-bearing arm: `cg why` on a NON-flagged / fully-grounded item must NOT return a closed-vocab
// legal list, so a stub that always returns the constant list fails.
//
#[test]
fn flagged_closed_vocab_item_exposes_legal_type_list_via_cg_why() {
    // Capture so a real out-of-vocab entity_type is FLAGGED in the store (a real unsure item).
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "why-flagged-context",
        "mystery component handles routing",
        &OutOfVocabExtractor,
    );
    assert!(
        !report.flagged_items.is_empty(),
        "the OutOfVocabExtractor capture must produce a flagged item to reach `cg why` against"
    );
    let flagged_item_id = &report.flagged_items[0].key;

    // `cg why` on the FLAGGED closed-vocab item must enumerate the legal EntityType vocabulary. The
    // path takes only the store + item id and must DERIVE flagged-ness + the closed field from the
    // resolved item (a stub that never reads the store cannot tell flagged from non-flagged).
    let legal = context_graph::grounding::legal_types_for_flagged_item(&store, flagged_item_id);
    let legal_types = legal.expect("a flagged closed-vocab item must expose its legal vocabulary");
    assert_eq!(
        legal_types.len(),
        EntityType::ALL.len(),
        "the legal entity type list reached through the flagged item must contain every variant of the canonical vocabulary, got: {:?}",
        legal_types
    );
    for variant in EntityType::ALL.iter() {
        let variant_name = format!("{:?}", variant);
        assert!(
            legal_types
                .iter()
                .any(|n| n.eq_ignore_ascii_case(&variant_name)),
            "EntityType::{} must appear in the legal list reached through the flagged item",
            variant_name
        );
    }

    // An unknown id sharing a token with a stored entity name still resolves to None.
    let why_ctx = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == "why-flagged-context")
        .map(|c| c.id)
        .expect("why-flagged-context exists");
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::Repository,
            name: "store".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: why_ctx,
        })
        .expect("seed a stored entity named 'store'");
    let token_overlap_probe = "no-such-item-in-store";
    let non_flagged =
        context_graph::grounding::legal_types_for_flagged_item(&store, token_overlap_probe);
    assert_eq!(
        non_flagged, None,
        "`cg why` on a NON-flagged item ('{token_overlap_probe}') must return None even when a stored \
         entity name ('store') shares a token with the id — resolution must be BY ID, not token \
         overlap; the token-overlap heuristic wrongly returns the legal list and FAILS here"
    );

    // A genuinely-flagged closed-vocab item still resolves (by its own id) — the flagged arm above
    // already exercised this; re-confirm it is not the all-None failure mode.
    assert!(
        context_graph::grounding::legal_types_for_flagged_item(&store, flagged_item_id).is_some(),
        "a genuinely-flagged closed-vocab item must still resolve to its legal list by id"
    );

    // The single vocabulary source must itself enumerate the full canonical
    // vocabulary (kept as a sanity check on the source).
    assert_eq!(legal_entity_type_names().len(), EntityType::ALL.len());
    let _ = why_flagged_closed_vocab; // Check the canonical vocabulary as well as the item-keyed lookup.
}

#[test]
fn empty_candidate_set_at_resolution_is_flagged() {
    // The empty-candidate path specifically: an empty candidate set must flag as EmptyCandidateSet,
    // not merely "not Grounded" and not some unrelated span reason.
    let empty_verdict = check_candidate_grounding(&[]);
    assert_eq!(
        empty_verdict,
        GroundingVerdict::Flagged(FlagReason::EmptyCandidateSet),
        "an empty candidate set at resolution must flag specifically as EmptyCandidateSet; \
         got {empty_verdict:?}"
    );

    // Contrast (proves the predicate exercises the candidate-set path, not a constant): a NON-empty
    // candidate set is Grounded. A stub that ignores the candidates (always Grounded, or always
    // Flagged) fails one of these two arms.
    let non_empty_verdict = check_candidate_grounding(&["context-graph".to_string()]);
    assert_eq!(
        non_empty_verdict,
        GroundingVerdict::Grounded,
        "a non-empty candidate set must be Grounded (the empty-set flag is specific to no candidates); \
         got {non_empty_verdict:?}"
    );

    // CAPTURE-PATH arm : the LIVE capture path must drive Stage-3 resolution for a value that
    // needs a candidate and, finding ZERO in the store, flag it with the empty-candidate reason —
    // never silently mint-as-resolved. an incorrect implementation's Stage-3 silently treats a zero-candidate decision as a
    // new insertion and flags nothing → this fails.
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "empty-candidate-context",
        "Goal G1: choose the cache layer\nDecision D1: adopt the redis cache serves G1",
        &SpanFaithfulExtractor,
    );
    assert!(
        report
            .flagged_items
            .iter()
            .any(|f| f.reason.contains("EmptyCandidateSet")),
        "a Stage-3 resolution with ZERO candidates must land in report.flagged_items with the \
         EmptyCandidateSet reason; an implementation that silently mints-as-resolved and flags nothing. flagged_items: {:?}",
        report.flagged_items
    );
}

// The flag predicate is a pure function of (span, chunk, value, vocab, candidates).
// It must ignore model_confidence entirely.
// Identical inputs produce identical flags regardless of model confidence.
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
