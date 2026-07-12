//! RED contract tests for the pass-focus half of staged extraction: a focused
//! extraction pass must render a FOCUSED contract — descriptor filtered to the pass's fields —
//! not the full-ontology ask re-run. Found live: production's extract_pass fell through to the
//! trait default, so "staged" measured self-consistency (full pass ×3, merged), not the probe-2
//! focused-pass lever. Authored and committed BEFORE implementation; implementation commits may
//! not touch this file.

use context_graph::harness::CaptureHarnessConfig;
use context_graph::ingest::ExtractionPass;

#[test]
fn pass_focus_filters_descriptor_fields() {
    let config = CaptureHarnessConfig::default();

    let full = config
        .schema_descriptor_for_pass(ExtractionPass::Full)
        .expect("descriptor");
    assert!(full.fields.iter().any(|f| f.name == "goals"));
    assert!(full.fields.iter().any(|f| f.name == "decisions"));
    assert!(full.fields.iter().any(|f| f.name == "entities"));

    let entities_only = config
        .schema_descriptor_for_pass(ExtractionPass::EntitiesOnly)
        .expect("descriptor");
    assert!(
        entities_only.fields.iter().any(|f| f.name == "entities"),
        "the entities pass asks for entities"
    );
    assert!(
        !entities_only.fields.iter().any(|f| f.name == "decisions"),
        "the entities pass does NOT re-ask the full ontology"
    );
    assert!(!entities_only.fields.iter().any(|f| f.name == "goals"));

    let decisions_only = config
        .schema_descriptor_for_pass(ExtractionPass::DecisionsOnly)
        .expect("descriptor");
    assert!(decisions_only.fields.iter().any(|f| f.name == "decisions"));
    // Decisions attach to goals (the trunk), so the decisions pass keeps goals available…
    assert!(decisions_only.fields.iter().any(|f| f.name == "goals"));
    // …but not the rest of the ontology.
    assert!(!decisions_only.fields.iter().any(|f| f.name == "entities"));
    assert!(
        !decisions_only
            .fields
            .iter()
            .any(|f| f.name == "observations")
    );
}

#[test]
fn focused_pass_prompt_differs_from_full() {
    let config = CaptureHarnessConfig::default();
    let inputs = context_graph::harness::ExtractionPromptInputs {
        context_label: "t",
        transcript: "some transcript",
        prior_memory_block: "none",
        steering_block: None,
    };
    let full = config.render_extraction_prompt_for_pass(&inputs, ExtractionPass::Full);
    let entities = config.render_extraction_prompt_for_pass(&inputs, ExtractionPass::EntitiesOnly);
    assert_ne!(full, entities, "a focused pass renders a focused prompt");
    // The full render stays byte-identical to the shipped single-pass render (no regression).
    assert_eq!(full, config.render_extraction_prompt(&inputs));
}

/// Source-scan guard (same class as the delegation guard): the production extractor must
/// actually OVERRIDE extract_pass — the trait default silently re-runs the full pass, which is
/// exactly the live gap this closes.
#[test]
fn production_extractor_overrides_extract_pass() {
    let main_rs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cg-cli/src/main.rs");
    let source = std::fs::read_to_string(&main_rs).expect("cg-cli main.rs readable");
    assert!(
        source.contains("fn extract_pass"),
        "ConnectedModelExtractor must override extract_pass with the pass-focused contract"
    );
}
