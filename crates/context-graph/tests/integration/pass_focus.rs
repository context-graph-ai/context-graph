//! Focused extraction renders only the requested descriptor fields, while the full pass
//! renders the full ontology.

use context_graph::harness::CaptureHarnessConfig;
use context_graph::ingest::ExtractionPass;

#[test]
fn pass_focus_filters_descriptor_fields() {
    let _alongside = crate::runs_alone::alongside_siblings();
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
    // Decisions attach to goals (the goal linkage), so the decisions pass keeps goals available…
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
    let _alongside = crate::runs_alone::alongside_siblings();
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
/// override extract_pass with the focused contract; the trait default re-runs the full pass.
#[test]
fn production_extractor_overrides_extract_pass() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let main_rs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cg-cli/src/main.rs");
    let source = std::fs::read_to_string(&main_rs).expect("cg-cli main.rs readable");
    assert!(
        source.contains("fn extract_pass"),
        "ConnectedModelExtractor must override extract_pass with the pass-focused contract"
    );
}
