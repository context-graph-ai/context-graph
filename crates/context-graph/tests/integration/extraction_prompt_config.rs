//! RED contract tests for criteria C3/C5 (criteria commit 99d99ee): the extraction prompt is
//! assembled by the LIBRARY from `CaptureHarnessConfig` (no CLI-resident framing), and every
//! recipe knob — item caps, prompt override, tier, chunk window/overlap, entity vocabulary — is
//! operator-reachable configuration. Authored and committed BEFORE implementation; implementation
//! commits may not touch this file.
//!
//! These are reachability/plumbing assertions (config → rendered prompt), never content
//! assertions on prompt wording — recipe CONTENT is measured by the bench against gold, per the
//! registered method.

use context_graph::harness::{CaptureHarnessConfig, ExtractionPromptInputs, ItemCaps};

fn inputs<'a>(transcript: &'a str) -> ExtractionPromptInputs<'a> {
    ExtractionPromptInputs {
        context_label: "test-context",
        transcript,
        prior_memory_block: "none",
        steering_block: None,
    }
}

#[test]
fn extraction_prompt_assembly_honors_caps_override_and_vocabulary() {
    let default_config = CaptureHarnessConfig::default();
    let default_prompt = default_config.render_extraction_prompt(&inputs("hello transcript"));
    // The transcript rides inside the assembly, delimited (capture keys on the markers).
    assert!(default_prompt.contains("hello transcript"));
    assert!(default_prompt.contains("<cg_transcript>"));
    // The schema block the descriptor renders is part of the assembly.
    let descriptor = default_config.schema_descriptor().expect("descriptor");
    assert!(default_prompt.contains(&descriptor.render_prompt_schema_block()));

    // Item caps are CONFIG: changing them changes the outgoing prompt.
    let capped = CaptureHarnessConfig {
        item_caps: Some(ItemCaps {
            goals: 2,
            entities: 16,
            decisions: 12,
            observations: 4,
        }),
        ..CaptureHarnessConfig::default()
    };
    let capped_prompt = capped.render_extraction_prompt(&inputs("hello transcript"));
    assert_ne!(default_prompt, capped_prompt, "caps must reach the prompt");
    assert!(
        capped_prompt.contains("16 entities"),
        "the configured entity cap must appear in the rendered prompt"
    );
    assert!(capped_prompt.contains("12 decisions"));

    // An operator prompt_override replaces the schema block verbatim.
    let overridden = CaptureHarnessConfig {
        prompt_override: Some("CUSTOM SCHEMA BLOCK".to_string()),
        ..CaptureHarnessConfig::default()
    };
    let overridden_prompt = overridden.render_extraction_prompt(&inputs("hello transcript"));
    assert!(overridden_prompt.contains("CUSTOM SCHEMA BLOCK"));
    assert!(!overridden_prompt.contains(&descriptor.render_prompt_schema_block()));
}

#[test]
fn custom_entity_vocabulary_reaches_schema_values() {
    // Owner decision 2026-07-03: the entity noun list is USER-SUPPLIED configuration, never
    // hardcoded ("user supplied, never in code, can be a yaml"). First slice of
    // cg-open-entity-vocabulary: the configured list feeds schema + prompt from ONE source.
    let custom = CaptureHarnessConfig {
        entity_vocabulary: Some(vec!["vehicle".to_string(), "lead".to_string()]),
        ..CaptureHarnessConfig::default()
    };
    assert_eq!(
        custom.allowed_entity_type_schema_values(),
        vec!["vehicle".to_string(), "lead".to_string()]
    );
    let descriptor = custom.schema_descriptor().expect("descriptor");
    assert_eq!(
        descriptor.entity_type_values,
        vec!["vehicle".to_string(), "lead".to_string()]
    );
    let prompt = custom.render_extraction_prompt(&inputs("t"));
    assert!(prompt.contains("vehicle"));
    assert!(prompt.contains("lead"));
}

#[test]
fn default_vocabulary_unchanged_thirteen_types() {
    let config = CaptureHarnessConfig::default();
    let values = config.allowed_entity_type_schema_values();
    assert_eq!(
        values.len(),
        13,
        "default stays the 13 built-ins — no regression"
    );
    assert!(values.contains(&"repository".to_string()));
    assert!(values.contains(&"file".to_string()));
}

#[test]
fn capture_config_parses_from_operator_toml() {
    let toml_text = r#"
[capture]
tier = "haiku"
chunk_window = 24000
chunk_overlap = 2000
recall_limit = 8
skills_enabled = true
review_mode = false
entity_vocabulary = ["vehicle", "lead"]

[capture.item_caps]
goals = 2
entities = 16
decisions = 12
observations = 4
"#;
    let config = CaptureHarnessConfig::from_toml_str(toml_text).expect("operator toml parses");
    assert_eq!(config.tier.as_deref(), Some("haiku"));
    assert_eq!(config.chunk_window, Some(24000));
    assert_eq!(config.chunk_overlap, Some(2000));
    assert_eq!(config.recall_limit, Some(8));
    assert_eq!(
        config.entity_vocabulary,
        Some(vec!["vehicle".to_string(), "lead".to_string()])
    );
    let caps = config.item_caps.expect("caps parsed");
    assert_eq!(
        (caps.goals, caps.entities, caps.decisions, caps.observations),
        (2, 16, 12, 4)
    );

    // An empty file is the defaults — an absent capture.toml never breaks capture.
    let empty = CaptureHarnessConfig::from_toml_str("").expect("empty toml is defaults");
    assert_eq!(empty.tier, None);
    assert_eq!(empty.item_caps, None);
    assert_eq!(empty.entity_vocabulary, None);

    // A malformed file fails LOUD (never silently reverts to defaults).
    assert!(CaptureHarnessConfig::from_toml_str("[capture\ntier=").is_err());
}
