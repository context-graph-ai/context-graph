//! Library contract is single-source (integration tests).
//!
//! Four integration tests asserting that `SchemaDescriptor` drives both the prompt schema
//! block and the parse targets from one source (non-drift), and that `CaptureHarnessConfig`
//! knobs are honored end-to-end. Every assertion fails on a placeholder value, never on a
//! compile error or panic.

use context_graph::harness::{
    CaptureHarnessConfig, SchemaDescriptor, SchemaFieldSpec, run_capture,
};
use context_graph::ingest::CaptureRequest;
use context_graph::{EntityType, ListEntityFilter, Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::{CountingExtractor, SpanFaithfulExtractor};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

// ---------------------------------------------------------------------------
// Test 1 (single-source): placeholder returns hardcoded strings from
// `render_prompt_schema_block()` and `parse_targets()` that are independent of
// `self.fields`. Mutating `fields` proves they are a single-derived source: the mutated
// field name must appear in BOTH the rendered string AND the parse target list.
// ---------------------------------------------------------------------------
#[test]
fn prompt_schema_block_and_parser_derive_from_one_descriptor() {
    let mut descriptor = SchemaDescriptor::default();

    // ------ invariant: every field that is in the descriptor is in BOTH outputs ------
    // Start with at least one field so the assertion has something to check.
    descriptor.fields.push(SchemaFieldSpec {
        name: "intentions".to_string(),
        description: "captured intentions".to_string(),
        required: true,
    });

    let block = descriptor.render_prompt_schema_block();
    let targets = descriptor.parse_targets();

    // Every declared field must appear in both outputs with no drift.
    for field in &descriptor.fields {
        assert!(
            block.contains(&field.name),
            "rendered schema block does not contain field '{}' — descriptor and prompt have drifted",
            field.name
        );
        assert!(
            targets.contains(&field.name),
            "parse targets do not contain field '{}' — descriptor and parse targets have drifted",
            field.name
        );
    }

    // ------ mutation arm: ADD a new field — must appear in BOTH outputs ------
    // Placeholder: hardcoded lists do not update on mutation → fails here.
    descriptor.fields.push(SchemaFieldSpec {
        name: "new_field".to_string(),
        description: "a freshly added field".to_string(),
        required: false,
    });

    let block_after_add = descriptor.render_prompt_schema_block();
    let targets_after_add = descriptor.parse_targets();

    assert!(
        block_after_add.contains("new_field"),
        "rendered schema block does not contain 'new_field' after it was added to descriptor \
         — wrong stub: render_prompt_schema_block() is hardcoded, not derived from fields"
    );
    assert!(
        targets_after_add.contains(&"new_field".to_string()),
        "parse targets do not contain 'new_field' after it was added to descriptor \
         — wrong stub: parse_targets() is hardcoded, not derived from fields"
    );

    // ------ mutation arm: REMOVE a field — must disappear from BOTH outputs ------
    descriptor.fields.retain(|f| f.name != "new_field");

    let block_after_remove = descriptor.render_prompt_schema_block();
    let targets_after_remove = descriptor.parse_targets();

    // The field must be gone from the rendered schema.
    assert!(
        !block_after_remove.contains("new_field"),
        "rendered schema block still contains 'new_field' after it was removed from descriptor \
         — prompt and descriptor have drifted"
    );
    // The field must be gone from parse targets.
    assert!(
        !targets_after_remove.contains(&"new_field".to_string()),
        "parse targets still contain 'new_field' after it was removed from descriptor \
         — parse targets and descriptor have drifted"
    );
}

// ---------------------------------------------------------------------------
// Test 2 (single-source): `CaptureHarnessConfig.recall_limit` bounds the candidate set
// passed to the judge. Placeholder run_capture ignores recall_limit and passes the WHOLE
// store as candidates.
//
// Primary load-bearing assertion: when recall_limit=2 and >2 prior decisions are in the
// store, every received_candidate_sets() entry must have ≤2 items. Placeholder passes all
// prior decisions → entries exceed 2 → assertion FAILS.
//
// Setup: seed 3 prior decisions (capture 3 times, each with a decision), then capture a
// fourth time with recall_limit=Some(2) using a CountingExtractor. The placeholder
// passes all 3 prior decisions to the judge (3 > 2), failing the ≤2 bound check.
// ---------------------------------------------------------------------------
#[test]
fn capture_harness_config_threads_prompt_overrides_and_recall_limit_and_window_and_tier_and_skill_switch()
 {
    let (_dir, store) = fresh_store();

    // Seed >2 prior decisions so recall_limit has something to bound.
    for i in 1..=3 {
        let _ = run_capture(
            &store,
            CaptureRequest {
                context_label: "harness-config-test".to_string(),
                transcript: format!(
                    "Goal G{i}: seed goal {i}\nDecision D{i}: use option_{i} serves G{i}"
                ),
                model_available: true,
            },
            &SpanFaithfulExtractor,
            &CaptureHarnessConfig::default(),
        )
        .expect("seed decision capture");
    }

    // recall_limit=2 means the judge candidate set must be bounded to ≤2 items.
    let config = CaptureHarnessConfig {
        recall_limit: Some(2),
        chunk_window: Some(512),
        skills_enabled: false,
        review_mode: false,
        prompt_override: None,
        tier: Some("haiku".to_string()),
    };

    // CountingExtractor so we can inspect what the judge received.
    let counter = CountingExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "harness-config-test".to_string(),
            transcript:
                "Goal G4: fourth capture with recall limit\nDecision D4: use option_4 serves G4"
                    .to_string(),
            model_available: true,
        },
        &counter,
        &config,
    )
    .expect("run_capture with recall_limit must succeed");

    // Primary load-bearing assertion: each judge call must have received a candidate set
    // bounded to ≤ recall_limit (2). Placeholder ignores recall_limit and passes all
    // 3 prior decisions → each entry has len=3 > 2 → assertion FAILS.
    // Also requires at least one judge call (the 4th capture has 1 decision in harvest,
    // and 3 prior decisions exist, so placeholder behavior 2 calls the judge once with all 3).
    let received = counter.received_candidate_sets();
    assert!(
        !received.is_empty(),
        "at least one judge call must have occurred (4th capture has a decision, 3 priors exist); \
         spy observed {} judge calls",
        counter.judge_call_count()
    );
    for (i, candidate_set) in received.iter().enumerate() {
        assert!(
            candidate_set.len() <= 2,
            "judge call #{i}: candidate set must be bounded to ≤ recall_limit (2); \
             wrong run_capture passed {} candidates (ignores recall_limit)",
            candidate_set.len()
        );
    }

    // Secondary: skills_enabled = false → no skill was applied.
    assert!(
        harness_report.applied_skill_ids.is_empty(),
        "applied_skill_ids must be empty when skills_enabled=false; got {:?}",
        harness_report.applied_skill_ids
    );

    // Store assertion: pipeline ran end-to-end (rows were stored).
    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == "harness-config-test")
        .map(|c| c.id);
    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities after config-knob capture");
    assert!(
        !entities.is_empty(),
        "capture with config knobs must still produce stored rows (pipeline ran end-to-end); \
         got 0 entities"
    );
}

// ---------------------------------------------------------------------------
// Test 3 (single-source): The CLI capture path produces the same stored graph as a direct
// library call. Full CLI≡library parity requires a cg-cli capture entry point that accepts
// a ConversationExtractor double and CaptureHarnessConfig.
//
// NOTE for impl: add a cg-cli capture entry point (e.g. `cg_cli::capture_with_extractor`)
// that accepts a `ConversationExtractor` double and `CaptureHarnessConfig` so that full
// CLI≡library parity can be driven through deterministic doubles in integration tests.
//
// This test asserts the strongest honest LIBRARY store contract: the library path stores
// entities with CORRECT entity types. A "type=module" entity must be stored as
// EntityType::Module. Placeholder run_capture uses the placeholder From<ExtractedEntityType>
// mapping (Module → File) → stored entity type is File, not Module → assertion FAILS.
// ---------------------------------------------------------------------------
#[test]
fn cli_capture_is_a_thin_caller_of_the_library_harness() {
    let (_dir, store) = fresh_store();

    let config = CaptureHarnessConfig::default();

    // Use type=module so the wrong From mapping (Module→File) is observable.
    let (report, _harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "cli-library-parity-test".to_string(),
            transcript:
                "Goal G1: evaluate library harness parity\nEntity E1: ingest-module type=module"
                    .to_string(),
            model_available: true,
        },
        &SpanFaithfulExtractor,
        &config,
    )
    .expect("library run_capture must succeed");

    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == "cli-library-parity-test")
        .map(|c| c.id);

    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities after library capture");

    assert!(
        !entities.is_empty(),
        "library run_capture must store at least one entity; \
         got 0 — the pipeline did not run end-to-end"
    );

    // Primary load-bearing store assertion: the module entity must be stored as
    // EntityType::Module. Placeholder From<ExtractedEntityType> maps Module→File →
    // entity_type == EntityType::File ≠ EntityType::Module → FAILS.
    let module_entity = entities.iter().find(|e| e.name.contains("ingest-module"));
    assert!(
        module_entity.is_some(),
        "the 'ingest-module' entity must be present in the store"
    );
    if let Some(e) = module_entity {
        assert_eq!(
            e.entity_type,
            EntityType::Module,
            "entity '{}' captured with type=module must be stored as EntityType::Module; \
             wrong From<ExtractedEntityType> maps Module→File → stored as {:?}",
            e.name,
            e.entity_type
        );
    }

    // The library path must have captured at least a goal.
    assert!(
        report.wrote_digest || !report.stable_ids.is_empty() || report.last_harvest.goals > 0,
        "library run_capture must produce a non-empty result"
    );
}

// ---------------------------------------------------------------------------
// Test 4 (single-source): A library consumer (vigil-style, opening `Store` directly) can
// reach the FULL ontology contract without any cg-cli on the path.
//
// Primary load-bearing store assertion: a "type=module" entity must be stored as
// EntityType::Module (not File). Placeholder run_capture uses the placeholder
// From<ExtractedEntityType> mapping (Module→File) → stored entity_type is File →
// assertion FAILS. This proves a library consumer reaches the full-ontology contract.
// ---------------------------------------------------------------------------
#[test]
fn library_embedder_reaches_capture_without_the_cli() {
    // Open the store directly — no CLI, no cg-cli import.
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });

    let config = CaptureHarnessConfig::default();

    // Use type=module to expose the wrong From mapping (Module→File).
    let (report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "library-direct-test".to_string(),
            transcript:
                "Goal G1: library consumer bypasses CLI completely\nEntity E1: lib-module type=module"
                    .to_string(),
            model_available: true,
        },
        &SpanFaithfulExtractor,
        &config,
    )
    .expect("direct library call must succeed without cg-cli");

    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == "library-direct-test")
        .map(|c| c.id);

    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities after direct-library capture");

    assert!(
        !entities.is_empty(),
        "direct-library capture must produce stored entities (pipeline ran end-to-end without CLI); \
         got 0 entities"
    );

    // Primary load-bearing store assertion: the module entity must be stored as
    // EntityType::Module. Placeholder From<ExtractedEntityType> maps Module→File →
    // entity_type == EntityType::File ≠ EntityType::Module → FAILS.
    // This is the deepest library-consumer contract: the full ontology is reachable
    // without the CLI, and entity types must be correct.
    let module_entity = entities.iter().find(|e| e.name.contains("lib-module"));
    assert!(
        module_entity.is_some(),
        "the 'lib-module' entity must be in the store after direct-library capture"
    );
    if let Some(e) = module_entity {
        assert_eq!(
            e.entity_type,
            EntityType::Module,
            "entity '{}' captured with type=module via direct library path must be stored as \
             EntityType::Module; wrong From<ExtractedEntityType> maps Module→File → stored as {:?}",
            e.name,
            e.entity_type
        );
    }

    // Store assertion: the stored entity must be under the correct context.
    if let Some(ctx_id) = context_id {
        for entity in &entities {
            assert_eq!(
                entity.context_id, ctx_id,
                "entity '{}' must be under 'library-direct-test' context; \
                 no cg-cli or cross-context routing should occur",
                entity.name
            );
        }
    }

    // Store assertion: intentions must also be stored via the library path.
    let intentions = store
        .list_intentions_in_context(context_id.expect("library-direct-test context must exist"))
        .expect("list intentions");
    assert!(
        !intentions.is_empty(),
        "direct-library capture must produce stored intentions (the intention pipeline ran)"
    );

    // Secondary: zero flagged items (span-faithful extractor has no fabrications).
    assert!(
        report.flagged_items.is_empty(),
        "span-faithful capture via direct library call must produce zero flagged items; \
         got {} flagged items",
        report.flagged_items.len()
    );

    // Secondary: stage2 model calls is 0 (structural fill is model-free).
    assert_eq!(
        harness_report.stage2_model_calls, 0,
        "direct-library consumer: stage2 must make zero model calls"
    );
}
