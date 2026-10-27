//! Unknown closed-vocabulary values remain available for review without coercion.

use context_graph::ingest::StructuredHarvest;

#[test]
fn custom_vocabulary_entity_type_parses_into_the_raw_lane_never_errors() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let reply = r#"{
        "summary": "sales session",
        "goals": [{"key": "g1", "text": "close the deal"}],
        "entities": [
            {"key": "e1", "name": "Dana", "entity_type": "lead"},
            {"key": "e2", "name": "src/auth.py", "entity_type": "file"}
        ],
        "decisions": [],
        "observations": [],
        "outcomes": []
    }"#;
    let harvest = StructuredHarvest::from_json_str(reply)
        .expect("a custom-vocabulary entity type must never fail the whole harvest parse");
    assert_eq!(harvest.entities.len(), 2);
    let dana = &harvest.entities[0];
    assert_eq!(
        dana.entity_type_raw.as_deref(),
        Some("lead"),
        "the unknown type string is preserved in the raw lane for the configured-vocabulary \
         review flag — never dropped, never coerced silently"
    );
    let file = &harvest.entities[1];
    assert!(
        file.entity_type_raw.is_none() || file.entity_type_raw.as_deref() == Some("file"),
        "built-in type strings resolve to their declared variants"
    );
}
