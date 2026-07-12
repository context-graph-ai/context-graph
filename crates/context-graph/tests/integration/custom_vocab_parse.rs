//! RED test for the S6 smoke catch (2026-07-04): with a USER-SUPPLIED vocabulary configured,
//! the model correctly emits a custom type (e.g. `"entity_type":"lead"`) — and the harvest
//! parser hard-errored the WHOLE extraction ("unknown variant `lead`") instead of routing the
//! value to the raw lane where the configured-vocabulary review flag lives. The
//! contract is "flags for review, never coerced" — erroring the capture is neither. Committed
//! failing before the fix; the fix commit may not touch this file.

use context_graph::ingest::StructuredHarvest;

#[test]
fn custom_vocabulary_entity_type_parses_into_the_raw_lane_never_errors() {
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
        "built-in types keep working exactly as before"
    );
}
