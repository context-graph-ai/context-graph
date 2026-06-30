//! Provider JSON Schema coverage (integration test).
//!
//! Asserts the single-source `SchemaDescriptor` renders a provider JSON Schema and prompt
//! shape-guidance that cover every parser-consumed full-extraction field, so a strict Native
//! provider is asked for the whole ontology shape and no field can drift between the prompt,
//! the wire schema, and the parser.

use context_graph::harness::CaptureHarnessConfig;
use serde_json::Value;
use std::collections::BTreeSet;

#[test]
fn provider_schema_and_prompt_guidance_cover_full_seed_shape() {
    let descriptor = CaptureHarnessConfig::default()
        .schema_descriptor()
        .expect("default capture config must expose a schema descriptor");
    let schema = descriptor.render_output_json_schema();

    assert_harvest_item_has_fields(
        &schema,
        "decisions",
        &[
            "key",
            "text",
            "goal_key",
            "entity_key",
            "relation",
            "decision_type",
            "agent_id",
            "confidence",
            "reasoning",
            "tags",
            "properties",
            "decision_type_span",
            "decision_type_competing_labels",
            "agent_id_span",
        ],
    );
    assert_harvest_item_has_fields(
        &schema,
        "observations",
        &[
            "key",
            "text",
            "entity_key",
            "path_change",
            "observation_type",
            "source",
            "observed_properties",
            "state_delta",
            "value_span",
        ],
    );
    assert_harvest_item_has_fields(
        &schema,
        "outcomes",
        &["key", "of_decision_key", "success", "notes", "value_span"],
    );

    let decision_properties = harvest_item_properties(&schema, "decisions");
    assert_schema_ref(
        decision_properties
            .get("relation")
            .expect("decision relation schema must exist"),
        "#/$defs/decision_relation",
        "decision relation",
    );
    assert_schema_ref(
        decision_properties
            .get("decision_type_span")
            .expect("decision_type_span schema must exist"),
        "#/$defs/extracted_span",
        "decision_type_span",
    );
    assert_schema_ref(
        decision_properties
            .get("agent_id_span")
            .expect("agent_id_span schema must exist"),
        "#/$defs/extracted_span",
        "agent_id_span",
    );
    assert_nullable_span_schema(
        schema
            .pointer("/$defs/extracted_span")
            .expect("extracted_span definition must exist"),
        "extracted_span definition",
    );
    assert_json_object_lane(
        decision_properties
            .get("properties")
            .expect("decision properties schema must exist"),
        "decision properties",
    );
    assert!(
        !decision_properties.contains_key("context_id"),
        "provider schema must not admit model-emitted context_id on decisions"
    );

    let observation_properties = harvest_item_properties(&schema, "observations");
    assert_schema_ref(
        observation_properties
            .get("value_span")
            .expect("observation value_span schema must exist"),
        "#/$defs/extracted_span",
        "observation value_span",
    );
    for field in ["observed_properties", "state_delta"] {
        assert_json_object_lane(
            observation_properties
                .get(field)
                .unwrap_or_else(|| panic!("{field} schema must exist")),
            field,
        );
    }
    assert!(
        !observation_properties.contains_key("context_id"),
        "provider schema must not admit model-emitted context_id on observations"
    );

    let outcome_properties = harvest_item_properties(&schema, "outcomes");
    assert_schema_ref(
        outcome_properties
            .get("value_span")
            .expect("outcome value_span schema must exist"),
        "#/$defs/extracted_span",
        "outcome value_span",
    );
    assert!(
        !outcome_properties.contains_key("context_id"),
        "provider schema must not admit model-emitted context_id on outcomes"
    );

    let guidance = descriptor.render_prompt_shape_guidance();
    for field in [
        "decision_type_span",
        "agent_id_span",
        "decision_type_competing_labels",
        "agent_id",
        "confidence",
        "tags",
        "properties",
        "observation_type",
        "source",
        "observed_properties",
        "state_delta",
        "value_span",
    ] {
        assert!(
            guidance.contains(field),
            "prompt shape guidance must name parser-consumed field `{field}`; guidance was: {guidance}"
        );
    }
}

fn harvest_item_properties<'a>(
    schema: &'a Value,
    harvest_field: &str,
) -> &'a serde_json::Map<String, Value> {
    schema
        .pointer(&format!("/properties/{harvest_field}/items/properties"))
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("{harvest_field} item properties must be an object"))
}

fn harvest_item_required(schema: &Value, harvest_field: &str) -> BTreeSet<String> {
    schema
        .pointer(&format!("/properties/{harvest_field}/items/required"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn assert_harvest_item_has_fields(schema: &Value, harvest_field: &str, fields: &[&str]) {
    let properties = harvest_item_properties(schema, harvest_field);
    let required = harvest_item_required(schema, harvest_field);
    for field in fields {
        assert!(
            properties.contains_key(*field),
            "{harvest_field} provider schema is missing parser-consumed field `{field}`"
        );
        assert!(
            required.contains(*field),
            "{harvest_field} provider schema must require `{field}` so strict Native providers emit it"
        );
    }
}

fn assert_nullable_span_schema(schema: &Value, field: &str) {
    assert_type_allows(schema, "object", field);
    assert_type_allows(schema, "null", field);
    assert_eq!(
        schema.get("additionalProperties"),
        Some(&Value::Bool(false)),
        "{field} span schema must reject unknown fields"
    );
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("{field} span properties must be an object"));
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    for span_field in ["text", "start", "end"] {
        assert!(
            properties.contains_key(span_field),
            "{field} span schema is missing `{span_field}`"
        );
        assert!(
            required.contains(span_field),
            "{field} span schema must require `{span_field}` with null allowed"
        );
    }
}

fn assert_schema_ref(schema: &Value, expected_ref: &str, field: &str) {
    assert_eq!(
        schema.get("$ref").and_then(Value::as_str),
        Some(expected_ref),
        "{field} schema must point at shared definition {expected_ref}; schema was: {schema}"
    );
}

fn assert_type_allows(schema: &Value, expected_type: &str, field: &str) {
    let type_value = schema
        .get("type")
        .unwrap_or_else(|| panic!("{field} schema must declare a type"));
    let allows = match type_value {
        Value::String(type_name) => type_name == expected_type,
        Value::Array(type_names) => type_names
            .iter()
            .any(|type_name| type_name.as_str() == Some(expected_type)),
        _ => false,
    };
    assert!(
        allows,
        "{field} schema must allow JSON type `{expected_type}`; schema was: {schema}"
    );
}

fn assert_json_object_lane(schema: &Value, field: &str) {
    assert_type_allows(schema, "object", field);
    let additional = schema
        .get("additionalProperties")
        .unwrap_or_else(|| panic!("{field} lane must declare bounded map values"));
    for scalar_type in ["string", "number", "boolean", "null"] {
        assert_type_allows(additional, scalar_type, field);
    }
}
