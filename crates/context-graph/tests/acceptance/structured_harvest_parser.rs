use context_graph::ingest::StructuredHarvest;
use serde_json::{Value, json};

#[test]
fn structured_harvest_accepts_nullable_top_level_collections_as_empty() {
    let harvest = StructuredHarvest::from_json_str(
        r#"{
            "summary": "live hosted model returned nullable empty collections",
            "goals": null,
            "entities": null,
            "decisions": null,
            "observations": null,
            "outcomes": null
        }"#,
    );
    assert!(
        harvest.is_ok(),
        "hosted structured-output models may emit null for absent top-level arrays; \
         capture must treat those as empty collections rather than failing before the learning loop: {harvest:?}",
    );
    let harvest = harvest.unwrap();

    assert_eq!(
        harvest.summary.as_deref(),
        Some("live hosted model returned nullable empty collections")
    );
    assert!(
        harvest.goals.is_empty()
            && harvest.entities.is_empty()
            && harvest.decisions.is_empty()
            && harvest.observations.is_empty()
            && harvest.outcomes.is_empty(),
        "all nullable top-level collections must deserialize as empty vectors"
    );

    for field in ["goals", "entities", "decisions", "observations", "outcomes"] {
        let raw = sentinel_harvest_with(field, Value::Null);
        let harvest = StructuredHarvest::from_json_str(&raw.to_string());
        assert!(
            harvest.is_ok(),
            "null top-level collection {field} must parse without dropping valid sibling collections: {harvest:?}",
        );
        assert_sentinel_harvest(harvest.unwrap(), field);
    }

    for field in ["goals", "entities", "decisions", "observations", "outcomes"] {
        for malformed_value in [json!("not-an-array"), json!({"not": "array"})] {
            let raw = sentinel_harvest_with(field, malformed_value);
            assert!(
                StructuredHarvest::from_json_str(&raw.to_string()).is_err(),
                "only null is tolerated for top-level collection {field}; malformed non-array values must remain rejected"
            );
        }
    }
}

#[test]
fn structured_harvest_accepts_nullable_nested_decision_collections_as_empty() {
    let harvest = StructuredHarvest::from_json_str(
        &json!({
            "summary": "live hosted model returned nullable nested decision collections",
            "goals": [],
            "entities": [],
            "decisions": [{
                "key": "decision-null-nested",
                "text": "keep decision with nullable nested arrays",
                "reasoning": null,
                "tags": null,
                "decision_type_competing_labels": null,
                "decision_type": "technology_choice",
                "agent_id": "hosted-model"
            }],
            "observations": [],
            "outcomes": []
        })
        .to_string(),
    );
    assert!(
        harvest.is_ok(),
        "hosted structured-output models may emit null for absent nested decision arrays; \
         capture must treat those as empty collections while preserving the decision: {harvest:?}",
    );
    let harvest = harvest.unwrap();

    assert_eq!(
        harvest.decisions.len(),
        1,
        "the decision row must survive nullable nested arrays"
    );
    let decision = &harvest.decisions[0];
    assert_eq!(decision.key, "decision-null-nested");
    assert_eq!(decision.text, "keep decision with nullable nested arrays");
    assert_eq!(
        decision.decision_type.as_ref().map(|field| field.as_str()),
        Some("technology_choice")
    );
    assert_eq!(decision.agent_id.as_deref(), Some("hosted-model"));
    assert_eq!(decision.reasoning, Vec::<String>::new());
    assert_eq!(decision.tags, Vec::<String>::new());
    assert_eq!(
        decision.decision_type_competing_labels,
        Vec::<String>::new()
    );

    for field in ["reasoning", "tags", "decision_type_competing_labels"] {
        let raw = nested_decision_harvest_with(field, Value::Null);
        let harvest = StructuredHarvest::from_json_str(&raw.to_string()).unwrap_or_else(|err| {
            panic!("null nested decision collection {field} must parse without dropping siblings: {err}")
        });
        assert_eq!(harvest.decisions.len(), 1);
        let decision = &harvest.decisions[0];
        assert_eq!(decision.key, "decision-nested-sentinel");
        assert_eq!(decision.text, "preserve sibling nested arrays");
        assert_eq!(
            decision.decision_type.as_ref().map(|field| field.as_str()),
            Some("technology_choice")
        );
        assert_eq!(decision.agent_id.as_deref(), Some("hosted-model"));
        assert_eq!(
            decision.reasoning,
            if field == "reasoning" {
                Vec::new()
            } else {
                strings(&["model supplied reason"])
            },
            "null {field} must not erase sibling reasoning"
        );
        assert_eq!(
            decision.tags,
            if field == "tags" {
                Vec::new()
            } else {
                strings(&["parser", "hosted-output"])
            },
            "null {field} must not erase sibling tags"
        );
        assert_eq!(
            decision.decision_type_competing_labels,
            if field == "decision_type_competing_labels" {
                Vec::new()
            } else {
                strings(&["architecture", "technology_choice"])
            },
            "null {field} must not erase sibling competing labels"
        );
    }

    for field in ["reasoning", "tags", "decision_type_competing_labels"] {
        for malformed_value in [json!("not-an-array"), json!({"not": "array"})] {
            let raw = nested_decision_harvest_with(field, malformed_value);
            assert!(
                StructuredHarvest::from_json_str(&raw.to_string()).is_err(),
                "only null is tolerated for nested decision collection {field}; malformed non-array values must remain rejected"
            );
        }
    }
}

fn nested_decision_harvest_with(field: &str, value: Value) -> Value {
    let mut harvest = json!({
        "summary": "malformed nested decision collection must stay rejected",
        "goals": [],
        "entities": [],
        "decisions": [{
            "key": "decision-nested-sentinel",
            "text": "preserve sibling nested arrays",
            "reasoning": ["model supplied reason"],
            "tags": ["parser", "hosted-output"],
            "decision_type_competing_labels": ["architecture", "technology_choice"],
            "decision_type": "technology_choice",
            "agent_id": "hosted-model"
        }],
        "observations": [],
        "outcomes": []
    });
    let target = harvest
        .get_mut("decisions")
        .and_then(Value::as_array_mut)
        .and_then(|decisions| decisions.first_mut())
        .and_then(Value::as_object_mut)
        .and_then(|decision| decision.get_mut(field))
        .expect("test only replaces a known nested decision collection");
    *target = value;
    harvest
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn sentinel_harvest_with(field: &str, value: Value) -> Value {
    let mut harvest = json!({
        "summary": "nullable collection preserves valid sibling rows",
        "goals": [{"key": "goal-sentinel", "text": "keep goal"}],
        "entities": [{
            "key": "entity-sentinel",
            "name": "context-graph",
            "path": "crates/context-graph",
            "entity_type": "repository"
        }],
        "decisions": [{
            "key": "decision-sentinel",
            "text": "keep decision",
            "goal_key": "goal-sentinel",
            "entity_key": "entity-sentinel",
            "decision_type": "technology_choice",
            "agent_id": "hosted-model",
            "reasoning": ["preserve sibling decision"],
            "tags": ["parser"],
            "properties": {"preserved": true}
        }],
        "observations": [{
            "key": "observation-sentinel",
            "text": "keep observation",
            "entity_key": "entity-sentinel",
            "observation_type": "benchmark",
            "source": "agent-session",
            "observed_properties": {"latency_ms": 42},
            "state_delta": {"status": "kept"}
        }],
        "outcomes": [{
            "key": "outcome-sentinel",
            "of_decision_key": "decision-sentinel",
            "success": true,
            "notes": "kept outcome"
        }]
    });
    let target = harvest
        .as_object_mut()
        .and_then(|object| object.get_mut(field))
        .expect("test only replaces a known top-level harvest collection");
    *target = value;
    harvest
}

fn assert_sentinel_harvest(harvest: StructuredHarvest, nulled_field: &str) {
    assert_eq!(
        harvest.summary.as_deref(),
        Some("nullable collection preserves valid sibling rows")
    );

    let goal_keys = harvest
        .goals
        .iter()
        .map(|goal| (goal.key.as_str(), goal.text.as_str()))
        .collect::<Vec<_>>();
    if nulled_field == "goals" {
        assert_eq!(goal_keys, Vec::<(&str, &str)>::new());
    } else {
        assert_eq!(goal_keys, vec![("goal-sentinel", "keep goal")]);
    }

    let entity_keys = harvest
        .entities
        .iter()
        .map(|entity| {
            (
                entity.key.as_str(),
                entity.name.as_str(),
                entity.path.as_str(),
            )
        })
        .collect::<Vec<_>>();
    if nulled_field == "entities" {
        assert_eq!(entity_keys, Vec::<(&str, &str, &str)>::new());
    } else {
        assert_eq!(
            entity_keys,
            vec![("entity-sentinel", "context-graph", "crates/context-graph")]
        );
    }

    let decision_keys = harvest
        .decisions
        .iter()
        .map(|decision| {
            (
                decision.key.as_str(),
                decision.text.as_str(),
                decision.goal_key.as_deref(),
                decision.entity_key.as_deref(),
                decision.decision_type.as_ref().map(|field| field.as_str()),
                decision.agent_id.as_deref(),
            )
        })
        .collect::<Vec<_>>();
    if nulled_field == "decisions" {
        assert_eq!(
            decision_keys,
            Vec::<(
                &str,
                &str,
                Option<&str>,
                Option<&str>,
                Option<&str>,
                Option<&str>
            )>::new()
        );
    } else {
        assert_eq!(
            decision_keys,
            vec![(
                "decision-sentinel",
                "keep decision",
                Some("goal-sentinel"),
                Some("entity-sentinel"),
                Some("technology_choice"),
                Some("hosted-model"),
            )]
        );
    }

    let observation_keys = harvest
        .observations
        .iter()
        .map(|observation| {
            (
                observation.key.as_str(),
                observation.text.as_str(),
                observation.entity_key.as_deref(),
                observation.observation_type.as_deref(),
                observation.source.as_deref(),
                observation.observed_properties["latency_ms"].as_i64(),
                observation.state_delta["status"].as_str(),
            )
        })
        .collect::<Vec<_>>();
    if nulled_field == "observations" {
        assert_eq!(
            observation_keys,
            Vec::<(
                &str,
                &str,
                Option<&str>,
                Option<&str>,
                Option<&str>,
                Option<i64>,
                Option<&str>,
            )>::new()
        );
    } else {
        assert_eq!(
            observation_keys,
            vec![(
                "observation-sentinel",
                "keep observation",
                Some("entity-sentinel"),
                Some("benchmark"),
                Some("agent-session"),
                Some(42),
                Some("kept"),
            )]
        );
    }

    let outcome_keys = harvest
        .outcomes
        .iter()
        .map(|outcome| {
            (
                outcome.key.as_str(),
                outcome.of_decision_key.as_deref(),
                outcome.success,
                outcome.notes.as_str(),
            )
        })
        .collect::<Vec<_>>();
    if nulled_field == "outcomes" {
        assert_eq!(outcome_keys, Vec::<(&str, Option<&str>, bool, &str)>::new());
    } else {
        assert_eq!(
            outcome_keys,
            vec![(
                "outcome-sentinel",
                Some("decision-sentinel"),
                true,
                "kept outcome"
            )]
        );
    }
}
