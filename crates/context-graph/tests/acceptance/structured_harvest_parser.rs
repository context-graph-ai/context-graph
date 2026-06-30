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
