use std::any::type_name_of_val;
use std::collections::HashMap;

use cg_core::*;
use chrono::{TimeZone, Utc};
use serde_json::{Value, json};
use uuid::Uuid;

fn ts(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
        .single()
        .expect("valid test timestamp")
}

fn map(entries: &[(&str, Value)]) -> HashMap<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn label_map(entries: &[(&str, &str)]) -> HashMap<String, String> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

fn decision_id() -> DecisionId {
    DecisionId::from(Uuid::from_u128(1))
}

fn entity_id() -> EntityId {
    EntityId::from(Uuid::from_u128(2))
}

fn observation_id() -> ObservationId {
    ObservationId::from(Uuid::from_u128(3))
}

fn signature_id() -> SignatureId {
    SignatureId::from(Uuid::from_u128(6))
}

fn context_id() -> ContextId {
    ContextId::from(Uuid::from_u128(8))
}

fn intention_id() -> IntentionId {
    IntentionId::from(Uuid::from_u128(9))
}

fn sample_reasoning_step() -> ReasoningStep {
    ReasoningStep::new(
        "analyzed".into(),
        "compared latency distributions".into(),
        "threshold too low".into(),
        Some("agent-1".into()),
        ts(2026, 1, 2, 3, 4, 5),
    )
}

fn sample_intention() -> Intention {
    Intention::new(
        "Protect children from unauthorized pickup".into(),
        context_id(),
        IntentionOrigin::Human,
        None,
        None,
        None,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
}

fn sample_decision() -> Decision {
    Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.7),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .expect("constructor returns Ok in scaffold")
}

fn sample_signature() -> Signature {
    Signature::try_new(
        "frequency_anomaly".into(),
        "Same person at multiple locations".into(),
        map(&[
            ("observation_types", json!(["person_detected"])),
            ("similarity_threshold", json!(0.82)),
        ]),
        label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]),
        0.75,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .expect("constructor returns Ok in scaffold")
}

#[test]
fn test_intention_status_has_exactly_four_variants() {
    let variants = IntentionStatus::ALL;
    assert_eq!(variants.len(), 4);
    assert_eq!(
        &variants[..],
        &[
            IntentionStatus::Active,
            IntentionStatus::Paused,
            IntentionStatus::Superseded,
            IntentionStatus::Archived,
        ],
    );
    for variant in variants {
        match variant {
            IntentionStatus::Active => {}
            IntentionStatus::Paused => {}
            IntentionStatus::Superseded => {}
            IntentionStatus::Archived => {}
        }
    }
}

#[test]
fn test_decision_status_has_exactly_three_variants() {
    let variants = DecisionStatus::ALL;
    assert_eq!(variants.len(), 3);
    assert_eq!(
        &variants[..],
        &[
            DecisionStatus::Active,
            DecisionStatus::Superseded,
            DecisionStatus::Invalidated,
        ],
    );
    for variant in variants {
        match variant {
            DecisionStatus::Active => {}
            DecisionStatus::Superseded => {}
            DecisionStatus::Invalidated => {}
        }
    }
}

#[test]
fn test_invalidation_status_has_exactly_four_variants() {
    let variants = InvalidationStatus::ALL;
    assert_eq!(variants.len(), 4);
    assert_eq!(
        &variants[..],
        &[
            InvalidationStatus::Pending,
            InvalidationStatus::Acknowledged,
            InvalidationStatus::Dismissed,
            InvalidationStatus::Resolved,
        ],
    );
    for variant in variants {
        match variant {
            InvalidationStatus::Pending => {}
            InvalidationStatus::Acknowledged => {}
            InvalidationStatus::Dismissed => {}
            InvalidationStatus::Resolved => {}
        }
    }
}

#[test]
fn test_severity_has_exactly_three_variants() {
    let variants = Severity::ALL;
    assert_eq!(variants.len(), 3);
    assert_eq!(
        &variants[..],
        &[Severity::Info, Severity::Warning, Severity::Critical],
    );
    for variant in variants {
        match variant {
            Severity::Info => {}
            Severity::Warning => {}
            Severity::Critical => {}
        }
    }
}

#[test]
fn test_signature_status_has_exactly_two_variants() {
    let variants = SignatureStatus::ALL;
    assert_eq!(variants.len(), 2);
    assert_eq!(
        &variants[..],
        &[SignatureStatus::Active, SignatureStatus::Retired],
    );
    for variant in variants {
        match variant {
            SignatureStatus::Active => {}
            SignatureStatus::Retired => {}
        }
    }
}

#[test]
fn test_blueprint_status_has_exactly_two_variants() {
    let variants = BlueprintStatus::ALL;
    assert_eq!(variants.len(), 2);
    assert_eq!(
        &variants[..],
        &[BlueprintStatus::Active, BlueprintStatus::Deprecated],
    );
    for variant in variants {
        match variant {
            BlueprintStatus::Active => {}
            BlueprintStatus::Deprecated => {}
        }
    }
}

#[test]
fn test_intention_origin_has_exactly_four_variants() {
    let variants = IntentionOrigin::ALL;
    assert_eq!(variants.len(), 4);
    assert_eq!(
        &variants[..],
        &[
            IntentionOrigin::Human,
            IntentionOrigin::Agent,
            IntentionOrigin::Signature,
            IntentionOrigin::Blueprint,
        ],
    );
    for variant in variants {
        match variant {
            IntentionOrigin::Human => {}
            IntentionOrigin::Agent => {}
            IntentionOrigin::Signature => {}
            IntentionOrigin::Blueprint => {}
        }
    }
}

#[test]
fn test_relationship_type_has_exactly_fifteen_variants() {
    let variants = RelationshipType::ALL;
    assert_eq!(variants.len(), 15);
    assert_eq!(
        &variants[..],
        &[
            RelationshipType::Serves,
            RelationshipType::BasedOn,
            RelationshipType::About,
            RelationshipType::Cites,
            RelationshipType::Supersedes,
            RelationshipType::RelatesTo,
            RelationshipType::ObservedOn,
            RelationshipType::HasOutcome,
            RelationshipType::Invalidates,
            RelationshipType::TriggeredBy,
            RelationshipType::MatchedBy,
            RelationshipType::Generated,
            RelationshipType::ExtractedFrom,
            RelationshipType::Instantiates,
            RelationshipType::Refines,
        ],
    );
    for variant in variants {
        match variant {
            RelationshipType::Serves => {}
            RelationshipType::BasedOn => {}
            RelationshipType::About => {}
            RelationshipType::Cites => {}
            RelationshipType::Supersedes => {}
            RelationshipType::RelatesTo => {}
            RelationshipType::ObservedOn => {}
            RelationshipType::HasOutcome => {}
            RelationshipType::Invalidates => {}
            RelationshipType::TriggeredBy => {}
            RelationshipType::MatchedBy => {}
            RelationshipType::Generated => {}
            RelationshipType::ExtractedFrom => {}
            RelationshipType::Instantiates => {}
            RelationshipType::Refines => {}
        }
    }
}

#[test]
fn test_intention_default_status_is_active() {
    let intention = sample_intention();
    assert_eq!(intention.status, IntentionStatus::Active);
}

#[test]
fn test_decision_default_status_is_active() {
    let decision = sample_decision();
    assert_eq!(decision.status, DecisionStatus::Active);
}

#[test]
fn test_invalidation_default_status_is_pending() {
    let invalidation = Invalidation::new(
        decision_id(),
        observation_id(),
        map(&[("entity_id", json!(entity_id().to_string()))]),
        Severity::Warning,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(invalidation.status, InvalidationStatus::Pending);
}

#[test]
fn test_signature_default_status_is_active() {
    let signature = sample_signature();
    assert_eq!(signature.status, SignatureStatus::Active);
}

#[test]
fn test_blueprint_default_status_is_active() {
    let blueprint = Blueprint::new(
        "Detect entity at location during time window".into(),
        "detection".into(),
        map(&[("entity_type", json!("SERVICE"))]),
        None,
        "1.0.0".into(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(blueprint.status, BlueprintStatus::Active);
}

#[test]
fn test_intention_requires_context_id() {
    let provided = context_id();
    let intention = Intention::new(
        "Protect children from unauthorized pickup".into(),
        provided,
        IntentionOrigin::Human,
        None,
        None,
        None,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(intention.context_id, provided);
}

#[test]
fn test_decision_requires_context_id() {
    let provided = context_id();
    let decision = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.7),
        None,
        provided,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(decision.context_id, provided);
}

#[test]
fn test_entity_requires_context_id() {
    let provided = context_id();
    let entity = Entity::new(
        "SERVICE".into(),
        "payment-service".into(),
        provided,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(entity.context_id, provided);
}

#[test]
fn test_observation_requires_context_id() {
    let provided = context_id();
    let observation = Observation::new(
        "config_change".into(),
        map(&[("path", json!("/etc/app/config.toml"))]),
        "git-hook".into(),
        entity_id(),
        provided,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(observation.context_id, provided);
}

#[test]
fn test_digest_requires_context_id() {
    let provided = context_id();
    let digest = Digest::new(
        "claude-cli".into(),
        None,
        "Discussed auth module refactoring".into(),
        vec![0.1, 0.2, 0.3],
        provided,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(digest.context_id, provided);
}

#[test]
fn test_intention_id_not_interchangeable_with_decision_id() {
    let uuid = Uuid::from_u128(11);
    let intention_id = IntentionId::from(uuid);
    let decision_id = DecisionId::from(uuid);
    assert_ne!(
        type_name_of_val(&intention_id),
        type_name_of_val(&decision_id)
    );
}

#[test]
fn test_entity_id_not_interchangeable_with_observation_id() {
    let uuid = Uuid::from_u128(12);
    let entity_id = EntityId::from(uuid);
    let observation_id = ObservationId::from(uuid);
    assert_ne!(
        type_name_of_val(&entity_id),
        type_name_of_val(&observation_id)
    );
}

#[test]
fn test_confidence_bounded_zero_to_one() {
    let valid_zero = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.0),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(valid_zero.confidence, Some(0.0));

    let valid_one = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(1.0),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(valid_one.confidence, Some(1.0));

    let valid_mid = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.5),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(valid_mid.confidence, Some(0.5));

    assert!(
        Decision::try_new(
            "architecture".into(),
            "Use abstraction layer for fault isolation".into(),
            None,
            Some(1.1),
            None,
            context_id(),
            None,
            None,
            ts(2026, 1, 2, 3, 4, 5),
        )
        .is_err()
    );

    assert!(
        Decision::try_new(
            "architecture".into(),
            "Use abstraction layer for fault isolation".into(),
            None,
            Some(-0.1),
            None,
            context_id(),
            None,
            None,
            ts(2026, 1, 2, 3, 4, 5),
        )
        .is_err()
    );
}

#[test]
fn test_signature_confidence_bounded_zero_to_one() {
    let valid_zero = Signature::try_new(
        "frequency_anomaly".into(),
        "Same person at multiple locations".into(),
        map(&[
            ("observation_types", json!(["person_detected"])),
            ("similarity_threshold", json!(0.82)),
        ]),
        label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]),
        0.0,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(valid_zero.confidence, 0.0);

    let valid_one = Signature::try_new(
        "frequency_anomaly".into(),
        "Same person at multiple locations".into(),
        map(&[
            ("observation_types", json!(["person_detected"])),
            ("similarity_threshold", json!(0.82)),
        ]),
        label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]),
        1.0,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(valid_one.confidence, 1.0);

    assert!(
        Signature::try_new(
            "frequency_anomaly".into(),
            "Same person at multiple locations".into(),
            map(&[
                ("observation_types", json!(["person_detected"])),
                ("similarity_threshold", json!(0.82)),
            ]),
            label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]),
            1.5,
            None,
            None,
            ts(2026, 1, 2, 3, 4, 5),
        )
        .is_err()
    );

    assert!(
        Signature::try_new(
            "frequency_anomaly".into(),
            "Same person at multiple locations".into(),
            map(&[
                ("observation_types", json!(["person_detected"])),
                ("similarity_threshold", json!(0.82)),
            ]),
            label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]),
            -0.01,
            None,
            None,
            ts(2026, 1, 2, 3, 4, 5),
        )
        .is_err()
    );
}

#[test]
fn test_decision_confidence_is_optional() {
    let decision = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        None,
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(decision.confidence, None);
}

#[test]
fn test_signature_confidence_is_required() {
    let signature = Signature::try_new(
        "frequency_anomaly".into(),
        "Same person at multiple locations".into(),
        map(&[
            ("observation_types", json!(["person_detected"])),
            ("similarity_threshold", json!(0.82)),
        ]),
        label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]),
        0.75,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(signature.confidence, 0.75);
}

#[test]
fn test_intention_has_created_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let intention = Intention::new(
        "Protect children from unauthorized pickup".into(),
        context_id(),
        IntentionOrigin::Human,
        None,
        None,
        None,
        None,
        None,
        None,
        provided,
    );
    assert_eq!(intention.created_at, provided);
}

#[test]
fn test_decision_has_created_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let decision = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.7),
        None,
        context_id(),
        None,
        None,
        provided,
    )
    .unwrap();
    assert_eq!(decision.created_at, provided);
}

#[test]
fn test_observation_has_timestamp() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let observation = Observation::new(
        "config_change".into(),
        map(&[("path", json!("/etc/app/config.toml"))]),
        "git-hook".into(),
        entity_id(),
        context_id(),
        provided,
    );
    assert_eq!(observation.timestamp, provided);
}

#[test]
fn test_invalidation_has_detected_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let invalidation = Invalidation::new(
        decision_id(),
        observation_id(),
        map(&[("entity_id", json!(entity_id().to_string()))]),
        Severity::Warning,
        None,
        None,
        None,
        provided,
    );
    assert_eq!(invalidation.detected_at, provided);
}

#[test]
fn test_outcome_has_recorded_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let outcome = Outcome::new(decision_id(), true, None, provided);
    assert_eq!(outcome.recorded_at, provided);
}

#[test]
fn test_outcome_requires_decision_id() {
    let provided = decision_id();
    let outcome = Outcome::new(provided, true, None, ts(2026, 1, 2, 3, 4, 5));
    assert_eq!(outcome.decision_id, provided);
}

#[test]
fn test_invalidation_requires_affected_decision_id() {
    let provided = decision_id();
    let invalidation = Invalidation::new(
        provided,
        observation_id(),
        map(&[("entity_id", json!(entity_id().to_string()))]),
        Severity::Warning,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(invalidation.affected_decision_id, provided);
}

#[test]
fn test_invalidation_requires_trigger_observation_id() {
    let provided = observation_id();
    let invalidation = Invalidation::new(
        decision_id(),
        provided,
        map(&[("entity_id", json!(entity_id().to_string()))]),
        Severity::Warning,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(invalidation.trigger_observation_id, provided);
}

#[test]
fn test_invalidation_requires_basis_diff() {
    let basis_diff = map(&[
        ("entity_id", json!(entity_id().to_string())),
        ("field", json!("status")),
        ("old_value", json!("active")),
        ("new_value", json!("archived")),
    ]);
    let invalidation = Invalidation::new(
        decision_id(),
        observation_id(),
        basis_diff.clone(),
        Severity::Warning,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(invalidation.basis_diff, basis_diff);
}

#[test]
fn test_invalidation_requires_severity() {
    let invalidation = Invalidation::new(
        decision_id(),
        observation_id(),
        map(&[
            ("entity_id", json!(entity_id().to_string())),
            ("field", json!("status")),
            ("old_value", json!("active")),
            ("new_value", json!("archived")),
        ]),
        Severity::Warning,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(invalidation.severity, Severity::Warning);
}

#[test]
fn test_observation_requires_entity_id() {
    let provided = entity_id();
    let observation = Observation::new(
        "config_change".into(),
        map(&[("path", json!("/etc/app/config.toml"))]),
        "git-hook".into(),
        provided,
        context_id(),
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(observation.entity_id, provided);
}

#[test]
fn test_observation_requires_source() {
    let observation = Observation::new(
        "config_change".into(),
        map(&[("path", json!("/etc/app/config.toml"))]),
        "git-hook".into(),
        entity_id(),
        context_id(),
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(observation.source, "git-hook");
}

#[test]
fn test_observation_requires_data() {
    let data = map(&[
        ("path", json!("/etc/app/config.toml")),
        ("status", json!("changed")),
    ]);
    let observation = Observation::new(
        "config_change".into(),
        data.clone(),
        "git-hook".into(),
        entity_id(),
        context_id(),
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(observation.data, data);
}

#[test]
fn test_digest_requires_source() {
    let digest = Digest::new(
        "claude-cli".into(),
        None,
        "Discussed auth module refactoring".into(),
        vec![0.1, 0.2, 0.3],
        context_id(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(digest.source, "claude-cli");
}

#[test]
fn test_digest_requires_summary() {
    let digest = Digest::new(
        "claude-cli".into(),
        None,
        "Discussed auth module refactoring".into(),
        vec![0.1, 0.2, 0.3],
        context_id(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(digest.summary, "Discussed auth module refactoring");
}

#[test]
fn test_digest_requires_embedding() {
    let digest = Digest::new(
        "claude-cli".into(),
        None,
        "Discussed auth module refactoring".into(),
        vec![0.1, 0.2, 0.3],
        context_id(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(digest.embedding, vec![0.1, 0.2, 0.3]);
}

#[test]
fn test_blueprint_requires_slots() {
    let slots = map(&[("entity_type", json!("SERVICE"))]);
    let blueprint = Blueprint::new(
        "Detect entity at location during time window".into(),
        "detection".into(),
        slots.clone(),
        None,
        "1.0.0".into(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(blueprint.slots, slots);
}

#[test]
fn test_blueprint_requires_version() {
    let blueprint = Blueprint::new(
        "Detect entity at location during time window".into(),
        "detection".into(),
        map(&[("entity_type", json!("SERVICE"))]),
        None,
        "1.0.0".into(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(blueprint.version, "1.0.0");
}

#[test]
fn test_context_requires_name() {
    let context = Context::new("client-a".into(), None, None, ts(2026, 1, 2, 3, 4, 5));
    assert_eq!(context.name, "client-a");
}

#[test]
fn test_relationship_has_valid_from() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let relationship = Relationship::new(
        RelationshipEndpoint::Decision(decision_id()),
        RelationshipEndpoint::Intention(intention_id()),
        RelationshipType::Serves,
        provided,
        None,
    );
    assert_eq!(relationship.valid_from, provided);
}

#[test]
fn test_relationship_valid_to_is_optional() {
    let without_valid_to = Relationship::new(
        RelationshipEndpoint::Decision(decision_id()),
        RelationshipEndpoint::Intention(intention_id()),
        RelationshipType::Serves,
        ts(2026, 1, 2, 3, 4, 5),
        None,
    );
    assert_eq!(without_valid_to.valid_to, None);

    let provided = ts(2026, 1, 2, 4, 4, 5);
    let with_valid_to = Relationship::new(
        RelationshipEndpoint::Decision(decision_id()),
        RelationshipEndpoint::Intention(intention_id()),
        RelationshipType::Serves,
        ts(2026, 1, 2, 3, 4, 5),
        Some(provided),
    );
    assert_eq!(with_valid_to.valid_to, Some(provided));
}

#[test]
fn test_relationship_connects_typed_ids() {
    let source = RelationshipEndpoint::Decision(decision_id());
    let target = RelationshipEndpoint::Intention(intention_id());
    let relationship = Relationship::new(
        source,
        target,
        RelationshipType::Serves,
        ts(2026, 1, 2, 3, 4, 5),
        None,
    );

    assert_eq!(relationship.source, source);
    assert_eq!(relationship.target, target);
    assert_ne!(
        type_name_of_val(&relationship.source),
        type_name_of_val(&Uuid::nil())
    );
    assert_ne!(
        type_name_of_val(&relationship.target),
        type_name_of_val(&Uuid::nil())
    );
}

#[test]
fn test_intention_origin_signature_allows_origin_ref() {
    let provided = signature_id();
    let intention = Intention::new(
        "Protect children from unauthorized pickup".into(),
        context_id(),
        IntentionOrigin::Signature,
        Some(provided),
        None,
        None,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(intention.origin_ref, Some(provided));
}

#[test]
fn test_intention_blueprint_fields_are_optional() {
    let intention = sample_intention();
    assert_eq!(intention.blueprint_id, None);
    assert_eq!(intention.bindings, None);
    assert_eq!(intention.blueprint_proposal, None);
}

#[test]
fn test_invalidation_resolved_at_is_optional() {
    let invalidation = Invalidation::new(
        decision_id(),
        observation_id(),
        map(&[
            ("entity_id", json!(entity_id().to_string())),
            ("field", json!("status")),
            ("old_value", json!("active")),
            ("new_value", json!("archived")),
        ]),
        Severity::Warning,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(invalidation.resolved_at, None);
}

#[test]
fn test_invalidation_resolution_fields_are_optional() {
    let invalidation = Invalidation::new(
        decision_id(),
        observation_id(),
        map(&[
            ("entity_id", json!(entity_id().to_string())),
            ("field", json!("status")),
            ("old_value", json!("active")),
            ("new_value", json!("archived")),
        ]),
        Severity::Warning,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(invalidation.resolution_action, None);
    assert_eq!(invalidation.resolution_decision_id, None);
}

#[test]
fn test_intention_requires_description() {
    let intention = Intention::new(
        "Protect children from unauthorized pickup".into(),
        context_id(),
        IntentionOrigin::Human,
        None,
        None,
        None,
        None,
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(
        intention.description,
        "Protect children from unauthorized pickup"
    );
}

#[test]
fn test_decision_requires_decision_type() {
    let decision = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.7),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(decision.decision_type, "architecture");
}

#[test]
fn test_decision_requires_description() {
    let decision = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.7),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(
        decision.description,
        "Use abstraction layer for fault isolation"
    );
}

#[test]
fn test_entity_requires_entity_type() {
    let entity = Entity::new(
        "SERVICE".into(),
        "payment-service".into(),
        context_id(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(entity.entity_type, "SERVICE");
}

#[test]
fn test_entity_requires_name() {
    let entity = Entity::new(
        "SERVICE".into(),
        "payment-service".into(),
        context_id(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(entity.name, "payment-service");
}

#[test]
fn test_entity_has_created_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let entity = Entity::new(
        "SERVICE".into(),
        "payment-service".into(),
        context_id(),
        None,
        provided,
    );
    assert_eq!(entity.created_at, provided);
}

#[test]
fn test_observation_requires_observation_type() {
    let observation = Observation::new(
        "config_change".into(),
        map(&[("path", json!("/etc/app/config.toml"))]),
        "git-hook".into(),
        entity_id(),
        context_id(),
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(observation.observation_type, "config_change");
}

#[test]
fn test_outcome_requires_success() {
    let true_outcome = Outcome::new(decision_id(), true, None, ts(2026, 1, 2, 3, 4, 5));
    assert!(true_outcome.success);

    let false_outcome = Outcome::new(decision_id(), false, None, ts(2026, 1, 2, 3, 4, 5));
    assert!(!false_outcome.success);
}

#[test]
fn test_signature_requires_signature_type() {
    let signature = sample_signature();
    assert_eq!(signature.signature_type, "frequency_anomaly");
}

#[test]
fn test_signature_requires_description() {
    let signature = sample_signature();
    assert_eq!(signature.description, "Same person at multiple locations");
}

#[test]
fn test_signature_requires_match_criteria() {
    let expected = map(&[
        ("observation_types", json!(["person_detected"])),
        ("similarity_threshold", json!(0.82)),
    ]);
    let signature = sample_signature();
    assert_eq!(signature.match_criteria, expected);
}

#[test]
fn test_signature_requires_scope_labels() {
    let expected = label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]);
    let signature = sample_signature();
    assert_eq!(signature.scope_labels, expected);
}

#[test]
fn test_signature_has_created_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let signature = Signature::try_new(
        "frequency_anomaly".into(),
        "Same person at multiple locations".into(),
        map(&[
            ("observation_types", json!(["person_detected"])),
            ("similarity_threshold", json!(0.82)),
        ]),
        label_map(&[("org", "bank-x"), ("use_case", "atm_monitoring")]),
        0.75,
        None,
        None,
        provided,
    )
    .unwrap();
    assert_eq!(signature.created_at, provided);
}

#[test]
fn test_digest_has_created_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let digest = Digest::new(
        "claude-cli".into(),
        None,
        "Discussed auth module refactoring".into(),
        vec![0.1, 0.2, 0.3],
        context_id(),
        None,
        provided,
    );
    assert_eq!(digest.created_at, provided);
}

#[test]
fn test_blueprint_requires_canonical_description() {
    let blueprint = Blueprint::new(
        "Detect entity at location during time window".into(),
        "detection".into(),
        map(&[("entity_type", json!("SERVICE"))]),
        None,
        "1.0.0".into(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(
        blueprint.canonical_description,
        "Detect entity at location during time window"
    );
}

#[test]
fn test_blueprint_requires_category() {
    let blueprint = Blueprint::new(
        "Detect entity at location during time window".into(),
        "detection".into(),
        map(&[("entity_type", json!("SERVICE"))]),
        None,
        "1.0.0".into(),
        None,
        ts(2026, 1, 2, 3, 4, 5),
    );
    assert_eq!(blueprint.category, "detection");
}

#[test]
fn test_blueprint_has_created_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let blueprint = Blueprint::new(
        "Detect entity at location during time window".into(),
        "detection".into(),
        map(&[("entity_type", json!("SERVICE"))]),
        None,
        "1.0.0".into(),
        None,
        provided,
    );
    assert_eq!(blueprint.created_at, provided);
}

#[test]
fn test_context_has_created_at() {
    let provided = ts(2026, 1, 2, 3, 4, 5);
    let context = Context::new("client-a".into(), None, None, provided);
    assert_eq!(context.created_at, provided);
}

#[test]
fn test_reasoning_step_has_required_fields() {
    let step = sample_reasoning_step();
    assert_eq!(step.operation, "analyzed");
    assert_eq!(step.details, "compared latency distributions");
    assert_eq!(step.result, "threshold too low");
}

#[test]
fn test_decision_reasoning_is_optional() {
    let none_reasoning = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        None,
        Some(0.7),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert_eq!(none_reasoning.reasoning, None);

    let some_reasoning = Decision::try_new(
        "architecture".into(),
        "Use abstraction layer for fault isolation".into(),
        Some(vec![sample_reasoning_step()]),
        Some(0.7),
        None,
        context_id(),
        None,
        None,
        ts(2026, 1, 2, 3, 4, 5),
    )
    .unwrap();
    assert!(
        some_reasoning
            .reasoning
            .as_ref()
            .is_some_and(|steps| !steps.is_empty())
    );
}
