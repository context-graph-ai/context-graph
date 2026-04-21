use std::any::type_name_of_val;

use context_graph::{
    AuditOperation, CreateDecision, DecisionId, DecisionStatus, EntityId, EntityMatchPolicy,
    EntityStatus, EntityType, IntentionId, IntentionOrigin, IntentionStatus, ObservationId,
    RelationshipType,
};
use uuid::Uuid;

// Guards against silent addition or removal of an IntentionStatus variant without updating the ALL array and state-machine tests.
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
}

// Pins the Decision state machine to {Active, Superseded, Archived}; any drift breaks archive and supersede transitions.
#[test]
fn test_decision_status_has_exactly_three_variants() {
    let variants = DecisionStatus::ALL;
    assert_eq!(variants.len(), 3);
    assert_eq!(
        &variants[..],
        &[
            DecisionStatus::Active,
            DecisionStatus::Superseded,
            DecisionStatus::Archived,
        ],
    );
}

// Pins IntentionOrigin to {Human, Agent, Signature, Blueprint}; provenance routing relies on this exact set.
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
}

// Freezes the typed edge vocabulary exposed to callers; adding a variant without updating ALL silently breaks traversal filters.
#[test]
fn test_relationship_type_variants() {
    let variants = RelationshipType::ALL;
    assert_eq!(variants.len(), 16);
    assert_eq!(
        &variants[..],
        &[
            RelationshipType::Serves,
            RelationshipType::BasedOn,
            RelationshipType::Supersedes,
            RelationshipType::HasOutcome,
            RelationshipType::DependsOn,
            RelationshipType::Cites,
            RelationshipType::About,
            RelationshipType::RelatesTo,
            RelationshipType::ObservedOn,
            RelationshipType::Invalidates,
            RelationshipType::TriggeredBy,
            RelationshipType::ExtractedFrom,
            RelationshipType::MatchedBy,
            RelationshipType::Generated,
            RelationshipType::Instantiates,
            RelationshipType::Refines,
        ],
    );
}

// Locks the r3 §3 EntityType vocabulary to the canonical 13 kinds in declaration order.
#[test]
fn test_entity_type_variants() {
    let variants = EntityType::ALL;
    assert_eq!(variants.len(), 13);
    assert_eq!(
        &variants[..],
        &[
            EntityType::Service,
            EntityType::Person,
            EntityType::Device,
            EntityType::Module,
            EntityType::Repository,
            EntityType::Configuration,
            EntityType::Infrastructure,
            EntityType::Location,
            EntityType::File,
            EntityType::Pipeline,
            EntityType::Database,
            EntityType::Team,
            EntityType::Document,
        ],
    );
}

// Keeps EntityStatus as a two-state binary; any third state would invalidate list_entities' include_archived toggle.
#[test]
fn test_entity_status_variants() {
    let variants = EntityStatus::ALL;
    assert_eq!(variants.len(), 2);
    assert_eq!(
        &variants[..],
        &[EntityStatus::Active, EntityStatus::Archived]
    );
}

// Confirms EntityMatchPolicy exposes exactly Any and All so list_decisions' based_on semantics stay exhaustive.
#[test]
fn test_entity_match_policy_variants() {
    let variants = EntityMatchPolicy::ALL;
    assert_eq!(variants.len(), 2);
    assert_eq!(
        &variants[..],
        &[EntityMatchPolicy::Any, EntityMatchPolicy::All],
    );
}

// Pins AuditOperation to the six recorded mutation kinds so audit filters cannot silently drop an operation class.
#[test]
fn test_audit_operation_variants() {
    let variants = AuditOperation::ALL;
    assert_eq!(variants.len(), 6);
    assert_eq!(
        &variants[..],
        &[
            AuditOperation::Create,
            AuditOperation::Update,
            AuditOperation::Archive,
            AuditOperation::Supersede,
            AuditOperation::RecordOutcome,
            AuditOperation::RecordObservation,
        ],
    );
}

// A freshly-created Intention defaults to Active; any other default would persist paused or archived goals by accident.
#[test]
fn test_intention_default_status_is_active() {
    assert_eq!(IntentionStatus::default(), IntentionStatus::Active);
}

// A freshly-created Decision defaults to Active; a different default would skew list_decisions' status filter.
#[test]
fn test_decision_default_status_is_active() {
    assert_eq!(DecisionStatus::default(), DecisionStatus::Active);
}

// A new Entity must enter as Active so list_entities' default excludes it only after an explicit archive.
#[test]
fn test_entity_status_default_is_active() {
    assert_eq!(EntityStatus::default(), EntityStatus::Active);
}

// Unspecified based_on filters default to Any; flipping the default would silently narrow historical callers' results.
#[test]
fn test_entity_match_policy_default_is_any() {
    assert_eq!(EntityMatchPolicy::default(), EntityMatchPolicy::Any);
}

// Guards against accidental unification of IntentionId and DecisionId newtypes; type-level distinction prevents cross-kind mix-ups at the API boundary.
#[test]
fn test_intention_id_not_interchangeable_with_decision_id() {
    let uuid = Uuid::from_u128(11);
    let intention = IntentionId::from(uuid);
    let decision = DecisionId::from(uuid);
    assert_ne!(type_name_of_val(&intention), type_name_of_val(&decision));
}

// Guards against collapsing EntityId and ObservationId into a single newtype; record_observation must not accept an entity id in the observation_id slot.
#[test]
fn test_entity_id_not_interchangeable_with_observation_id() {
    let uuid = Uuid::from_u128(12);
    let entity = EntityId::from(uuid);
    let observation = ObservationId::from(uuid);
    assert_ne!(type_name_of_val(&entity), type_name_of_val(&observation));
}

// Id Display must emit the canonical UUID string so CLI output, audit logs, and error messages stay comparable across processes.
#[test]
fn test_id_display_formats_uuid() {
    let uuid = Uuid::from_u128(42);
    let id = IntentionId::from(uuid);
    assert_eq!(format!("{}", id), uuid.to_string());
}

// Uuid -> Id -> Uuid must round-trip byte-exact; a lossy conversion would corrupt every id passed through serialization.
#[test]
fn test_id_from_uuid_roundtrip() {
    let uuid = Uuid::from_u128(42);
    let id = IntentionId::from(uuid);
    let back: Uuid = id.into();
    assert_eq!(back, uuid);
}

// CreateDecision::default() must leave confidence as None so callers explicitly opt in to a numeric rating instead of inheriting a silent default.
#[test]
fn test_decision_confidence_is_optional() {
    assert_eq!(CreateDecision::default().confidence, None);
}
