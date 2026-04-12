#![allow(clippy::too_many_arguments, clippy::new_without_default)]

use std::collections::HashMap;

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub type IntentionId = Uuid;
pub type DecisionId = Uuid;
pub type EntityId = Uuid;
pub type ObservationId = Uuid;
pub type OutcomeId = Uuid;
pub type InvalidationId = Uuid;
pub type SignatureId = Uuid;
pub type DigestId = Uuid;
pub type BlueprintId = Uuid;
pub type ContextId = Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfidenceError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentionStatus {
    Active,
    Paused,
    Superseded,
    Archived,
}

impl IntentionStatus {
    pub const ALL: [Self; 3] = [Self::Active, Self::Superseded, Self::Archived];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionStatus {
    Active,
    Superseded,
    Invalidated,
}

impl DecisionStatus {
    pub const ALL: [Self; 2] = [Self::Active, Self::Superseded];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidationStatus {
    Pending,
    Acknowledged,
    Dismissed,
    Resolved,
}

impl InvalidationStatus {
    pub const ALL: [Self; 3] = [Self::Pending, Self::Acknowledged, Self::Dismissed];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

impl Severity {
    pub const ALL: [Self; 2] = [Self::Warning, Self::Critical];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureStatus {
    Active,
    Retired,
}

impl SignatureStatus {
    pub const ALL: [Self; 1] = [Self::Active];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlueprintStatus {
    Active,
    Deprecated,
}

impl BlueprintStatus {
    pub const ALL: [Self; 1] = [Self::Active];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentionOrigin {
    Human,
    Agent,
    Signature,
    Blueprint,
}

impl IntentionOrigin {
    pub const ALL: [Self; 3] = [Self::Human, Self::Agent, Self::Signature];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    Serves,
    BasedOn,
    About,
    Cites,
    Supersedes,
    RelatesTo,
    ObservedOn,
    HasOutcome,
    Invalidates,
    TriggeredBy,
    MatchedBy,
    Generated,
    ExtractedFrom,
    Instantiates,
    Refines,
}

impl RelationshipType {
    pub const ALL: [Self; 12] = [
        Self::Serves,
        Self::BasedOn,
        Self::About,
        Self::Cites,
        Self::Supersedes,
        Self::RelatesTo,
        Self::ObservedOn,
        Self::HasOutcome,
        Self::Invalidates,
        Self::TriggeredBy,
        Self::MatchedBy,
        Self::Generated,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipEndpoint {
    Intention(IntentionId),
    Decision(DecisionId),
    Entity(EntityId),
    Observation(ObservationId),
    Outcome(OutcomeId),
    Invalidation(InvalidationId),
    Signature(SignatureId),
    Digest(DigestId),
    Blueprint(BlueprintId),
    Context(ContextId),
}

fn wrong_timestamp() -> DateTime<Utc> {
    Utc.timestamp_opt(0, 0).single().expect("epoch is valid")
}

fn empty_map() -> HashMap<String, Value> {
    HashMap::new()
}

fn empty_string_map() -> HashMap<String, String> {
    HashMap::new()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub operation: String,
    pub details: String,
    pub result: String,
    pub agent_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl ReasoningStep {
    pub fn new(
        _operation: String,
        _details: String,
        _result: String,
        _agent_id: Option<String>,
        _timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            operation: "stub-operation".into(),
            details: "stub-details".into(),
            result: "stub-result".into(),
            agent_id: Some("stub-agent".into()),
            timestamp: wrong_timestamp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Intention {
    pub id: IntentionId,
    pub description: String,
    pub status: IntentionStatus,
    pub context_id: ContextId,
    pub origin: IntentionOrigin,
    pub origin_ref: Option<SignatureId>,
    pub blueprint_id: Option<BlueprintId>,
    pub bindings: Option<HashMap<String, Value>>,
    pub blueprint_proposal: Option<String>,
    pub properties: Option<HashMap<String, Value>>,
    pub tags: Option<Vec<String>>,
    pub created_at: DateTime<Utc>,
}

impl Intention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _description: String,
        _context_id: ContextId,
        _origin: IntentionOrigin,
        _origin_ref: Option<SignatureId>,
        _blueprint_id: Option<BlueprintId>,
        _bindings: Option<HashMap<String, Value>>,
        _blueprint_proposal: Option<String>,
        _properties: Option<HashMap<String, Value>>,
        _tags: Option<Vec<String>>,
        _created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            description: "stub-intention".into(),
            status: IntentionStatus::Archived,
            context_id: Uuid::nil(),
            origin: IntentionOrigin::Human,
            origin_ref: Some(Uuid::nil()),
            blueprint_id: Some(Uuid::nil()),
            bindings: Some(empty_map()),
            blueprint_proposal: Some("stub-blueprint".into()),
            properties: Some(empty_map()),
            tags: Some(vec!["stub".into()]),
            created_at: wrong_timestamp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub id: DecisionId,
    pub decision_type: String,
    pub description: String,
    pub reasoning: Option<Vec<ReasoningStep>>,
    pub confidence: Option<f64>,
    pub status: DecisionStatus,
    pub agent_id: Option<String>,
    pub context_id: ContextId,
    pub properties: Option<HashMap<String, Value>>,
    pub tags: Option<Vec<String>>,
    pub created_at: DateTime<Utc>,
}

impl Decision {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        _decision_type: String,
        _description: String,
        _reasoning: Option<Vec<ReasoningStep>>,
        _confidence: Option<f64>,
        _agent_id: Option<String>,
        _context_id: ContextId,
        _properties: Option<HashMap<String, Value>>,
        _tags: Option<Vec<String>>,
        _created_at: DateTime<Utc>,
    ) -> Result<Self, ConfidenceError> {
        Ok(Self {
            id: Uuid::nil(),
            decision_type: "stub-decision-type".into(),
            description: "stub-decision".into(),
            reasoning: Some(vec![]),
            confidence: Some(0.123),
            status: DecisionStatus::Superseded,
            agent_id: Some("stub-agent".into()),
            context_id: Uuid::nil(),
            properties: Some(empty_map()),
            tags: Some(vec!["stub".into()]),
            created_at: wrong_timestamp(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub entity_type: String,
    pub name: String,
    pub context_id: ContextId,
    pub properties: Option<HashMap<String, Value>>,
    pub created_at: DateTime<Utc>,
}

impl Entity {
    pub fn new(
        _entity_type: String,
        _name: String,
        _context_id: ContextId,
        _properties: Option<HashMap<String, Value>>,
        _created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            entity_type: "stub-entity-type".into(),
            name: "stub-entity".into(),
            context_id: Uuid::nil(),
            properties: Some(empty_map()),
            created_at: wrong_timestamp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub id: ObservationId,
    pub observation_type: String,
    pub data: HashMap<String, Value>,
    pub source: String,
    pub entity_id: EntityId,
    pub context_id: ContextId,
    pub timestamp: DateTime<Utc>,
}

impl Observation {
    pub fn new(
        _observation_type: String,
        _data: HashMap<String, Value>,
        _source: String,
        _entity_id: EntityId,
        _context_id: ContextId,
        _timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            observation_type: "stub-observation-type".into(),
            data: empty_map(),
            source: "stub-source".into(),
            entity_id: Uuid::nil(),
            context_id: Uuid::nil(),
            timestamp: wrong_timestamp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub id: OutcomeId,
    pub decision_id: DecisionId,
    pub success: bool,
    pub notes: Option<String>,
    pub recorded_at: DateTime<Utc>,
}

impl Outcome {
    pub fn new(
        _decision_id: DecisionId,
        success: bool,
        _notes: Option<String>,
        _recorded_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            decision_id: Uuid::nil(),
            success: !success,
            notes: Some("stub-notes".into()),
            recorded_at: wrong_timestamp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Invalidation {
    pub id: InvalidationId,
    pub status: InvalidationStatus,
    pub affected_decision_id: DecisionId,
    pub trigger_observation_id: ObservationId,
    pub basis_diff: HashMap<String, Value>,
    pub severity: Severity,
    pub detected_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution_action: Option<String>,
    pub resolution_decision_id: Option<DecisionId>,
}

impl Invalidation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _affected_decision_id: DecisionId,
        _trigger_observation_id: ObservationId,
        _basis_diff: HashMap<String, Value>,
        _severity: Severity,
        _resolved_at: Option<DateTime<Utc>>,
        _resolution_action: Option<String>,
        _resolution_decision_id: Option<DecisionId>,
        _detected_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            status: InvalidationStatus::Acknowledged,
            affected_decision_id: Uuid::nil(),
            trigger_observation_id: Uuid::nil(),
            basis_diff: empty_map(),
            severity: Severity::Info,
            detected_at: wrong_timestamp(),
            resolved_at: Some(wrong_timestamp()),
            resolution_action: Some("stub-resolution".into()),
            resolution_decision_id: Some(Uuid::nil()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signature {
    pub id: SignatureId,
    pub signature_type: String,
    pub description: String,
    pub match_criteria: HashMap<String, Value>,
    pub scope_labels: HashMap<String, String>,
    pub confidence: f64,
    pub origin_context_ids: Option<Vec<ContextId>>,
    pub status: SignatureStatus,
    pub created_at: DateTime<Utc>,
    pub last_matched_at: Option<DateTime<Utc>>,
}

impl Signature {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        _signature_type: String,
        _description: String,
        _match_criteria: HashMap<String, Value>,
        _scope_labels: HashMap<String, String>,
        confidence: f64,
        _origin_context_ids: Option<Vec<ContextId>>,
        _last_matched_at: Option<DateTime<Utc>>,
        _created_at: DateTime<Utc>,
    ) -> Result<Self, ConfidenceError> {
        Ok(Self {
            id: Uuid::nil(),
            signature_type: "stub-signature-type".into(),
            description: "stub-signature".into(),
            match_criteria: empty_map(),
            scope_labels: empty_string_map(),
            confidence: confidence + 0.123,
            origin_context_ids: Some(vec![]),
            status: SignatureStatus::Retired,
            created_at: wrong_timestamp(),
            last_matched_at: Some(wrong_timestamp()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Digest {
    pub id: DigestId,
    pub source: String,
    pub session_ref: Option<String>,
    pub summary: String,
    pub embedding: Vec<f32>,
    pub context_id: ContextId,
    pub properties: Option<HashMap<String, Value>>,
    pub created_at: DateTime<Utc>,
}

impl Digest {
    pub fn new(
        _source: String,
        _session_ref: Option<String>,
        _summary: String,
        _embedding: Vec<f32>,
        _context_id: ContextId,
        _properties: Option<HashMap<String, Value>>,
        _created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            source: "stub-source".into(),
            session_ref: Some("stub-session".into()),
            summary: "stub-summary".into(),
            embedding: vec![],
            context_id: Uuid::nil(),
            properties: Some(empty_map()),
            created_at: wrong_timestamp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blueprint {
    pub id: BlueprintId,
    pub canonical_description: String,
    pub category: String,
    pub slots: HashMap<String, Value>,
    pub default_config_shape: Option<HashMap<String, Value>>,
    pub version: String,
    pub status: BlueprintStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl Blueprint {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _canonical_description: String,
        _category: String,
        _slots: HashMap<String, Value>,
        _default_config_shape: Option<HashMap<String, Value>>,
        _version: String,
        _updated_at: Option<DateTime<Utc>>,
        _created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            canonical_description: "stub-blueprint".into(),
            category: "stub-category".into(),
            slots: empty_map(),
            default_config_shape: Some(empty_map()),
            version: "0.0.0-stub".into(),
            status: BlueprintStatus::Deprecated,
            created_at: wrong_timestamp(),
            updated_at: Some(wrong_timestamp()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    pub id: ContextId,
    pub name: String,
    pub labels: Option<HashMap<String, String>>,
    pub properties: Option<HashMap<String, Value>>,
    pub created_at: DateTime<Utc>,
}

impl Context {
    pub fn new(
        _name: String,
        _labels: Option<HashMap<String, String>>,
        _properties: Option<HashMap<String, Value>>,
        _created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::nil(),
            name: "stub-context".into(),
            labels: Some(empty_string_map()),
            properties: Some(empty_map()),
            created_at: wrong_timestamp(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relationship {
    pub source: RelationshipEndpoint,
    pub target: RelationshipEndpoint,
    pub relationship_type: RelationshipType,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
}

impl Relationship {
    pub fn new(
        _source: RelationshipEndpoint,
        _target: RelationshipEndpoint,
        _relationship_type: RelationshipType,
        _valid_from: DateTime<Utc>,
        _valid_to: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            source: RelationshipEndpoint::Context(Uuid::nil()),
            target: RelationshipEndpoint::Context(Uuid::nil()),
            relationship_type: RelationshipType::RelatesTo,
            valid_from: wrong_timestamp(),
            valid_to: Some(wrong_timestamp()),
        }
    }
}
