#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

macro_rules! id_newtype {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(pub Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl From<Uuid> for $name {
            fn from(value: Uuid) -> Self {
                Self(value)
            }
        }

        impl From<$name> for Uuid {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

id_newtype!(IntentionId);
id_newtype!(DecisionId);
id_newtype!(EntityId);
id_newtype!(ObservationId);
id_newtype!(OutcomeId);
id_newtype!(InvalidationId);
id_newtype!(SignatureId);
id_newtype!(DigestId);
id_newtype!(BlueprintId);
id_newtype!(ContextId);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceError {
    pub value: f64,
}

impl fmt::Display for ConfidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "confidence must be between 0.0 and 1.0 inclusive: {}",
            self.value
        )
    }
}

impl Error for ConfidenceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentionStatus {
    Active,
    Paused,
    Superseded,
    Archived,
}

impl IntentionStatus {
    pub const ALL: [Self; 4] = [Self::Active, Self::Paused, Self::Superseded, Self::Archived];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionStatus {
    Active,
    Superseded,
    Invalidated,
}

impl DecisionStatus {
    pub const ALL: [Self; 3] = [Self::Active, Self::Superseded, Self::Invalidated];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidationStatus {
    Pending,
    Acknowledged,
    Dismissed,
    Resolved,
}

impl InvalidationStatus {
    pub const ALL: [Self; 4] = [
        Self::Pending,
        Self::Acknowledged,
        Self::Dismissed,
        Self::Resolved,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

impl Severity {
    pub const ALL: [Self; 3] = [Self::Info, Self::Warning, Self::Critical];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureStatus {
    Active,
    Retired,
}

impl SignatureStatus {
    pub const ALL: [Self; 2] = [Self::Active, Self::Retired];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlueprintStatus {
    Active,
    Deprecated,
}

impl BlueprintStatus {
    pub const ALL: [Self; 2] = [Self::Active, Self::Deprecated];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentionOrigin {
    Human,
    Agent,
    Signature,
    Blueprint,
}

impl IntentionOrigin {
    pub const ALL: [Self; 4] = [Self::Human, Self::Agent, Self::Signature, Self::Blueprint];
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
    pub const ALL: [Self; 15] = [
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
        Self::ExtractedFrom,
        Self::Instantiates,
        Self::Refines,
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

fn validate_confidence(value: f64) -> Result<f64, ConfidenceError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(ConfidenceError { value });
    }
    Ok(value)
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
        operation: String,
        details: String,
        result: String,
        agent_id: Option<String>,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            operation,
            details,
            result,
            agent_id,
            timestamp,
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
    pub fn new(
        description: String,
        context_id: ContextId,
        origin: IntentionOrigin,
        origin_ref: Option<SignatureId>,
        blueprint_id: Option<BlueprintId>,
        bindings: Option<HashMap<String, Value>>,
        blueprint_proposal: Option<String>,
        properties: Option<HashMap<String, Value>>,
        tags: Option<Vec<String>>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: IntentionId::new(),
            description,
            status: IntentionStatus::Active,
            context_id,
            origin,
            origin_ref,
            blueprint_id,
            bindings,
            blueprint_proposal,
            properties,
            tags,
            created_at,
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
    pub fn try_new(
        decision_type: String,
        description: String,
        reasoning: Option<Vec<ReasoningStep>>,
        confidence: Option<f64>,
        agent_id: Option<String>,
        context_id: ContextId,
        properties: Option<HashMap<String, Value>>,
        tags: Option<Vec<String>>,
        created_at: DateTime<Utc>,
    ) -> Result<Self, ConfidenceError> {
        let confidence = confidence.map(validate_confidence).transpose()?;

        Ok(Self {
            id: DecisionId::new(),
            decision_type,
            description,
            reasoning,
            confidence,
            status: DecisionStatus::Active,
            agent_id,
            context_id,
            properties,
            tags,
            created_at,
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
        entity_type: String,
        name: String,
        context_id: ContextId,
        properties: Option<HashMap<String, Value>>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: EntityId::new(),
            entity_type,
            name,
            context_id,
            properties,
            created_at,
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
        observation_type: String,
        data: HashMap<String, Value>,
        source: String,
        entity_id: EntityId,
        context_id: ContextId,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            id: ObservationId::new(),
            observation_type,
            data,
            source,
            entity_id,
            context_id,
            timestamp,
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
        decision_id: DecisionId,
        success: bool,
        notes: Option<String>,
        recorded_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: OutcomeId::new(),
            decision_id,
            success,
            notes,
            recorded_at,
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
    pub fn new(
        affected_decision_id: DecisionId,
        trigger_observation_id: ObservationId,
        basis_diff: HashMap<String, Value>,
        severity: Severity,
        resolved_at: Option<DateTime<Utc>>,
        resolution_action: Option<String>,
        resolution_decision_id: Option<DecisionId>,
        detected_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: InvalidationId::new(),
            status: InvalidationStatus::Pending,
            affected_decision_id,
            trigger_observation_id,
            basis_diff,
            severity,
            detected_at,
            resolved_at,
            resolution_action,
            resolution_decision_id,
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
    pub fn try_new(
        signature_type: String,
        description: String,
        match_criteria: HashMap<String, Value>,
        scope_labels: HashMap<String, String>,
        confidence: f64,
        origin_context_ids: Option<Vec<ContextId>>,
        last_matched_at: Option<DateTime<Utc>>,
        created_at: DateTime<Utc>,
    ) -> Result<Self, ConfidenceError> {
        let confidence = validate_confidence(confidence)?;

        Ok(Self {
            id: SignatureId::new(),
            signature_type,
            description,
            match_criteria,
            scope_labels,
            confidence,
            origin_context_ids,
            status: SignatureStatus::Active,
            created_at,
            last_matched_at,
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
        source: String,
        session_ref: Option<String>,
        summary: String,
        embedding: Vec<f32>,
        context_id: ContextId,
        properties: Option<HashMap<String, Value>>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: DigestId::new(),
            source,
            session_ref,
            summary,
            embedding,
            context_id,
            properties,
            created_at,
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
    pub fn new(
        canonical_description: String,
        category: String,
        slots: HashMap<String, Value>,
        default_config_shape: Option<HashMap<String, Value>>,
        version: String,
        updated_at: Option<DateTime<Utc>>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: BlueprintId::new(),
            canonical_description,
            category,
            slots,
            default_config_shape,
            version,
            status: BlueprintStatus::Active,
            created_at,
            updated_at,
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
        name: String,
        labels: Option<HashMap<String, String>>,
        properties: Option<HashMap<String, Value>>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: ContextId::new(),
            name,
            labels,
            properties,
            created_at,
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
        source: RelationshipEndpoint,
        target: RelationshipEndpoint,
        relationship_type: RelationshipType,
        valid_from: DateTime<Utc>,
        valid_to: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            source,
            target,
            relationship_type,
            valid_from,
            valid_to,
        }
    }
}
