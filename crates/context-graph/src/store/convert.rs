use contextdb_core::Error as EngineError;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

pub(super) fn map_engine_err(err: EngineError) -> CgError {
    match err {
        EngineError::ImmutableTable(t) => match t.as_str() {
            "observations" => CgError::ImmutableEntity {
                kind: "observation",
            },
            "digests" => CgError::ImmutableEntity { kind: "digest" },
            other => CgError::Engine(format!("unexpected immutable table: {other}")),
        },
        EngineError::NotFound(msg) => CgError::NotFound {
            id: Some(msg),
            missing: None,
        },
        EngineError::InvalidStateTransition(msg) => CgError::EngineStateMachineError(msg),
        EngineError::PropagationAborted {
            table,
            column,
            from,
            to,
        } => CgError::EngineStateMachineError(format!(
            "propagation aborted: {table}.{column} {from} -> {to}"
        )),
        EngineError::CycleDetected { .. } => CgError::AcyclicCitesViolation,
        EngineError::UniqueViolation { table, .. } => CgError::UniqueViolation {
            entity: static_table_name(&table),
            id: None,
        },
        EngineError::ForeignKeyViolation {
            table,
            column,
            ref_table,
        } => CgError::ForeignKeyViolation {
            table: static_table_name(&table),
            column: static_column_name(&column),
            ref_table: static_table_name(&ref_table),
        },
        EngineError::VectorDimensionMismatch { expected, got } => {
            CgError::VectorDimensionMismatch { expected, got }
        }
        EngineError::BfsDepthExceeded(d) => CgError::Engine(format!("bfs depth exceeded: {d}")),
        other => CgError::Engine(format!("unexpected engine error: {other}")),
    }
}

fn static_table_name(name: &str) -> &'static str {
    match name {
        "contexts" => "context",
        "intentions" => "intention",
        "decisions" => "decision",
        "entities" => "entity",
        "observations" => "observation",
        "outcomes" => "outcome",
        "digests" => "digest",
        "invalidations" => "invalidation",
        "signatures" => "signature",
        "blueprints" => "blueprint",
        "edges" => "edge",
        "entity_snapshots" => "entity_snapshot",
        "intention_snapshots" => "intention_snapshot",
        "approvals" => "approval",
        "audit_log" => "audit_log",
        _ => "unknown",
    }
}

fn static_column_name(name: &str) -> &'static str {
    match name {
        "id" => "id",
        "context_id" => "context_id",
        "entity_id" => "entity_id",
        "intention_id" => "intention_id",
        "decision_id" => "decision_id",
        "observation_id" => "observation_id",
        "source_id" => "source_id",
        "target_id" => "target_id",
        "affected_decision_id" => "affected_decision_id",
        "trigger_observation_id" => "trigger_observation_id",
        "resolution_decision_id" => "resolution_decision_id",
        "origin_ref" => "origin_ref",
        "blueprint_id" => "blueprint_id",
        _ => "unknown",
    }
}

pub(super) fn intention_status_str(status: IntentionStatus) -> &'static str {
    match status {
        IntentionStatus::Active => "active",
        IntentionStatus::Paused => "paused",
        IntentionStatus::Superseded => "superseded",
        IntentionStatus::Archived => "archived",
    }
}

pub(super) fn intention_status_from_str(s: &str) -> Result<IntentionStatus> {
    match s {
        "active" => Ok(IntentionStatus::Active),
        "paused" => Ok(IntentionStatus::Paused),
        "superseded" => Ok(IntentionStatus::Superseded),
        "archived" => Ok(IntentionStatus::Archived),
        other => Err(CgError::Engine(format!(
            "unknown intention status: {other}"
        ))),
    }
}

pub(super) fn intention_origin_str(origin: IntentionOrigin) -> &'static str {
    match origin {
        IntentionOrigin::Human => "human",
        IntentionOrigin::Agent => "agent",
        IntentionOrigin::Signature => "signature",
        IntentionOrigin::Blueprint => "blueprint",
    }
}

pub(super) fn intention_origin_from_str(s: &str) -> Result<IntentionOrigin> {
    match s {
        "human" => Ok(IntentionOrigin::Human),
        "agent" => Ok(IntentionOrigin::Agent),
        "signature" => Ok(IntentionOrigin::Signature),
        "blueprint" => Ok(IntentionOrigin::Blueprint),
        other => Err(CgError::Engine(format!(
            "unknown intention origin: {other}"
        ))),
    }
}

pub(super) fn decision_status_from_str(s: &str) -> Result<DecisionStatus> {
    match s {
        "active" => Ok(DecisionStatus::Active),
        "superseded" => Ok(DecisionStatus::Superseded),
        "archived" => Ok(DecisionStatus::Archived),
        other => Err(CgError::Engine(format!("unknown decision status: {other}"))),
    }
}

pub(super) fn entity_type_str(et: EntityType) -> &'static str {
    match et {
        EntityType::Service => "SERVICE",
        EntityType::Person => "PERSON",
        EntityType::Device => "DEVICE",
        EntityType::Module => "MODULE",
        EntityType::Repository => "REPOSITORY",
        EntityType::Configuration => "CONFIGURATION",
        EntityType::Infrastructure => "INFRASTRUCTURE",
        EntityType::Location => "LOCATION",
        EntityType::File => "FILE",
        EntityType::Pipeline => "PIPELINE",
        EntityType::Database => "DATABASE",
        EntityType::Team => "TEAM",
        EntityType::Document => "DOCUMENT",
    }
}

pub(super) fn entity_type_from_str(s: &str) -> Result<EntityType> {
    match s {
        "SERVICE" => Ok(EntityType::Service),
        "PERSON" => Ok(EntityType::Person),
        "DEVICE" => Ok(EntityType::Device),
        "MODULE" => Ok(EntityType::Module),
        "REPOSITORY" => Ok(EntityType::Repository),
        "CONFIGURATION" => Ok(EntityType::Configuration),
        "INFRASTRUCTURE" => Ok(EntityType::Infrastructure),
        "LOCATION" => Ok(EntityType::Location),
        "FILE" => Ok(EntityType::File),
        "PIPELINE" => Ok(EntityType::Pipeline),
        "DATABASE" => Ok(EntityType::Database),
        "TEAM" => Ok(EntityType::Team),
        "DOCUMENT" => Ok(EntityType::Document),
        other => Err(CgError::Engine(format!("unknown entity type: {other}"))),
    }
}

pub(super) fn entity_status_from_str(s: &str) -> Result<EntityStatus> {
    match s {
        "active" => Ok(EntityStatus::Active),
        "archived" => Ok(EntityStatus::Archived),
        other => Err(CgError::Engine(format!("unknown entity status: {other}"))),
    }
}

pub(super) fn relationship_type_str(rel: RelationshipType) -> &'static str {
    match rel {
        RelationshipType::Serves => "SERVES",
        RelationshipType::BasedOn => "BASED_ON",
        RelationshipType::Supersedes => "SUPERSEDES",
        RelationshipType::HasOutcome => "HAS_OUTCOME",
        RelationshipType::DependsOn => "DEPENDS_ON",
        RelationshipType::Cites => "CITES",
        RelationshipType::About => "ABOUT",
        RelationshipType::RelatesTo => "RELATES_TO",
        RelationshipType::ObservedOn => "OBSERVED_ON",
        RelationshipType::Invalidates => "INVALIDATES",
        RelationshipType::TriggeredBy => "TRIGGERED_BY",
        RelationshipType::ExtractedFrom => "EXTRACTED_FROM",
        RelationshipType::MatchedBy => "MATCHED_BY",
        RelationshipType::Generated => "GENERATED",
        RelationshipType::Instantiates => "INSTANTIATES",
        RelationshipType::Refines => "REFINES",
    }
}

pub(super) fn audit_operation_str(op: AuditOperation) -> &'static str {
    match op {
        AuditOperation::Create => "create",
        AuditOperation::Update => "update",
        AuditOperation::Archive => "archive",
        AuditOperation::Supersede => "supersede",
        AuditOperation::RecordOutcome => "record_outcome",
        AuditOperation::RecordObservation => "record_observation",
    }
}

pub(super) fn audit_operation_from_str(s: &str) -> Result<AuditOperation> {
    match s {
        "create" => Ok(AuditOperation::Create),
        "update" => Ok(AuditOperation::Update),
        "archive" => Ok(AuditOperation::Archive),
        "supersede" => Ok(AuditOperation::Supersede),
        "record_outcome" => Ok(AuditOperation::RecordOutcome),
        "record_observation" => Ok(AuditOperation::RecordObservation),
        other => Err(CgError::Engine(format!("unknown audit operation: {other}"))),
    }
}
