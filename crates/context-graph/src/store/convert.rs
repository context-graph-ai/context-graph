use contextdb_core::Error as EngineError;

use crate::Result;
use crate::error::CgError;
use crate::error::EngineError as CgEngineError;
use crate::types::*;

pub(super) fn map_engine_err(err: EngineError) -> CgError {
    match err {
        EngineError::ImmutableTable(t) => {
            if is_embedding_handle_table(&t) {
                CgError::EngineFault {
                    source: CgEngineError::ImmutableTable { table: t },
                }
            } else {
                match t.as_str() {
                    "observations" => CgError::ImmutableEntity {
                        kind: "observation",
                    },
                    "digests" => CgError::ImmutableEntity { kind: "digest" },
                    other => CgError::Engine(format!("unexpected immutable table: {other}")),
                }
            }
        }
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
        EngineError::CycleDetected { .. } => CgError::CyclicCitation,
        EngineError::UniqueViolation { table, column } => {
            if is_embedding_handle_table(&table) {
                CgError::EngineFault {
                    source: CgEngineError::UniqueViolation { table, column },
                }
            } else {
                CgError::UniqueViolation {
                    entity: static_table_name(&table),
                    id: None,
                }
            }
        }
        EngineError::ForeignKeyViolation {
            child_table,
            child_columns,
            parent_table,
            parent_columns: _,
        } => {
            if is_embedding_handle_table(&child_table) {
                CgError::EngineFault {
                    source: CgEngineError::ForeignKeyViolation {
                        table: child_table,
                        column: child_columns.join(","),
                        ref_table: parent_table,
                    },
                }
            } else {
                CgError::ForeignKeyViolation {
                    table: static_table_name(&child_table),
                    column: static_column_name(foreign_key_display_column(&child_columns)),
                    ref_table: static_table_name(&parent_table),
                }
            }
        }
        EngineError::VectorIndexDimensionMismatch {
            expected, actual, ..
        } => CgError::VectorDimensionMismatch {
            expected,
            got: actual,
        },
        EngineError::ContextScopeViolation { .. } => CgError::CrossContextNotSupported,
        EngineError::DatabaseLocked { holder_pid, path } => {
            CgError::StoreLocked { holder_pid, path }
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
        "evidence" => "evidence",
        "embedding_spaces" => "embedding_space",
        "embedding_space_bindings" => "embedding_space_binding",
        "embedding_handles" => "embedding_handle",
        "evidence_embedding_handles" => "evidence_embedding_handle",
        "entity_reference_embedding_handles" => "entity_reference_embedding_handle",
        "digest_embedding_handles" => "digest_embedding_handle",
        "evidence_availability_log" => "evidence_availability_log",
        "entity_snapshot_observations" => "entity_snapshot_observation",
        "decision_basis_snapshots" => "decision_basis_snapshot",
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

fn is_embedding_handle_table(table: &str) -> bool {
    matches!(
        table,
        "evidence_embedding_handles"
            | "entity_reference_embedding_handles"
            | "digest_embedding_handles"
    )
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
        "derived_from" => "derived_from",
        "evidence_id" => "evidence_id",
        "snapshot_id" => "snapshot_id",
        "embedding_space_id" => "embedding_space_id",
        _ => "unknown",
    }
}

fn foreign_key_display_column(columns: &[String]) -> &str {
    columns
        .iter()
        .find(|column| column.as_str() != "context_id")
        .or_else(|| columns.first())
        .map(String::as_str)
        .unwrap_or("unknown")
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
        "invalidated" => Ok(DecisionStatus::Invalidated),
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
        EntityType::Animal => "ANIMAL",
        EntityType::Vehicle => "VEHICLE",
    }
}

/// Resolve a RAW model-emitted entity-type label (any case, e.g. `"module"`, `"Repository"`) to its
/// `EntityType` variant through the canonical `entity_type_from_str` conversion. Returns `None` for a
/// label the closed vocabulary does not recognize (e.g. `"wormhole"`) — the caller flags it
/// out-of-vocab rather than silently coercing it to a default. This is the single conversion the
/// capture path routes a model string through (per the schema-descriptor / grounding single source).
pub(crate) fn entity_type_from_label(s: &str) -> Option<EntityType> {
    entity_type_from_str(&s.trim().to_ascii_uppercase()).ok()
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
        "ANIMAL" => Ok(EntityType::Animal),
        "VEHICLE" => Ok(EntityType::Vehicle),
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

pub(super) fn relationship_type_from_str(s: &str) -> Result<RelationshipType> {
    match s {
        "SERVES" => Ok(RelationshipType::Serves),
        "BASED_ON" => Ok(RelationshipType::BasedOn),
        "SUPERSEDES" => Ok(RelationshipType::Supersedes),
        "HAS_OUTCOME" => Ok(RelationshipType::HasOutcome),
        "DEPENDS_ON" => Ok(RelationshipType::DependsOn),
        "CITES" => Ok(RelationshipType::Cites),
        "ABOUT" => Ok(RelationshipType::About),
        "RELATES_TO" => Ok(RelationshipType::RelatesTo),
        "OBSERVED_ON" => Ok(RelationshipType::ObservedOn),
        "INVALIDATES" => Ok(RelationshipType::Invalidates),
        "TRIGGERED_BY" => Ok(RelationshipType::TriggeredBy),
        "EXTRACTED_FROM" => Ok(RelationshipType::ExtractedFrom),
        "MATCHED_BY" => Ok(RelationshipType::MatchedBy),
        "GENERATED" => Ok(RelationshipType::Generated),
        "INSTANTIATES" => Ok(RelationshipType::Instantiates),
        "REFINES" => Ok(RelationshipType::Refines),
        other => Err(CgError::Engine(format!(
            "unknown relationship type: {other}"
        ))),
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
        AuditOperation::AcknowledgeInvalidation => "acknowledge_invalidation",
        AuditOperation::ResolveInvalidation => "resolve_invalidation",
        AuditOperation::DismissInvalidation => "dismiss_invalidation",
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
        "acknowledge_invalidation" => Ok(AuditOperation::AcknowledgeInvalidation),
        "resolve_invalidation" => Ok(AuditOperation::ResolveInvalidation),
        "dismiss_invalidation" => Ok(AuditOperation::DismissInvalidation),
        other => Err(CgError::Engine(format!("unknown audit operation: {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // A locked store is a distinct, recoverable condition ("another process owns
    // this store"). The converter must surface it as a matchable `CgError` variant
    // carrying the holder pid and path, so callers classify it by type rather than
    // by substring-matching Debug text (which silently rots when the format shifts).
    #[test]
    fn database_locked_maps_to_typed_store_locked_variant() {
        let path = PathBuf::from("/data/cg/memory.db");
        let mapped = map_engine_err(EngineError::DatabaseLocked {
            holder_pid: 4242,
            path: path.clone(),
        });
        assert_eq!(
            mapped,
            CgError::StoreLocked {
                holder_pid: 4242,
                path,
            }
        );
    }
}
