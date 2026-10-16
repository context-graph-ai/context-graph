use std::path::{Path, PathBuf};

use contextdb_core::Error as EngineError;
use contextdb_core::read_contract::{
    ReadFailureDetail, ReadFailureKind, ReadFailureLimit, ReaderBreadcrumb,
};

use crate::Result;
use crate::error::EngineError as CgEngineError;
use crate::error::{CgError, ReaderIdentity};
use crate::types::*;

/// Turn what the engine refused an OPEN of the store at `store_path` into what a caller decides on.
///
/// One refusal can only be told properly by the caller that knows the path: readers still reading
/// a store hold a writer off, and the substrate's reader detail carries the count and the reader
/// identities but no path — deliberately, since that detail is wire-encoded. So the store is named
/// here, by the open that was refused on it, and nowhere else: the path-less conversion below
/// never invents one. Every other refusal converts exactly as it does anywhere else, so opening is
/// not a second vocabulary.
pub(super) fn map_open_err(store_path: &Path, err: EngineError) -> CgError {
    if let EngineError::ReadFailure(failure) = &err
        && let ReadFailureDetail::HeldByReaders(detail) = failure.detail()
    {
        // The count and the named readers are carried apart, as the substrate reports them: a
        // reader it counted but could not identify still holds the store, so a caller that read
        // the count off the list would under-report who is in the way.
        return CgError::StoreHeldByReaders {
            observed_direct_readers: detail.observed_direct_readers,
            readers: detail.verified_readers.iter().map(project_reader).collect(),
            path: store_path.to_path_buf(),
        };
    }
    map_engine_err(err)
}

/// One reader the substrate verified, as this crate reports it.
///
/// The record is destructured EXHAUSTIVELY on purpose — no `..`. A field added to it upstream stops
/// this build, which is how "the projection drops nothing" is guaranteed structurally rather than
/// promised in a comment. The process id widens `u32` to `u64` losslessly so every process id cg
/// reports is one width; the run identity is the plain value the substrate's newtype wraps, because
/// the substrate's types stop at this crate's edge.
fn project_reader(reader: &ReaderBreadcrumb) -> ReaderIdentity {
    let ReaderBreadcrumb {
        process_id,
        process_name,
        process_start,
    } = reader;
    ReaderIdentity {
        process_id: u64::from(*process_id),
        process_name: process_name.clone(),
        process_start: process_start.0,
    }
}

/// Turn what the engine refused a CONSUMER-DOOR open of the store at `store_path` into what a
/// consumer decides on.
///
/// The read-only [`crate::ConsumerReader`] and
/// [`crate::Store::open_existing_for_change`] — promise a caller three answers it can act on
/// differently: a store that is NOT THERE, one that is there and needs a WRITABLE open to settle
/// its committed image before it can be read, and one that is there and cannot be read at all.
/// The middle one is the only refusal here a consumer may answer by opening the store writable,
/// which is exactly why it is not folded into either of the others. Nothing
/// else on this surface makes that distinction, and it is deliberately not folded into
/// [`map_open_err`]: create-capable openers create an absent store rather than refusing one,
/// so they use the conversion appropriate for that behavior. Contention keeps its own answers —
/// a writer holding the store and readers holding it
/// off are momentary conditions, not damage — so those pass through the shared conversion.
pub(crate) fn map_consumer_open_err(store_path: &Path, err: EngineError) -> CgError {
    match &err {
        // The engine's own "nothing at this path", by either road: the writable door's typed
        // refusal, and the reading route's store-not-found.
        EngineError::StoreMissing { .. } => {
            return CgError::StoreMissing {
                path: store_path.to_path_buf(),
            };
        }
        EngineError::ReadFailure(failure) if failure.kind() == ReadFailureKind::StoreNotFound => {
            return CgError::StoreMissing {
                path: store_path.to_path_buf(),
            };
        }
        // The store is READABLE as a file and its committed image needs a writable open to settle
        // — an unclean close, or state only a writable hydration performs. Its own answer, because
        // it is the one refusal on this door that a consumer may act on by opening the store
        // WRITABLE: folding it into "unreadable" would either cost that consumer a store it could
        // have recovered, or teach it to parse prose and then risk a mutating open on damage.
        EngineError::ReadFailure(failure)
            if failure.kind() == ReadFailureKind::DirectReadRequiresWriter =>
        {
            return CgError::StoreNeedsWritableRecovery {
                path: store_path.to_path_buf(),
                reason: err.to_string(),
            };
        }
        // Somebody is holding the store. Nothing is wrong with it, and the caller is told who by
        // the one conversion that already knows how to say so.
        EngineError::ReaderCoordinationUnavailable { .. } => {
            return map_engine_err(err);
        }
        EngineError::DatabaseLocked { .. } => {}
        EngineError::ReadFailure(failure)
            if matches!(
                failure.detail(),
                ReadFailureDetail::HeldByWriter(_) | ReadFailureDetail::HeldByReaders(_)
            ) => {}
        EngineError::MemoryBudgetExceeded {
            budget_limit_bytes, ..
        } => {
            return CgError::Engine(format!(
                "the read went past the memory it is allowed: {budget_limit_bytes} bytes"
            ));
        }
        EngineError::ReadFailure(failure)
            if matches!(failure.detail(), ReadFailureDetail::OwnerLimitExceeded(_)) =>
        {
            return map_engine_err(err);
        }
        // Everything else refused this open with the store sitting right there: a permission, a
        // mount, a metadata failure, damage. The store is named and what the substrate found
        // travels with it, because the next step depends entirely on which of those it is.
        _ => {
            return CgError::StoreUnreadable {
                path: store_path.to_path_buf(),
                reason: err.to_string(),
            };
        }
    }
    map_open_err(store_path, err)
}

/// Turn what the engine refused into what a context-graph caller decides on.
///
/// Crate-internal on purpose: a consumer is entitled to the typed [`CgError`], not to the machine
/// that produces it, and a public converter would put the substrate's error vocabulary — the very
/// thing this crate absorbs — into every consumer's build. This is the single conversion; there is
/// no second path onto [`CgError`].
pub(crate) fn map_engine_err(err: EngineError) -> CgError {
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
        // A refusal from inside this process always knows which process holds the store, so it
        // reports one; the optional field records what is unknown at OTHER refusals, it does not
        // make this one unknown.
        EngineError::DatabaseLocked { holder_pid, path } => CgError::StoreLocked {
            holder_pid: Some(u64::from(holder_pid)),
            path,
        },
        // A writer holds the store: the caller is told which store, always, and which process
        // holds it only when the holder published one. Nothing is invented for an unpublished
        // holder and nothing is narrowed — an operator sent after process zero, or after a
        // truncated id, is sent after the wrong process.
        EngineError::ReadFailure(ref failure)
            if matches!(failure.detail(), ReadFailureDetail::HeldByWriter(_)) =>
        {
            let ReadFailureDetail::HeldByWriter(detail) = failure.detail() else {
                unreachable!("the guard above admits only the writer-contention detail")
            };
            CgError::StoreLocked {
                holder_pid: detail.process_id,
                path: PathBuf::from(&detail.store_path),
            }
        }
        EngineError::ReaderCoordinationUnavailable {
            operation,
            path,
            reason,
        } => CgError::ReaderCoordinationUnavailable {
            operation,
            path,
            reason,
        },
        EngineError::ReadFailure(failure)
            if matches!(
                failure.detail(),
                ReadFailureDetail::OwnerLimitExceeded(detail)
                    if detail.limit == ReadFailureLimit::ActiveMs
            ) =>
        {
            let ReadFailureDetail::OwnerLimitExceeded(detail) = failure.detail() else {
                unreachable!("the guard admits only the active-time limit detail")
            };
            CgError::ReadActiveTimeExceeded {
                limit_ms: detail.value,
            }
        }
        EngineError::ReadFailure(failure) => CgError::Engine(failure.to_string()),
        // The working memory cg is allowed on this machine ran out. Nothing is damaged and the
        // statement wrote nothing, so this is a budget the operator can raise -- not an engine
        // fault, and not prose for them to parse.
        EngineError::MemoryBudgetExceeded {
            requested_bytes,
            budget_limit_bytes,
            ..
        } => CgError::WriteDidNotFitTheMemoryLimit {
            requested_bytes,
            limit_bytes: budget_limit_bytes,
        },
        EngineError::BfsDepthExceeded(d) => CgError::Engine(format!("bfs depth exceeded: {d}")),
        // A schema the engine will not accept — a consumer-declared table among
        // them. The engine already says what is wrong and what to declare instead,
        // so its wording is what the caller reads.
        EngineError::SchemaInvalid { reason } => CgError::Engine(reason),
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
    use contextdb_core::read_contract::{
        HeldByReadersDetail, HeldByWriterDetail, ProcessStartIdentity, ReadFailure,
        ReadFailureDetail, ReadFailureKind,
    };

    #[test]
    fn active_time_refusals_keep_their_typed_condition() {
        use contextdb_core::read_contract::OwnerLimitExceededDetail;
        let refusal = |limit| {
            EngineError::ReadFailure(ReadFailure::owner_limit_exceeded(
                OwnerLimitExceededDetail {
                    limit,
                    value: 5_000,
                    required: None,
                    statement: None,
                },
            ))
        };
        assert_eq!(
            map_engine_err(refusal(ReadFailureLimit::ActiveMs)),
            CgError::ReadActiveTimeExceeded { limit_ms: 5_000 },
        );
        assert!(
            matches!(
                map_engine_err(refusal(ReadFailureLimit::Work)),
                CgError::Engine(_)
            ),
            "another read limit must remain a terminal unrelated failure"
        );
    }

    /// The store path a refusal is about, as the substrate renders it: a string, always present.
    const HELD_STORE_PATH: &str = "/data/cg/memory.db";

    /// The writer-contention refusal the substrate hands up, carrying what it knows about the
    /// holder. These cases live here, beside the converter, because they are the only place the
    /// lossless claim can be made at all: no real store on this machine can be held by a process
    /// whose id is above the 32-bit range, and a consumer must not be handed a public converter to
    /// buy the coverage.
    fn held_by_writer(process_id: Option<u64>) -> EngineError {
        EngineError::ReadFailure(
            ReadFailure::new(
                ReadFailureKind::HeldByWriter,
                ReadFailureDetail::HeldByWriter(HeldByWriterDetail {
                    process_id,
                    store_path: HELD_STORE_PATH.to_string(),
                }),
            )
            .expect("a held-by-writer refusal carries the writer-contention detail"),
        )
    }

    /// What the substrate hands up about a store readers are still reading: the number of readers
    /// it observed, and the ones it could verify by name. The fixture counts MORE than it names on
    /// purpose — a reader counted but not identified is a real state, and it is the case a
    /// conversion that read the count off the list would get wrong.
    fn held_by_readers() -> EngineError {
        EngineError::ReadFailure(
            ReadFailure::held_by_readers(HeldByReadersDetail {
                observed_direct_readers: 2,
                verified_readers: vec![ReaderBreadcrumb {
                    process_id: 4242,
                    process_name: "cg-status".to_string(),
                    process_start: ProcessStartIdentity(9_876_543_210),
                }],
            })
            .expect("a held-by-readers refusal carries verifiable reader evidence"),
        )
    }

    // A writer refused because readers are reading is told exactly that, with the store it was
    // refused on: a momentary condition a runtime retries, never a failure it stops on.
    #[test]
    fn an_open_refused_by_reading_readers_becomes_the_busy_store_condition() {
        let store_path = PathBuf::from(HELD_STORE_PATH);

        assert_eq!(
            map_open_err(&store_path, held_by_readers()),
            CgError::StoreHeldByReaders {
                observed_direct_readers: 2,
                // This crate's own reader identity, every field of the substrate's record
                // projected: the pid widened losslessly, the name, and which run it is.
                readers: vec![ReaderIdentity {
                    process_id: 4242,
                    process_name: "cg-status".to_string(),
                    process_start: 9_876_543_210,
                }],
                path: store_path,
            }
        );
    }

    // The count is the substrate's own observation and the named readers are its own values: a
    // conversion that inferred one from the other would tell an operator that one reader is in the
    // way when two are.
    #[test]
    fn the_observed_count_survives_even_when_more_readers_were_counted_than_named() {
        let mapped = map_open_err(Path::new(HELD_STORE_PATH), held_by_readers());

        let CgError::StoreHeldByReaders {
            observed_direct_readers,
            readers,
            ..
        } = mapped
        else {
            panic!("readers reading a store is its own condition: {mapped:?}");
        };
        assert_eq!(observed_direct_readers, 2);
        assert_eq!(readers.len(), 1);
        assert_eq!(readers[0].process_name, "cg-status");
        assert_eq!(readers[0].process_id, 4242);
        assert_eq!(readers[0].process_start, 9_876_543_210);
    }

    // The store's name comes from the open that was refused on it, so the path-less conversion
    // never fabricates one — it has nothing truthful to put there.
    #[test]
    fn the_path_less_conversion_never_invents_a_store_for_the_busy_condition() {
        assert!(
            !matches!(
                map_engine_err(held_by_readers()),
                CgError::StoreHeldByReaders { .. }
            ),
            "only the caller that knows which store it opened may name one"
        );
    }

    // Opening is not a second vocabulary: everything that is not reader contention converts at an
    // open exactly as it converts anywhere else.
    #[test]
    fn an_open_refused_for_any_other_reason_converts_the_way_it_always_does() {
        let path = PathBuf::from(HELD_STORE_PATH);
        let refusal = || EngineError::DatabaseLocked {
            holder_pid: 4242,
            path: path.clone(),
        };

        assert_eq!(map_open_err(&path, refusal()), map_engine_err(refusal()));
    }

    // An operator is told which process to stop, and which store it is holding.
    #[test]
    fn a_holder_that_published_its_process_id_is_carried_back_whole() {
        assert_eq!(
            map_engine_err(held_by_writer(Some(4242))),
            CgError::StoreLocked {
                holder_pid: Some(4242),
                path: PathBuf::from(HELD_STORE_PATH),
            }
        );
    }

    // "The holder published no process id" is a real state. Zero is a process id an operator would
    // go looking for, so an unknown holder must never be rendered as one — while the store path,
    // which is always known, still reaches the caller.
    #[test]
    fn a_holder_that_published_no_process_id_is_reported_unknown_and_never_as_zero() {
        let mapped = map_engine_err(held_by_writer(None));

        assert_eq!(
            mapped,
            CgError::StoreLocked {
                holder_pid: None,
                path: PathBuf::from(HELD_STORE_PATH),
            }
        );
        assert_ne!(
            mapped,
            CgError::StoreLocked {
                holder_pid: Some(0),
                path: PathBuf::from(HELD_STORE_PATH),
            }
        );
    }

    // The process id is carried at the width the substrate reports it in: truncating it names a
    // DIFFERENT process to the operator, which is worse than naming none.
    #[test]
    fn a_process_id_wider_than_thirty_two_bits_survives_the_conversion() {
        let wide = u64::from(u32::MAX) + 1;

        assert_eq!(
            map_engine_err(held_by_writer(Some(wide))),
            CgError::StoreLocked {
                holder_pid: Some(wide),
                path: PathBuf::from(HELD_STORE_PATH),
            }
        );
    }

    // A locked store is a distinct, recoverable condition ("another process owns
    // this store"). The converter must surface it as a matchable `CgError` variant
    // carrying the holder pid and path, so callers classify it by type rather than
    // by substring-matching Debug text (which silently rots when the format shifts).
    // A refusal from inside this process always knows the holder, so it reports one: the optional
    // field records what is unknown elsewhere, it does not make this one unknown.
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
                holder_pid: Some(4242),
                path,
            }
        );
    }

    #[test]
    fn reader_coordination_refusal_keeps_its_operation_path_and_reason() {
        let refusal = EngineError::ReaderCoordinationUnavailable {
            operation: "authoritative purge".into(),
            path: "/data/cg/runtime".into(),
            reason: "reader registry cannot be authenticated".into(),
        };
        let mapped = map_consumer_open_err(Path::new("/data/cg/store.db"), refusal);
        assert!(matches!(
            mapped,
            CgError::ReaderCoordinationUnavailable { .. }
        ));
        assert_eq!(
            mapped.to_string(),
            "authoritative purge cannot safely coordinate readers through /data/cg/runtime: reader registry cannot be authenticated"
        );
    }
}
