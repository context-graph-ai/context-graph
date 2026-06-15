use std::fmt;

use crate::{
    ContextId, DecisionId, DecisionStatus, EvidenceId, IntentionId, IntentionStatus,
    InvalidationId, InvalidationStatus, TxId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    UniqueViolation {
        table: String,
        column: String,
    },
    ImmutableTable {
        table: String,
    },
    ForeignKeyViolation {
        table: String,
        column: String,
        ref_table: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CgError {
    InvalidIdentity {
        context: &'static str,
    },
    UniqueViolation {
        entity: &'static str,
        id: Option<String>,
    },
    MissingIntention,
    IntentionNotFound {
        id: IntentionId,
    },
    NotFound {
        id: Option<String>,
        missing: Option<&'static str>,
    },
    ImmutableField {
        field: String,
    },
    ImmutableEntity {
        kind: &'static str,
    },
    IllegalTransition {
        from: DecisionStatus,
        to: DecisionStatus,
    },
    MissingEvidenceRef,
    InvalidEvidenceHash {
        hash: String,
    },
    EmbeddingSpaceIdMismatch {
        embedder_declared: String,
        registry_declared: String,
    },
    EmbeddingSpaceConfigMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    CrossSpaceQueryRefused {
        query_space: String,
        candidate_space: String,
        query_dim: Option<usize>,
        candidate_dim: Option<usize>,
    },
    CrossSpaceCosineRefused {
        live_space: String,
        reference_space: String,
    },
    EmbeddingDimensionMismatch {
        space_id: String,
        expected: u32,
        found: u32,
    },
    EmbeddingSpaceIdReserved {
        id: String,
    },
    UnknownEmbeddingSpace {
        id: String,
    },
    InvalidEmbeddingSpaceId {
        reason: &'static str,
    },
    InconsistentEmbeddingSpaceDeclaration {
        space_id: String,
    },
    CrossContextEmbeddingTargetRejected {
        handle_context: ContextId,
        target_context: ContextId,
    },
    EvidenceNotFound {
        evidence_id: EvidenceId,
    },
    ReindexAlreadyInProgress {
        embedding_space_id: String,
    },
    ReindexPaused {
        embedding_space_id: String,
        last_committed_cursor: usize,
    },
    ReservedProducerTag {
        tag: String,
    },
    EmbedderDeclaresNoSupportedInputs {
        embedding_space_id: String,
    },
    EmptyVector,
    InvalidVectorFloat {
        field: &'static str,
        index: usize,
        actual: &'static str,
    },
    EmptyEmbeddingVector,
    InvalidEmbeddingVectorFloat,
    EntityArchived {
        entity_id: crate::EntityId,
        archived_at: TxId,
    },
    EntityNotFound {
        entity_id: crate::EntityId,
    },
    ProducerFieldExceedsMaxLength {
        field: &'static str,
        max: usize,
        actual: usize,
    },
    SignatureSpaceNotDeclaredLocally {
        signature_id: String,
        declared_space: String,
        locally_declared_spaces: Vec<String>,
    },
    CompositeQueryRequiresExplicitFusionPolicy,
    CompositeQueryClausesShareSpace {
        space_id: String,
    },
    TextEmbedderDisabled,
    EmbedderModelNotAvailable {
        source: String,
        path: String,
    },
    EmbedderModelCorrupt {
        path: String,
    },
    EmbedderModelRevisionMismatch {
        expected: String,
        actual: String,
    },
    OfflineModeNoCache {
        path: String,
    },
    UnsupportedImportFormat {
        format: String,
    },
    TranscriptTooLargeForModel {
        model_context: usize,
        required: usize,
    },
    UnsupportedEvidenceKind {
        kind: crate::EvidenceKind,
    },
    RawMediaNotStored,
    EvidenceUnavailable,
    InvalidMimeForEvidenceKind,
    InvalidSourceRefScheme {
        scheme: String,
        host: Option<String>,
    },
    InvalidEvidenceRegionBounds {
        field: &'static str,
    },
    EmbeddingTargetNotFound {
        id: crate::EvidenceId,
    },
    LegacyVectorStoreDetected,
    LegacySchemaDetected {
        found_version: u32,
        expected_version: u32,
    },
    MissingVectorColumnForSpace {
        table: String,
        column: String,
    },
    InvalidDerivedFromReference,
    AmbiguousEmbeddingSpace,
    DuplicateEvidenceId {
        id: crate::EvidenceId,
    },
    CrossSpaceQueryRejected {
        query_space: String,
        target_space: String,
    },
    IntentionIllegalTransition {
        from: IntentionStatus,
        to: IntentionStatus,
    },
    ActiveDecisionsReference {
        intention_id: IntentionId,
        decision_ids: Vec<String>,
    },
    CrossContextRelationship,
    CrossContextObservation,
    CrossContextViolation,
    CrossContextNotSupported,
    ActiveIntentionProtection,
    CyclicCitation,
    InvalidDepth {
        requested: u32,
        cap: u8,
    },
    VectorDimensionMismatch {
        expected: usize,
        got: usize,
    },
    EngineStateMachineError(String),
    InvalidTransition {
        from: InvalidationStatus,
        to: InvalidationStatus,
    },
    InvalidationNotFound {
        id: InvalidationId,
    },
    ResolutionDecisionDoesNotServeIntention {
        invalidation_id: InvalidationId,
        decision_id: DecisionId,
        expected_intention_ids: Vec<IntentionId>,
    },
    ResolutionDecisionRequiresActiveDecision {
        invalidation_id: InvalidationId,
        decision_id: DecisionId,
        current_status: DecisionStatus,
    },
    BasisStillValidRequiresActiveDecision {
        invalidation_id: InvalidationId,
        decision_id: DecisionId,
        current_status: DecisionStatus,
    },
    DecisionNotFound {
        id: DecisionId,
    },
    ResolveActionMissingDecision {
        invalidation_id: InvalidationId,
    },
    ResolveActionUnexpectedDecision {
        invalidation_id: InvalidationId,
    },
    ForeignKeyViolation {
        table: &'static str,
        column: &'static str,
        ref_table: &'static str,
    },
    Rollback {
        txid: TxId,
    },
    EngineFault {
        source: EngineError,
    },
    Engine(String),
}

impl fmt::Display for CgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgError::CrossContextNotSupported => write!(
                f,
                "this discovery view is scoped to one context at a time; use the cross-client retrospective view for a combined answer"
            ),
            _ => write!(f, "{self:?}"),
        }
    }
}

impl std::error::Error for CgError {}
