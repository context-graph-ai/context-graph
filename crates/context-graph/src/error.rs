use std::fmt;

use crate::{
    ContextBindingKey, ContextId, DecisionId, DecisionStatus, EventPlacementPolicy, EvidenceId,
    IntentionId, IntentionSnapshotId, IntentionStatus, InvalidationId, InvalidationStatus, TxId,
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

/// Why one attempt of an event append stopped: a domain refusal the caller receives as-is, or an
/// engine error the append loop inspects before deciding whether to retry.
pub(crate) enum AppendAttemptError {
    Domain(CgError),
    Engine(contextdb_core::Error),
}

impl From<CgError> for AppendAttemptError {
    fn from(error: CgError) -> Self {
        Self::Domain(error)
    }
}

/// The concrete source-history invariant that a resumable import found broken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceHistoryDivergence {
    ShorterThanCheckpoint {
        checkpoint_offset: u64,
        observed_len: u64,
    },
    PrefixHashMismatch {
        checkpoint_offset: u64,
        expected_sha256: String,
        actual_sha256: String,
    },
    InvalidByteBoundary {
        expected_offset: u64,
        adapter_offset: u64,
    },
    AdapterVersionMismatch {
        expected: String,
        actual: String,
    },
    LocatorIdentityMismatch {
        locator_ref: String,
        expected_source_instance_ref: String,
        actual_source_instance_ref: String,
    },
    OutOfOrderPosition {
        committed_offset: u64,
        candidate_offset: u64,
    },
    SourcePositionOrderMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    OutOfOrderSourcePosition {
        committed: Vec<u64>,
        candidate: Vec<u64>,
    },
}

/// Why an interpretation candidate was refused before it became commit-capable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpretationAdmissionRefusalReason {
    ReleasedTargetUnauthorized,
    ReleasedTargetUnpublished,
    DeclarationMismatch,
    InvalidCandidate,
    PreservationContractRefused,
    AnchorRefused,
}

/// Why an interpretation batch can no longer commit under its persisted claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpretationLeaseRefusalReason {
    StaleClaim,
    ExpiredClaim,
    SourceChanged,
    MembershipChanged,
}

/// Why the requested interpretation/search head pair cannot be activated atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpretationActivationRefusalReason {
    PartialGeneration,
    GenerationNotReady,
    SourceWatermarkMismatch,
    SearchInterpretationMismatch,
}

/// One reader that is holding a store open, as context-graph reports it.
///
/// This crate's OWN type, mirroring every field the substrate verified about a reader — never the
/// substrate's value passed through. That is what the adapter is for: a consumer branching on
/// [`CgError::StoreHeldByReaders`] builds against context-graph's vocabulary and nothing below it,
/// the same way [`CgError::StoreLocked`] reports a plain process id and path.
///
/// Nothing is dropped in the projection for a structural reason rather than a promised one: the
/// conversion destructures the substrate's reader record exhaustively, so a field added upstream
/// stops the build instead of going quietly missing here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderIdentity {
    /// The process to go and find. The substrate reports a `u32`; this crate widens it losslessly
    /// so every process id on this error surface — reader pids here, the holder pid in
    /// [`CgError::StoreLocked`] — is one width, since they name processes in the same operating
    /// system pid space and get compared to each other. Widening never truncates.
    pub process_id: u64,
    /// What that process is, so an operator is not left hunting a bare number.
    pub process_name: String,
    /// Which RUN of that process holds the store, so a recycled process id is still told apart.
    /// The plain value the substrate's newtype wraps: its types stop at this crate's edge.
    pub process_start: u64,
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
    ContextAlreadyExists {
        name: String,
    },
    ContextBindingConflict {
        key: Box<ContextBindingKey>,
        requested_context_id: ContextId,
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
    /// A consumer-declared table was refused before its DDL reached the engine.
    /// `subject` is the offending table name, or the namespace when the namespace
    /// itself is what context-graph refused.
    ConsumerTableRefused {
        subject: String,
        reason: &'static str,
    },
    /// A consumer-declared table already in the store no longer matches the
    /// declaration the consumer opened with. `mismatch` names every difference
    /// found, so the consumer can see which of its own policies — a scope label,
    /// a sync direction, a column — is not the one in force. context-graph
    /// changes nothing: what to do about the table is the consumer's call.
    ConsumerTableDrifted {
        table: String,
        mismatch: String,
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
    /// More than one immutable history branch covers the requested point.
    /// Call `state_at_on_branch` with one of these heads to choose explicitly.
    AmbiguousTemporalState {
        entity_id: crate::EntityId,
        at: TxId,
        branch_heads: Vec<crate::EntitySnapshotId>,
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
    /// An embedder was asked for role-aware event-search preparation (`prepare_document`,
    /// `embed_prepared_document`, `prepare_query`, `embed_prepared_query`) and does not implement
    /// it. The default trait methods answer this instead of silently falling back to plain-text
    /// embedding, so a caller never gets an unlabelled vector back under an event-search space id.
    EventSearchPreparationUnsupported {
        embedding_space_id: String,
    },
    /// A prepared event-search document or query was sealed under one embedder's encoding
    /// profile and handed to a different embedder's `embed_prepared_document`/`embed_prepared_query`.
    /// Every embedder's own preparation stays honored; a mismatched profile is refused rather than
    /// run through inference the sealing embedder never agreed to.
    ///
    /// The bundled text model answers the same way when inference authorized for another
    /// embedder's encoding, or for the other role, is relayed into its own inference hooks: the
    /// vector it would produce carries the reserved event-history identity, which promises tokens
    /// this model itself prepared. Prepare and embed on one embedder — the text you meant to embed
    /// under that encoding must go through that embedder's own `prepare_document`/`prepare_query`
    /// first.
    EventSearchPreparationProfileMismatch {
        expected_profile: String,
        actual_profile: String,
    },
    /// Interpretation search is exposed before its persistence and execution exist. A scaffold
    /// call always refuses explicitly; it never reports a successful no-op.
    InterpretationSearchScaffoldOnly {
        operation: &'static str,
    },
    InterpretationAdmissionRefused {
        context_id: ContextId,
        connector_id: String,
        target_interpretation_version: String,
        reason: InterpretationAdmissionRefusalReason,
    },
    InterpretationLeaseRefused {
        context_id: ContextId,
        connector_id: String,
        generation_id: crate::InterpretationGenerationId,
        reason: InterpretationLeaseRefusalReason,
    },
    InterpretationActivationRefused {
        context_id: ContextId,
        connector_id: String,
        interpretation_generation_id: crate::InterpretationGenerationId,
        search_generation_id: crate::EventSearchGenerationId,
        reason: InterpretationActivationRefusalReason,
    },
    EmptyContextSelector,
    EmptyEventSearchQuery,
    EventSearchLimitExceeded {
        requested: u32,
        maximum: u32,
    },
    HistorySearchNotReady {
        context_ids: Vec<ContextId>,
    },
    IncomparableEmbeddingSpaces {
        embedding_space_ids: Vec<String>,
    },
    EventSearchPassageTargetRequiresProjection {
        passage_id: crate::EventSearchPassageId,
    },
    /// [`crate::embedder::nomic::EventHistoryTextEmbedder::over`]/`registration` refuse to build
    /// the canonical event-history registration over an underlying embedder that cannot actually
    /// serve it: a non-768 dimension, a config that can fetch, an explicitly non-canonical
    /// revision pin, or a missing or invalid local artifact. Admission verifies the canonical
    /// revision marker and local files directly and never fetches or loads the model solely to
    /// decide registration. `reason` names which check failed.
    EventHistoryEmbedderConstructionRefused {
        reason: String,
    },
    /// `EventHistoryTextEmbedder::embed` (the plain, non-role [`crate::embedder::Embedder::embed`]
    /// entrypoint) refuses to guess whether the caller meant the document or the query role — a
    /// caller must go through the checked `prepare_document`/`embed_prepared_document` or
    /// `prepare_query`/`embed_prepared_query` entrypoints of the sealed
    /// [`crate::embedder::EventSearchEmbedder`] extension trait instead — bring it into scope with
    /// `use context_graph::EventSearchEmbedder;` and call them on whatever embedder handle you
    /// hold.
    EventHistoryPlainEmbedRequiresExplicitRole,
    /// [`crate::embedder::EventSearchDocumentBuilder::seal`]/
    /// [`crate::embedder::EventSearchQueryBuilder::seal`] refuse to produce a structurally
    /// invalid sealed value: an omitted `token_count`, omitted `payload`, omitted
    /// `byte_boundaries` (document role only), or a `token_count` inconsistent with `payload`'s
    /// length must each be reported here rather than panicking the host or silently defaulting.
    /// `reason` names which check failed.
    EventSearchPreparedValueInvalid {
        reason: String,
    },
    /// Store registration and the checked `prepare_document`/`prepare_query` and
    /// `embed_prepared_document`/`embed_prepared_query` entrypoints on the sealed
    /// [`crate::embedder::EventSearchEmbedder`] extension trait refuse an
    /// embedder that declares the canonical event-history profile without having the exact
    /// library-controlled Nomic raw or event-history wrapper runtime type at canonical dimension.
    /// Object layout and forwarding from a genuine embedder do not authorize another type.
    /// `declared_profile` names the unverifiable declaration.
    EventHistoryProfileIdentityUnverified {
        declared_profile: String,
    },
    /// The checked, library-owned `embed_prepared_document`/`embed_prepared_query` entrypoints on
    /// the sealed [`crate::embedder::EventSearchEmbedder`] extension trait refuse a prepared value
    /// whose exact `token_count` exceeds that same
    /// embedder's declared [`crate::embedder::Embedder::event_search_capability`] — enforced once,
    /// for every embedder, before any implementer hook runs, rather than left to each hook's own
    /// discretion. Role-neutral and distinct from the pre-existing
    /// [`CgError::TranscriptTooLargeForModel`], which stays bound to the unrelated legacy
    /// plain-embedding path.
    PreparedInputExceedsModelLimit {
        token_count: usize,
        limit: usize,
    },
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
    /// More than one immutable intention-history branch is live at the requested transaction.
    /// Call `intention_at_on_branch` with one returned branch head; guessing would rewrite the
    /// user's recorded meaning.
    AmbiguousIntentionHistory {
        intention_id: IntentionId,
        at: TxId,
        branch_head_ids: Vec<IntentionSnapshotId>,
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
    /// The store is already open and held by a writer; the caller is told which
    /// store, and which process holds it when the holder published one. A distinct,
    /// recoverable condition — callers match this variant instead of
    /// substring-matching the engine's Debug text.
    ///
    /// `holder_pid` is optional and 64 bits wide because that is exactly what the
    /// substrate reports: a holder that published no process record is an unknown
    /// holder, never a holder called zero, and a process id is carried at the width
    /// it was reported in rather than truncated into a narrower one — a truncated id
    /// names a DIFFERENT process to the operator reading it.
    StoreLocked {
        holder_pid: Option<u64>,
        path: std::path::PathBuf,
    },
    /// The store could not be taken by a writer because direct readers are still reading it.
    ///
    /// A momentary, self-clearing condition — the readers finish and the writer's next attempt
    /// succeeds — and its own matchable variant precisely so a consumer branches on the type
    /// instead of reading a message: a runtime that stops on a busy store never comes up, and an
    /// operator told the engine failed goes looking for damage that is not there.
    ///
    /// The three parts are what an operator and a retry loop each need, and they are kept
    /// separate: `observed_direct_readers` is the number of readers the substrate OBSERVED,
    /// `readers` are the ones it could VERIFY and name (fewer, when a reader was counted but not
    /// identified — so the count is never inferred from the list), projected into this crate's own
    /// [`ReaderIdentity`] with every field the substrate verified, and `path` names the store —
    /// supplied by the open the writer was refused on, because the substrate's reader detail
    /// carries no path.
    StoreHeldByReaders {
        observed_direct_readers: u64,
        readers: Vec<ReaderIdentity>,
        path: std::path::PathBuf,
    },
    /// There is no store at `path`, and the door that was asked for it created nothing.
    ///
    /// Its own answer, kept apart from [`CgError::StoreUnreadable`] on purpose: "it is not there"
    /// sends an operator to restore a backup, which is the wrong thing to do about a store that IS
    /// there and cannot be looked at. A link whose target is gone lands here too — nothing was
    /// created through it.
    StoreMissing {
        path: std::path::PathBuf,
    },
    /// The store at `path` is there and this process could not read it — a permission, a mount, a
    /// metadata failure, damage — with what the substrate found in `reason`.
    ///
    /// Never reported as missing, and never answered with first-start defaults: a caller told an
    /// unreadable store is empty writes its data into a deployment nobody will look for.
    StoreUnreadable {
        path: std::path::PathBuf,
        reason: String,
    },
    /// The store at `path` is there and readable, and its committed image needs a WRITABLE open to
    /// settle before anything can be read from it — the last writer left state only a writable
    /// hydration performs. `reason` carries the substrate's own words for the condition.
    ///
    /// Its own answer, kept apart from [`CgError::StoreUnreadable`] because the two send a
    /// consumer opposite ways. This store is RECOVERABLE: a consumer that must keep serving may go
    /// on to the writable existing-only door, recover it, and carry on. An unreadable store is
    /// exactly where that must not happen — reaching for a writable handle on damage nobody has
    /// diagnosed is the worst available move. A consumer told the recoverable case as
    /// "unreadable" either abandons a store it could have taken back, or learns to parse prose and
    /// then takes a mutating open on a genuinely damaged one.
    StoreNeedsWritableRecovery {
        path: std::path::PathBuf,
        reason: String,
    },
    /// A `:memory:` store was asked for a read-only reading surface. There is no committed file to
    /// read, so the request is refused by type before any engine call; `requested` repeats what
    /// the caller asked for.
    InMemoryStoreHasNoReadSurface {
        requested: String,
    },
    /// A statement reached a table outside the reader's declaration — context-graph's own,
    /// the engine's own, or one the consumer owns and did not declare. The declaration is the
    /// whole of what a reader can see, and the refusal names what was reached for.
    UndeclaredTableRefused {
        table: String,
    },
    /// A statement that would change the store was handed to the read-only reading surface. It is
    /// refused before it reaches the engine, and the refusal repeats the statement it would not
    /// run.
    ReadOnlySurfaceRefusedWrite {
        statement: String,
    },
    SourceEventDiverged {
        context_id: ContextId,
        connector_id: String,
        session_ref: String,
        source_event_key: String,
        stored_hash: Box<String>,
        candidate_hash: Box<String>,
    },
    SourceHistoryDiverged {
        source_instance_ref: String,
        cause: SourceHistoryDivergence,
    },
    SourceRecordMalformed {
        connector_id: String,
        session_ref: String,
        position: serde_json::Value,
        reason: String,
    },
    SourcePayloadWouldBeTruncated {
        connector_id: String,
        source_event_key: String,
        reason: String,
    },
    InvalidSourceCommitBoundary {
        reason: String,
    },
    AppendContention {
        context_id: ContextId,
        connector_id: String,
        source_instance_ref: String,
    },
    TenantPolicyBindingRequired {
        expected: EventPlacementPolicy,
    },
    EventPlacementPolicyMismatch {
        expected: EventPlacementPolicy,
        actual: Option<EventPlacementPolicy>,
        reason: String,
    },
    Engine(String),
    /// Forgetting history on a tenant-hub Store needs the engine's edge-local discard, which this
    /// build does not issue; nothing was erased.
    EventHistoryForgetRequiresEdgeDiscard {
        placement: crate::types::EventPlacementState,
    },
}

impl fmt::Display for CgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgError::CrossContextNotSupported => write!(
                f,
                "this discovery view is scoped to one context at a time; use the cross-client retrospective view for a combined answer"
            ),
            // Nothing has gone wrong on this machine: somebody is reading. The operator is told
            // that, which store it is about, and which processes to look for if the wait outlasts
            // their patience — never that the engine did something unexpected.
            CgError::StoreHeldByReaders {
                observed_direct_readers,
                readers,
                path,
            } => write!(
                f,
                "the store at {} is busy: {observed_direct_readers} direct {} still reading it, so a writer cannot take it yet; it clears when they finish. {}",
                path.display(),
                if *observed_direct_readers == 1 {
                    "reader is"
                } else {
                    "readers are"
                },
                describe_readers(*observed_direct_readers, readers)
            ),
            CgError::StoreMissing { path } => write!(
                f,
                "there is no store at {}: this door was asked to work on a store that already exists, so it created nothing — point it at an existing store, or open it with a creating door if a new one is wanted",
                path.display()
            ),
            // Said apart from "missing" deliberately: the operator is told the store IS there and
            // that what stands in the way is this machine, so they go and fix a mount or a
            // permission instead of restoring a backup over a store that is fine.
            CgError::StoreUnreadable { path, reason } => write!(
                f,
                "the store at {} is there and could not be read: {reason}",
                path.display()
            ),
            // Said apart from "unreadable" deliberately: the store is fine and the fix is to open
            // it writable once, which is a step the operator (or the consumer's own recovery path)
            // can take right now — nothing to restore, nothing to diagnose.
            CgError::StoreNeedsWritableRecovery { path, reason } => write!(
                f,
                "the store at {} needs a writable open to settle its committed image before it can be read: {reason}",
                path.display()
            ),
            CgError::InMemoryStoreHasNoReadSurface { requested } => write!(
                f,
                "'{requested}' has no committed file to read, so it has no read-only reading surface; open a file-backed store instead"
            ),
            CgError::UndeclaredTableRefused { table } => write!(
                f,
                "this reader did not declare `{table}`, so it cannot read it; declare the table to read it"
            ),
            CgError::ReadOnlySurfaceRefusedWrite { statement } => write!(
                f,
                "this is a read-only reading surface and will not run a statement that changes the store: {statement}"
            ),
            CgError::EventSearchPreparationUnsupported { embedding_space_id } => write!(
                f,
                "the embedder for `{embedding_space_id}` does not do role-aware event-search preparation, so it cannot encode a passage or a search phrase for event history; it will not fall back to unlabelled plain-text embedding"
            ),
            CgError::EventSearchPreparationProfileMismatch {
                expected_profile,
                actual_profile,
            } => write!(
                f,
                "this text was prepared for `{actual_profile}` and cannot be embedded by `{expected_profile}`; prepare it with the same embedder that will embed it"
            ),
            CgError::InterpretationSearchScaffoldOnly { operation } => write!(
                f,
                "`{operation}` is a public step-5/6 scaffold; its persistence and execution are not implemented yet"
            ),
            CgError::InterpretationAdmissionRefused {
                context_id,
                connector_id,
                target_interpretation_version,
                reason,
            } => write!(
                f,
                "interpretation release `{target_interpretation_version}` for connector `{connector_id}` was refused in Context `{context_id}`: {}; no candidate was admitted and no interpretation rows or head changed",
                interpretation_admission_refusal_message(*reason),
            ),
            CgError::InterpretationLeaseRefused {
                context_id,
                connector_id,
                generation_id,
                reason,
            } => write!(
                f,
                "interpretation generation `{generation_id}` for connector `{connector_id}` in Context `{context_id}` cannot commit its claimed batch: {}; no batch was committed",
                interpretation_lease_refusal_message(*reason),
            ),
            CgError::InterpretationActivationRefused {
                context_id,
                connector_id,
                interpretation_generation_id,
                search_generation_id,
                reason,
            } => write!(
                f,
                "interpretation generation `{interpretation_generation_id}` and search generation `{search_generation_id}` for connector `{connector_id}` in Context `{context_id}` cannot be activated: {}; neither active head changed",
                interpretation_activation_refusal_message(*reason),
            ),
            CgError::EmptyContextSelector => write!(
                f,
                "event-history operations require at least one explicitly selected Context"
            ),
            CgError::EmptyEventSearchQuery => {
                write!(f, "event-history search requires a non-empty query")
            }
            CgError::EventSearchLimitExceeded { requested, maximum } => write!(
                f,
                "event-history search requested {requested} hits; the hard maximum is {maximum}"
            ),
            CgError::HistorySearchNotReady { context_ids } => write!(
                f,
                "event history has no ready search generation for the selected Contexts {context_ids:?}"
            ),
            CgError::IncomparableEmbeddingSpaces {
                embedding_space_ids,
            } => write!(
                f,
                "the selected ready event-history generations use incomparable embedding spaces {embedding_space_ids:?}"
            ),
            CgError::EventSearchPassageTargetRequiresProjection { passage_id } => write!(
                f,
                "event-search passage `{passage_id}` embeddings are written only by the event-search projection"
            ),
            CgError::SourceHistoryDiverged {
                source_instance_ref,
                ..
            } => write!(
                f,
                "the source history for `{source_instance_ref}` no longer matches its committed checkpoint; this source was not changed by this attempt. Do not retry it until you restore the previously imported history or import it under a new source identity"
            ),
            CgError::SourceRecordMalformed {
                connector_id,
                session_ref,
                position,
                reason,
            } => write!(
                f,
                "the record at {position} in {connector_id} session `{session_ref}` is malformed: {reason}. Fix the record before importing this source again; earlier committed batches remain available"
            ),
            CgError::TenantPolicyBindingRequired { .. } => write!(
                f,
                "this import requires an authenticated tenant-hub binding; no Context binding, event schema, or event row was written. Finish tenant enrollment, then retry import"
            ),
            CgError::EventPlacementPolicyMismatch {
                expected, actual, ..
            } => write!(
                f,
                "this import requires {}, but the store is configured for {}; it made no placement change. Use a store configured for {} rather than treating this as a migration",
                event_placement_policy_name(*expected),
                actual
                    .as_ref()
                    .map_or("no event placement policy", |actual| {
                        event_placement_policy_name(*actual)
                    }),
                event_placement_policy_name(*expected),
            ),
            CgError::ContextBindingConflict { key, .. } => write!(
                f,
                "the project reference `{}/{}` is already linked to a different Context; this binding was not changed. Use the Context already linked to this reference or choose a different project reference",
                key.namespace, key.opaque_reference,
            ),
            CgError::AppendContention {
                source_instance_ref,
                ..
            } => write!(
                f,
                "another writer changed source `{source_instance_ref}` while this import was appending; no event batch was committed for this contention. Wait for the other writer to exit, then retry this batch"
            ),
            CgError::EmbedderModelNotAvailable { source, path } => write!(
                f,
                "the text-embedding model from {source} is not available at {path}. Install it with `cg model fetch --defaults`, then try the operation again"
            ),
            _ => write!(f, "{self:?}"),
        }
    }
}

fn interpretation_admission_refusal_message(
    reason: InterpretationAdmissionRefusalReason,
) -> &'static str {
    match reason {
        InterpretationAdmissionRefusalReason::ReleasedTargetUnauthorized => {
            "the requested connector or target version does not match the installed release authority; supply the authorized connector and exact target version"
        }
        InterpretationAdmissionRefusalReason::ReleasedTargetUnpublished => {
            "the installed objects do not publish the requested target; install a released mapper and preservation contract for that exact target"
        }
        InterpretationAdmissionRefusalReason::DeclarationMismatch => {
            "the installed mapper and preservation contract declarations differ; install objects with identical release declarations"
        }
        InterpretationAdmissionRefusalReason::InvalidCandidate => {
            "the mapper produced an invalid effective event; fix the released mapper output before rebuilding"
        }
        InterpretationAdmissionRefusalReason::PreservationContractRefused => {
            "the installed preservation contract rejected the candidate; fix the mapping or contract before rebuilding"
        }
        InterpretationAdmissionRefusalReason::AnchorRefused => {
            "the candidate anchor set is incomplete, invalid, or disagrees with the released declaration; produce the complete valid anchor set before rebuilding"
        }
    }
}

fn interpretation_lease_refusal_message(reason: InterpretationLeaseRefusalReason) -> &'static str {
    match reason {
        InterpretationLeaseRefusalReason::StaleClaim => {
            "the claim token or epoch is no longer current; discard this mapped result and request a fresh claim"
        }
        InterpretationLeaseRefusalReason::ExpiredClaim => {
            "the claim expired before commit; discard this mapped result and request a fresh claim"
        }
        InterpretationLeaseRefusalReason::SourceChanged => {
            "a claimed captured event or anchor no longer matches its source hash; discard this mapped result and rebuild from a fresh source snapshot"
        }
        InterpretationLeaseRefusalReason::MembershipChanged => {
            "the sealed generation membership no longer matches the claim; discard this mapped result and request a fresh claim"
        }
    }
}

fn interpretation_activation_refusal_message(
    reason: InterpretationActivationRefusalReason,
) -> &'static str {
    match reason {
        InterpretationActivationRefusalReason::PartialGeneration => {
            "a generation's completed counts do not cover its sealed membership; finish or rebuild the incomplete generation"
        }
        InterpretationActivationRefusalReason::GenerationNotReady => {
            "one or both generations are not ready; finish both generations before retrying activation"
        }
        InterpretationActivationRefusalReason::SourceWatermarkMismatch => {
            "the generations do not match the current source watermark; rebuild them from the current source snapshot"
        }
        InterpretationActivationRefusalReason::SearchInterpretationMismatch => {
            "the search generation was built from a different interpretation snapshot; build its replacement from the requested interpretation generation"
        }
    }
}

fn event_placement_policy_name(policy: EventPlacementPolicy) -> &'static str {
    match policy {
        EventPlacementPolicy::LocalOnly => "local only",
        EventPlacementPolicy::TenantHub => "tenant hub",
    }
}

/// Every holder of the store, as an operator would go looking for them: the ones the substrate
/// verified by process id and name, and — when it observed more readers than it could identify —
/// the remainder, disclosed as holders that published no identity.
///
/// A reader counted and not named is a live process holding this store right now. Stopping at the
/// verified list would send an operator after the one process they were given, and leave them with
/// a store still busy after they killed it. The unnamed remainder is therefore stated rather than
/// dropped — the same move [`CgError::StoreLocked`] makes for a writer that published no pid — and
/// it is stated only when there is one, so a refusal about fully identified readers carries no
/// boilerplate about anonymity.
fn describe_readers(observed_direct_readers: u64, readers: &[ReaderIdentity]) -> String {
    let named: Vec<String> = readers
        .iter()
        .map(|reader| format!("process {} ({})", reader.process_id, reader.process_name))
        .collect();
    let unnamed = observed_direct_readers.saturating_sub(named.len() as u64);
    let also = if named.is_empty() { "" } else { " other" };
    let unnamed_holders = match unnamed {
        0 => None,
        1 => Some(format!(
            "1{also} holder has not published an identity, so there is no process to name for it"
        )),
        many => Some(format!(
            "{many}{also} holders have not published an identity, so there is no process to name for them"
        )),
    };
    match (named.is_empty(), unnamed_holders) {
        (false, None) => format!("Verified {}", named.join(", ")),
        (false, Some(unnamed_holders)) => {
            format!("Verified {}; {unnamed_holders}", named.join(", "))
        }
        (true, Some(unnamed_holders)) => unnamed_holders,
        // Nothing was observed and nothing was named: the count above is the whole answer.
        (true, None) => "Verified no reader by name".to_string(),
    }
}

impl std::error::Error for CgError {}
