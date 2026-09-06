use std::fmt;

use crate::{
    ContextId, DecisionId, DecisionStatus, EvidenceId, IntentionId, IntentionSnapshotId,
    IntentionStatus, InvalidationId, InvalidationStatus, TxId,
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
    Engine(String),
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
            _ => write!(f, "{self:?}"),
        }
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
