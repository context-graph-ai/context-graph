pub mod load_observer;
pub mod mock;
#[cfg(feature = "nomic-embedder")]
pub mod nomic;

use std::any::Any;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::embedding_space::BuiltinSpaceRegistry;
use crate::{CgError, EmbeddingSpace, EmbeddingSpaceDecl, EvidenceKind, EvidenceProducer, Result};

pub(crate) const NOMIC_DEFAULT_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";
const NOMIC_DEFAULT_REPOSITORY: &str = "nomic-ai/nomic-embed-text-v1.5";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingInput {
    Text(String),
    ImageFrame {
        uri: String,
    },
    /// An in-memory image (encoded or raw bytes, embedder-defined). The input path for
    /// consumers that embed on a hot path and hold the image in memory — no temp-file
    /// URI detour. Maps to `EvidenceKind::ImageFrame`.
    ImageBytes(Vec<u8>),
    AudioSegment {
        uri: String,
    },
    VideoSegment {
        uri: String,
    },
    StructuredSignal(serde_json::Value),
}

impl EmbeddingInput {
    /// The evidence kind this input actually is — the one mapping every embedder uses when it
    /// reports [`CgError::UnsupportedEvidenceKind`], so a caller that sent audio is told `audio`
    /// and a caller that sent a video segment is told `video segment`, never the single kind the
    /// refusing embedder happens to accept. In-memory image bytes and a referenced image frame
    /// are the same kind of evidence, so both answer [`EvidenceKind::ImageFrame`].
    pub fn evidence_kind(&self) -> EvidenceKind {
        match self {
            EmbeddingInput::Text(_) => EvidenceKind::Text,
            EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(_) => {
                EvidenceKind::ImageFrame
            }
            EmbeddingInput::AudioSegment { .. } => EvidenceKind::AudioSegment,
            EmbeddingInput::VideoSegment { .. } => EvidenceKind::VideoSegment,
            EmbeddingInput::StructuredSignal(_) => EvidenceKind::StructuredSignal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    pub embedding_space_id: String,
    pub dimension: usize,
    pub producer: EvidenceProducer,
    pub vector: Vec<f32>,
}

/// An event-search document, prepared for embedding by a specific embedder's role-aware
/// preparation. Opaque outside this crate: there is no public constructor, so a caller can only
/// obtain one from the checked [`EventSearchEmbedder::prepare_document`] entrypoint and can only
/// feed it to the checked `embed_prepared_document` entrypoint — never to `embed_prepared_query`,
/// which takes the distinct [`PreparedEventSearchQuery`] type instead. That type boundary is what
/// stops a caller from constructing or cross-feeding a prepared value by hand.
///
/// ```compile_fail
/// use context_graph::embedder::PreparedEventSearchDocument;
/// // No public fields and no public constructor: a caller outside this crate cannot build one.
/// let _doc = PreparedEventSearchDocument {
///     token_count: 0,
///     byte_boundaries: Vec::new(),
/// };
/// ```
///
/// Building the fields is not enough either: the only route from plain values to a sealed
/// [`PreparedEventSearchDocument`] is [`EventSearchSeal::document_builder`], and
/// [`EventSearchSeal`] itself has no public constructor — only this crate's own checked
/// `prepare_document`/`prepare_query` entrypoints on [`EventSearchEmbedder`] create one, and only
/// for the length of a single call into an [`Embedder`] implementation's own hook.
///
/// ```compile_fail
/// use context_graph::embedder::EventSearchSeal;
/// // No public constructor: an external consumer cannot mint a sealing capability, so it cannot
/// // reach `EventSearchSeal::document_builder`/`query_builder` at all. The arguments match the
/// // constructor exactly, so privacy is the only thing left to refuse this — a call at the wrong
/// // arity would "pass" this block even if the constructor were opened up. The durable guard is
/// // `tests/ui/event_search_seal_has_no_public_constructor.rs`, whose committed compiler-output
/// // snapshot pins the refusal to privacy by name.
/// let _seal = EventSearchSeal::new("attacker_profile".to_string(), true);
/// ```
///
/// Role cross-feed is refused at the type level, not at runtime: `embed_prepared_query` takes
/// only [`PreparedEventSearchQuery`], so a document value cannot be substituted even though both
/// types are produced by the same sealing mechanism.
///
/// ```compile_fail
/// use context_graph::embedder::{
///     Embedder, EventSearchEmbedder, PreparedEventSearchDocument, PreparedEventSearchQuery,
/// };
///
/// // `EventSearchEmbedder` is imported so this call RESOLVES: without it the block would fail on
/// // a missing method and say nothing at all about cross-feeding.
/// fn only_accepts_query(embedder: &dyn Embedder, prepared: &PreparedEventSearchQuery) {
///     let _ = embedder.embed_prepared_query(prepared);
/// }
///
/// fn cross_feed(embedder: &dyn Embedder, doc: &PreparedEventSearchDocument) {
///     // Type error: `doc` is a `&PreparedEventSearchDocument`, not a `&PreparedEventSearchQuery`.
///     only_accepts_query(embedder, doc);
/// }
/// ```
///
/// An external consumer must not be able to read the sealed model input out from under the profile
/// check either: there is no payload accessor on this type at all. The token ids reach an embedder
/// only as an [`EventSearchInference`], which the checked entrypoints mint after the profile and
/// token-ceiling checks have already passed.
///
/// ```compile_fail
/// use context_graph::embedder::PreparedEventSearchDocument;
///
/// fn read_payload(prepared: &PreparedEventSearchDocument) -> &[u32] {
///     // `payload` must not be a public method reachable from outside this crate.
///     prepared.payload()
/// }
/// ```
///
/// Nor must a consumer be able to skip the checked `embed_prepared_document` entrypoint's profile
/// verification by calling the implementer-facing `embed_prepared_document_hook` directly on a
/// concrete embedder it holds — the check must live somewhere a caller cannot route around. The
/// hook takes an [`EventSearchInference`], which only the checked entrypoint can create, so a
/// prepared value is not something a caller can hand it.
///
/// ```compile_fail
/// use context_graph::embedder::{
///     Embedder, EventSearchEmbedder, EventSearchSeal, PreparedEventSearchDocument,
/// };
/// use context_graph::{
///     DistanceMetric, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EvidenceKind, Result,
/// };
///
/// struct ThirdPartyEmbedder;
///
/// impl Embedder for ThirdPartyEmbedder {
///     fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
///         unimplemented!()
///     }
///     fn supported_inputs(&self) -> Vec<EvidenceKind> {
///         vec![EvidenceKind::Text]
///     }
///     fn embedding_space(&self) -> EmbeddingSpace {
///         EmbeddingSpace {
///             embedding_space_id: "third_party".to_string(),
///             dimension: 1,
///             metric: DistanceMetric::Cosine,
///             model_family: "third_party".to_string(),
///             model_name: "third_party".to_string(),
///             model_version: "1".to_string(),
///             supported_evidence_kinds: vec![EvidenceKind::Text],
///             default_for_evidence_kind: None,
///             bindings: Vec::new(),
///         }
///     }
/// }
///
/// let embedder = ThirdPartyEmbedder;
/// let prepared = embedder.prepare_document("hi").expect("preparation succeeds");
/// // A consumer must not be able to reach the unchecked hook directly — only the checked
/// // `embed_prepared_document` entrypoint, which verifies the profile first.
/// let _ = embedder.embed_prepared_document_hook(&prepared);
/// ```
///
/// What a third-party consumer CAN and must still be able to do is prepare and embed through the
/// checked route:
///
/// ```
/// use context_graph::embedder::{
///     Embedder, EventSearchCapability, EventSearchEmbedder, EventSearchInference, EventSearchSeal,
///     PreparedEventSearchDocument,
/// };
/// use context_graph::{
///     DistanceMetric, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EvidenceKind, Result,
/// };
///
/// struct ThirdPartyEmbedder;
///
/// impl Embedder for ThirdPartyEmbedder {
///     fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
///         unimplemented!()
///     }
///     fn supported_inputs(&self) -> Vec<EvidenceKind> {
///         vec![EvidenceKind::Text]
///     }
///     fn embedding_space(&self) -> EmbeddingSpace {
///         EmbeddingSpace {
///             embedding_space_id: "third_party".to_string(),
///             dimension: 1,
///             metric: DistanceMetric::Cosine,
///             model_family: "third_party".to_string(),
///             model_name: "third_party".to_string(),
///             model_version: "1".to_string(),
///             supported_evidence_kinds: vec![EvidenceKind::Text],
///             default_for_evidence_kind: None,
///             bindings: Vec::new(),
///         }
///     }
///     fn event_search_capability(&self) -> Option<EventSearchCapability> {
///         EventSearchCapability::new(64)
///     }
///     fn prepare_document_hook(
///         &self,
///         text: &str,
///         seal: &EventSearchSeal,
///     ) -> Result<PreparedEventSearchDocument> {
///         let ids: Vec<u32> = text.bytes().map(u32::from).collect();
///         let boundaries = (0..text.len()).map(|index| (index, index + 1)).collect();
///         seal.document_builder()
///             .token_count(ids.len())
///             .byte_boundaries(boundaries)
///             .payload(ids)
///             .seal()
///     }
///     fn embed_prepared_document_hook(
///         &self,
///         inference: &EventSearchInference,
///     ) -> Result<EmbeddingOutput> {
///         Ok(EmbeddingOutput {
///             embedding_space_id: self.embedding_space().embedding_space_id,
///             dimension: 1,
///             producer: context_graph::EvidenceProducer {
///                 system: "doctest".to_string(),
///                 model_name: "doctest".to_string(),
///                 model_version: "1".to_string(),
///                 pipeline_version: "1".to_string(),
///             },
///             vector: vec![inference.token_ids().len() as f32],
///         })
///     }
/// }
///
/// let embedder = ThirdPartyEmbedder;
/// // The checked entrypoints are reached on the concrete handle, with no trait-object coercion.
/// let prepared = embedder.prepare_document("hi").expect("preparation succeeds");
/// let output = embedder
///     .embed_prepared_document(&prepared)
///     .expect("checked embedding succeeds");
/// assert_eq!(output.embedding_space_id, "third_party");
/// ```
#[derive(Clone)]
pub struct PreparedEventSearchDocument {
    token_count: usize,
    byte_boundaries: Vec<(usize, usize)>,
    payload: Vec<u32>,
    encoding_profile_id: String,
    canonical_authority: bool,
}

/// Prints what a reader legitimately needs to identify a prepared document — how many tokens it
/// holds, how many source spans it maps, and which encoding profile sealed it — and never the
/// sealed token payload itself. The payload is the sealing embedder's private model input; a log
/// line, a panic message or a test failure that printed it would put the exact token stream
/// somewhere it was never meant to travel, so it is reported only as a count.
impl std::fmt::Debug for PreparedEventSearchDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEventSearchDocument")
            .field("token_count", &self.token_count)
            .field(
                "byte_boundaries",
                &BoundaryCount(self.byte_boundaries.len()),
            )
            .field("encoding_profile_id", &self.encoding_profile_id)
            .field("payload", &RedactedPayload)
            .finish()
    }
}

/// Stands in for a sealed token payload in `Debug` output: it prints a fixed marker and carries
/// no value at all, so no formatting route can reach the token ids through it.
struct RedactedPayload;

impl std::fmt::Debug for RedactedPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<sealed model input, not printed>")
    }
}

/// Reports how many source spans a prepared document maps without printing the spans themselves,
/// whose byte offsets would let a reader reconstruct the tokenization the payload holds.
struct BoundaryCount(usize);

impl std::fmt::Debug for BoundaryCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{} source spans, not printed>", self.0)
    }
}

impl PreparedEventSearchDocument {
    /// The exact final token count of the prepared model input, special tokens included.
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Token-by-token `(start_byte, end_byte)` offsets into the caller's original, unprefixed
    /// source text — exactly one ordered, UTF-8-safe span per token in a sealed prepared
    /// document. Task-prefix and special tokens use zero-width spans. Sealing refuses an absent
    /// map or a map whose length disagrees with the token payload or count.
    pub fn byte_boundaries(&self) -> &[(usize, usize)] {
        &self.byte_boundaries
    }

    /// The encoding-profile id of the [`Embedder`] that sealed this value (see
    /// [`Embedder::encoding_profile_id`]). An `embed_prepared_document` implementation compares
    /// this against its own profile before inference and refuses a mismatch with
    /// [`CgError::EventSearchPreparationProfileMismatch`].
    pub fn encoding_profile_id(&self) -> &str {
        &self.encoding_profile_id
    }

    /// Whether the checked preparation entrypoint established that the sealing embedder had the
    /// exact library-controlled runtime type for the canonical event-history encoding. A value
    /// that merely names the reserved profile without this structural authority is refused.
    pub(crate) fn has_canonical_authority(&self) -> bool {
        self.canonical_authority
    }
}

/// An event-search query, prepared for embedding by a specific embedder's role-aware preparation.
/// Opaque outside this crate, mirroring [`PreparedEventSearchDocument`]: no public constructor,
/// and it is a distinct type from the document preparation so the two can never be cross-fed.
#[derive(Clone)]
pub struct PreparedEventSearchQuery {
    token_count: usize,
    payload: Vec<u32>,
    encoding_profile_id: String,
    canonical_authority: bool,
}

/// Mirrors [`PreparedEventSearchDocument`]'s redacted `Debug`: token count and sealing profile
/// are printed, the sealed model input never is.
impl std::fmt::Debug for PreparedEventSearchQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEventSearchQuery")
            .field("token_count", &self.token_count)
            .field("encoding_profile_id", &self.encoding_profile_id)
            .field("payload", &RedactedPayload)
            .finish()
    }
}

impl PreparedEventSearchQuery {
    /// The exact final token count of the prepared model input, special tokens included.
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Mirrors [`PreparedEventSearchDocument::encoding_profile_id`] for the query role.
    pub fn encoding_profile_id(&self) -> &str {
        &self.encoding_profile_id
    }

    /// Mirrors [`PreparedEventSearchDocument::has_canonical_authority`] for the query role.
    pub(crate) fn has_canonical_authority(&self) -> bool {
        self.canonical_authority
    }
}

/// A library-owned, unforgeable capability that authorizes turning plain preparation values into
/// a sealed [`PreparedEventSearchDocument`]/[`PreparedEventSearchQuery`]. There is no public
/// constructor: only this crate's own checked `prepare_document`/`prepare_query` entrypoints on
/// [`EventSearchEmbedder`] create one (stamped with the calling embedder's own
/// [`Embedder::encoding_profile_id`]), and each is handed to an [`Embedder`] implementation's
/// hook — [`Embedder::prepare_document_hook`]/[`Embedder::prepare_query_hook`] — only for the
/// duration of that one call. An implementation can reach [`EventSearchSeal::document_builder`]/
/// [`EventSearchSeal::query_builder`] only through a seal it was handed this way, so it can never
/// mint a sealed value except inside that hook, and never under a profile other than its own.
pub struct EventSearchSeal {
    encoding_profile_id: String,
    canonical_authority: bool,
}

impl EventSearchSeal {
    fn new(encoding_profile_id: String, canonical_authority: bool) -> Self {
        Self {
            encoding_profile_id,
            canonical_authority,
        }
    }

    /// Start building a sealed event-search document from this seal's own profile.
    pub fn document_builder(&self) -> EventSearchDocumentBuilder<'_> {
        EventSearchDocumentBuilder {
            seal: self,
            token_count: None,
            byte_boundaries: None,
            payload: None,
        }
    }

    /// Start building a sealed event-search query from this seal's own profile.
    pub fn query_builder(&self) -> EventSearchQueryBuilder<'_> {
        EventSearchQueryBuilder {
            seal: self,
            token_count: None,
            payload: None,
        }
    }
}

/// Builder for a sealed [`PreparedEventSearchDocument`], obtainable only from
/// [`EventSearchSeal::document_builder`]. No public constructor of its own: the seal is the sole
/// route to one.
pub struct EventSearchDocumentBuilder<'a> {
    seal: &'a EventSearchSeal,
    token_count: Option<usize>,
    byte_boundaries: Option<Vec<(usize, usize)>>,
    payload: Option<Vec<u32>>,
}

impl<'a> EventSearchDocumentBuilder<'a> {
    /// The exact final token count of the prepared model input, special tokens included.
    pub fn token_count(mut self, token_count: usize) -> Self {
        self.token_count = Some(token_count);
        self
    }

    /// Token-by-token `(start_byte, end_byte)` offsets into the original, unprefixed rendered
    /// text — exact and UTF-8 safe.
    pub fn byte_boundaries(mut self, byte_boundaries: Vec<(usize, usize)>) -> Self {
        self.byte_boundaries = Some(byte_boundaries);
        self
    }

    /// The embedder's own private prepared model input. Opaque to every consumer of the sealed
    /// wrapper; it reaches the sealing embedder again only as the [`EventSearchInference`] the
    /// checked `embed_prepared_document` entrypoint mints once its checks have passed.
    pub fn payload(mut self, payload: Vec<u32>) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Consume the builder, sealing the accumulated values under this builder's seal's profile.
    ///
    /// Fallible by contract, never panicking. Every prepared document carries an exact token
    /// count, the exact model input those tokens are, and a source span for each of them — a
    /// consumer splitting an oversized unit needs all three to cut it at real boundaries. So a
    /// hook that omits `token_count`, `payload` or `byte_boundaries`, or that supplies a
    /// `token_count` or a boundary list whose length disagrees with `payload`, gets back a typed
    /// [`CgError::EventSearchPreparedValueInvalid`] naming the disagreement, rather than crashing
    /// the host or sealing a value whose count silently lies about its contents.
    pub fn seal(self) -> Result<PreparedEventSearchDocument> {
        let token_count =
            self.token_count
                .ok_or_else(|| CgError::EventSearchPreparedValueInvalid {
                    reason: "a prepared event-search document was sealed without a token count"
                        .to_string(),
                })?;
        let payload = self
            .payload
            .ok_or_else(|| CgError::EventSearchPreparedValueInvalid {
                reason: "a prepared event-search document was sealed without a model input"
                    .to_string(),
            })?;
        let byte_boundaries =
            self.byte_boundaries
                .ok_or_else(|| CgError::EventSearchPreparedValueInvalid {
                    reason: "a prepared event-search document was sealed without the source byte boundaries a consumer needs to split it"
                        .to_string(),
                })?;
        if token_count != payload.len() {
            return Err(CgError::EventSearchPreparedValueInvalid {
                reason: format!(
                    "a prepared event-search document reports {token_count} tokens but its model input holds {}",
                    payload.len()
                ),
            });
        }
        if byte_boundaries.len() != token_count {
            return Err(CgError::EventSearchPreparedValueInvalid {
                reason: format!(
                    "a prepared event-search document reports {token_count} tokens but maps {} source spans",
                    byte_boundaries.len()
                ),
            });
        }
        Ok(PreparedEventSearchDocument {
            token_count,
            byte_boundaries,
            payload,
            encoding_profile_id: self.seal.encoding_profile_id.clone(),
            canonical_authority: self.seal.canonical_authority,
        })
    }
}

/// Builder for a sealed [`PreparedEventSearchQuery`], obtainable only from
/// [`EventSearchSeal::query_builder`]. Mirrors [`EventSearchDocumentBuilder`] for the query role.
pub struct EventSearchQueryBuilder<'a> {
    seal: &'a EventSearchSeal,
    token_count: Option<usize>,
    payload: Option<Vec<u32>>,
}

impl<'a> EventSearchQueryBuilder<'a> {
    /// The exact final token count of the prepared model input, special tokens included.
    pub fn token_count(mut self, token_count: usize) -> Self {
        self.token_count = Some(token_count);
        self
    }

    /// The embedder's own private prepared model input.
    pub fn payload(mut self, payload: Vec<u32>) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Consume the builder, sealing the accumulated values under this builder's seal's profile.
    /// Mirrors [`EventSearchDocumentBuilder::seal`]: fallible and never panicking, refusing an
    /// omitted token count, an omitted model input, or a token count that disagrees with the
    /// model input's length with [`CgError::EventSearchPreparedValueInvalid`]. The query role
    /// maps no source spans — nothing splits a search phrase — so it has no boundary check.
    pub fn seal(self) -> Result<PreparedEventSearchQuery> {
        let token_count =
            self.token_count
                .ok_or_else(|| CgError::EventSearchPreparedValueInvalid {
                    reason: "a prepared event-search query was sealed without a token count"
                        .to_string(),
                })?;
        let payload = self
            .payload
            .ok_or_else(|| CgError::EventSearchPreparedValueInvalid {
                reason: "a prepared event-search query was sealed without a model input"
                    .to_string(),
            })?;
        if token_count != payload.len() {
            return Err(CgError::EventSearchPreparedValueInvalid {
                reason: format!(
                    "a prepared event-search query reports {token_count} tokens but its model input holds {}",
                    payload.len()
                ),
            });
        }
        Ok(PreparedEventSearchQuery {
            token_count,
            payload,
            encoding_profile_id: self.seal.encoding_profile_id.clone(),
            canonical_authority: self.seal.canonical_authority,
        })
    }
}

/// Which side of an event search a prepared value was made for: the stored passage or the phrase a
/// user searched with. The two are prepared under different task prefixes, so an embedder that
/// wants to branch on the role — logging, metrics, a role-specific pooling step — reads it from
/// the inference it is handed rather than guessing from the token stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventSearchRole {
    /// A stored passage, prepared by `prepare_document`.
    Document,
    /// A search phrase, prepared by `prepare_query`.
    Query,
}

/// The authorization to run inference on one prepared event-search value, together with the model
/// input to run it on.
///
/// This is what an embedder's `embed_prepared_document_hook`/`embed_prepared_query_hook` receives,
/// and it is the only route from a sealed prepared value to its token ids. It has no public
/// constructor: the checked, library-owned `embed_prepared_document`/`embed_prepared_query`
/// entrypoints on [`EventSearchEmbedder`] create one only after the encoding profile has been verified
/// against the embedder about to run, the canonical event-history identity has been proven rather
/// than merely declared, and the exact token count has been checked against the embedder's own
/// [`Embedder::event_search_capability`]. So an inference that exists is one every check already
/// passed, and a consumer holding a prepared value cannot reach the unchecked hook — it has
/// nothing to call it with, and no way to read the sealed model input on its own.
///
/// It also carries, privately, exactly WHICH embedding the checks passed for: the verified
/// encoding profile, whether this library itself vouched for the canonical event-history
/// identity, and the role the checked entrypoint authorized. Holding a valid inference is
/// therefore not the same as being allowed to run it — an embedder handed one that was authorized
/// for someone else's encoding, or for the other role, can say so and refuse. That is what stops
/// an inference from being relayed out of the embedder it was issued for.
pub struct EventSearchInference {
    token_ids: Vec<u32>,
    role: EventSearchRole,
    authorized_profile_id: String,
    // Read by the bundled Nomic hook, which is the only implementer in this library; a build with
    // no bundled embedder compiles the field and its reader with nothing to call them.
    #[cfg_attr(not(feature = "nomic-embedder"), allow(dead_code))]
    canonical_authority: bool,
}

impl EventSearchInference {
    /// Issue the capability. Private on purpose: this is what makes the checked entrypoints the
    /// only path to inference.
    fn new(
        token_ids: Vec<u32>,
        role: EventSearchRole,
        authorized_profile_id: String,
        canonical_authority: bool,
    ) -> Self {
        Self {
            token_ids,
            role,
            authorized_profile_id,
            canonical_authority,
        }
    }

    /// The encoding profile the checked entrypoint verified before issuing this inference — the
    /// profile of the embedder the tokens were prepared by and authorized for.
    #[cfg_attr(not(feature = "nomic-embedder"), allow(dead_code))]
    pub(crate) fn authorized_profile_id(&self) -> &str {
        &self.authorized_profile_id
    }

    /// Whether this library itself established the canonical event-history identity for the value
    /// this inference was minted from, rather than the profile merely having been declared.
    #[cfg_attr(not(feature = "nomic-embedder"), allow(dead_code))]
    pub(crate) fn has_canonical_authority(&self) -> bool {
        self.canonical_authority
    }

    /// The prepared model input to run inference on — the exact token ids the sealing embedder
    /// produced, task prefix and special tokens included, in order.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// How many tokens the model input holds. Already checked against this embedder's declared
    /// [`Embedder::event_search_capability`] before this value was created.
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    /// Whether these tokens are a stored passage or a search phrase.
    pub fn role(&self) -> EventSearchRole {
        self.role
    }
}

/// Prints the shape of the inference and never the model input itself, mirroring the redacted
/// `Debug` of the prepared types it is minted from: the token stream is the sealing embedder's
/// private model input and does not belong in a log line or a panic message.
impl std::fmt::Debug for EventSearchInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventSearchInference")
            .field("role", &self.role)
            .field("authorized_profile_id", &self.authorized_profile_id)
            .field("token_count", &self.token_ids.len())
            .field("token_ids", &RedactedPayload)
            .finish()
    }
}

/// A declaration that an [`Embedder`] implements exact role-aware event-search preparation and
/// inference, carrying the finite prepared-token ceiling those operations enforce.
///
/// The ceiling is non-zero by construction, so support and its limit cannot drift into separate
/// signals. [`Embedder::event_search_capability`] returning `None` means the operation is
/// unsupported; returning this value means every checked entrypoint may invoke the corresponding
/// implementer hook and will refuse oversized prepared inference first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventSearchCapability {
    token_limit: NonZeroUsize,
}

impl EventSearchCapability {
    /// Declare support with a finite, non-zero prepared-token ceiling. Zero returns `None` rather
    /// than constructing an invalid capability.
    pub const fn new(token_limit: usize) -> Option<Self> {
        match NonZeroUsize::new(token_limit) {
            Some(token_limit) => Some(Self { token_limit }),
            None => None,
        }
    }

    /// The maximum exact prepared-token count accepted before model inference.
    pub const fn token_limit(self) -> usize {
        self.token_limit.get()
    }
}

pub trait Embedder: Send + Sync + Any {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput>;
    fn supported_inputs(&self) -> Vec<EvidenceKind>;
    fn embedding_space(&self) -> EmbeddingSpace;
    fn max_sequence_length(&self) -> usize {
        usize::MAX
    }

    /// Whether this embedder implements exact role-aware event-search preparation and inference,
    /// together with the mandatory finite prepared-token ceiling. The default is unsupported.
    /// This is distinct from [`Embedder::max_sequence_length`], which governs the unrelated
    /// plain-embedding path and remains unlimited by default.
    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        None
    }

    /// A certified upper bound on tokens lost when slicing a prepared document's source.
    ///
    /// `Some(loss)` promises that, for every UTF-8 source slice containing `m` original
    /// nonzero token spans wholly inside it (including duplicate spans), preparing
    /// `prefix + slice` yields at least `m.saturating_sub(loss)` tokens. The declaration
    /// must hold for the actual preparation pipeline throughout its use, including task
    /// instructions, normalization, added tokens and postprocessing. It permits pruning
    /// impossible candidate ends; exact preparation still selects the maximal fitting end.
    ///
    /// The default `None` makes no such promise and retains exhaustive candidate selection.
    /// Token counts themselves need not be monotone under slicing.
    fn event_search_slice_token_loss_bound(&self, _prefix: &str) -> Option<usize> {
        None
    }

    /// This embedder's encoding-profile id: what two embedders must share for one's event-search
    /// preparation to be honored by the other's `embed_prepared_*`. Defaults to this embedder's
    /// declared [`EmbeddingSpace::embedding_space_id`], which already encodes model identity,
    /// dimension and input-encoding version — a preparation made under one model, dimension or
    /// encoding revision is never silently accepted by another. Purely a declaration of identity,
    /// safe to customize: the enforcement step that compares two profile ids for equality is not
    /// implementer code (see the [`EventSearchEmbedder`] entrypoints below).
    fn encoding_profile_id(&self) -> String {
        self.embedding_space().embedding_space_id
    }

    /// Implementer hook: prepare plain rendered text as an event-search document, using `seal` —
    /// a capability this method never constructs itself, only ever receives from the checked,
    /// library-owned [`EventSearchEmbedder::prepare_document`] entrypoint — to turn the result into
    /// the sealed, caller-facing [`PreparedEventSearchDocument`] via
    /// [`EventSearchSeal::document_builder`]. Exactly one system-added `search_document: ` task
    /// prefix outside the untouched content, plus the exact final token count and original-text
    /// byte boundaries. An embedder that cannot prepare exactly answers
    /// [`CgError::EventSearchPreparationUnsupported`] and never silently delegates to
    /// [`Embedder::embed`] on plain text.
    fn prepare_document_hook(
        &self,
        _text: &str,
        _seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        Err(CgError::EventSearchPreparationUnsupported {
            embedding_space_id: self.embedding_space().embedding_space_id,
        })
    }

    /// Implementer hook: embed a value this same embedder previously prepared through the
    /// checked `prepare_document` entrypoint. It is handed an [`EventSearchInference`] — the model
    /// input plus the library's authorization to run on it — which the checked, library-owned
    /// [`EventSearchEmbedder::embed_prepared_document`] entrypoint creates only after it has verified
    /// the prepared value's encoding profile and its token count. An implementation never needs
    /// to, and cannot, perform those checks itself, and nothing outside this crate can construct
    /// an inference to call this hook with.
    fn embed_prepared_document_hook(
        &self,
        _inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Err(CgError::EventSearchPreparationUnsupported {
            embedding_space_id: self.embedding_space().embedding_space_id,
        })
    }

    /// Implementer hook: embed several values this same embedder prepared through the checked
    /// `prepare_document` entrypoint, in one call, answering one output per inference in the same
    /// order. Reached only through [`EventSearchEmbedder::embed_prepared_documents`], which checks
    /// every prepared value exactly as the single-document entrypoint does. The default embeds one
    /// by one through [`Embedder::embed_prepared_document_hook`]; a model that can run one padded,
    /// masked forward pass over the whole batch overrides it.
    fn embed_prepared_documents_hook(
        &self,
        inferences: &[EventSearchInference],
    ) -> Result<Vec<EmbeddingOutput>> {
        inferences
            .iter()
            .map(|inference| self.embed_prepared_document_hook(inference))
            .collect()
    }

    /// Implementer hook: prepare a plain query as an event-search query, mirroring
    /// [`Embedder::prepare_document_hook`] for the query role — exactly one system-added
    /// `search_query: ` task prefix outside the untouched content, plus the exact final token
    /// count.
    fn prepare_query_hook(
        &self,
        _text: &str,
        _seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        Err(CgError::EventSearchPreparationUnsupported {
            embedding_space_id: self.embedding_space().embedding_space_id,
        })
    }

    /// Implementer hook: embed a value this same embedder previously prepared through the
    /// checked `prepare_query` entrypoint. Mirrors [`Embedder::embed_prepared_document_hook`] for
    /// the query role — handed the same library-minted [`EventSearchInference`], carrying
    /// [`EventSearchRole::Query`], only after every check already ran.
    fn embed_prepared_query_hook(
        &self,
        _inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Err(CgError::EventSearchPreparationUnsupported {
            embedding_space_id: self.embedding_space().embedding_space_id,
        })
    }
}

/// Seals [`EventSearchEmbedder`]. The supertrait lives in a module private to this crate, so no
/// outside crate can name it and therefore none can implement the extension trait either — which
/// is what makes the four checked entrypoints library-owned rather than implementer-owned.
mod sealed {
    /// Declared `pub(super)`, not `pub`: the module is already private, and keeping the trait
    /// itself crate-internal is what makes "no outside crate can implement this" checkable by
    /// reading one line.
    pub(super) trait Sealed {}

    impl<T: super::Embedder + ?Sized> Sealed for T {}
}

/// The checked, library-owned event-search entrypoints, callable on whatever embedder handle a
/// caller happens to be holding: the bundled embedder's own concrete type, a `&dyn Embedder` view,
/// and a shared `Arc<dyn Embedder>` all reach the same four methods.
///
/// This is a SEALED extension trait — its supertrait is private to this crate — granted by exactly
/// one blanket implementation over every [`Embedder`]. So no implementation can override, weaken or
/// substitute these four methods: an outside crate cannot implement this trait at all, and there is
/// no per-type implementation for anyone to diverge from. Only the hooks on [`Embedder`]
/// (`prepare_document_hook`, `prepare_query_hook`, `embed_prepared_document_hook`,
/// `embed_prepared_query_hook`) and [`Embedder::event_search_capability`] remain implementer
/// extension points; preparation, profile verification and the token-ceiling check are not.
///
/// Bring it into scope with `use context_graph::EventSearchEmbedder;`, then call the role methods
/// straight on the embedder you hold:
///
/// ```no_run
/// use std::sync::Arc;
///
/// use context_graph::embedder::nomic::NomicTextEmbedder;
/// use context_graph::{EmbedderConfig, EventSearchEmbedder};
///
/// # fn main() -> context_graph::Result<()> {
/// let model = Arc::new(NomicTextEmbedder::new(
///     EmbedderConfig::default_nomic_offline(),
/// ));
/// // No `&dyn Embedder` coercion: the concrete handle carries the checked role methods too.
/// let passage = model.prepare_document("the incident review closed today")?;
/// let _vector = model.embed_prepared_document(&passage)?;
/// # Ok(())
/// # }
/// ```
// The `private_bounds` lint warns that a `pub` trait is bounded by one an outside crate cannot
// name. Here that is the entire point: `Sealed` being unnameable outside this crate is what makes
// `EventSearchEmbedder` unimplementable outside it, so the four checked entrypoints are the
// library's and never an implementer's.
#[allow(private_bounds)]
pub trait EventSearchEmbedder: Embedder + sealed::Sealed {
    /// Mints a fresh [`EventSearchSeal`] stamped with this embedder's own
    /// [`Embedder::encoding_profile_id`] and whether this value has the exact library-controlled
    /// runtime type for the reserved event-history encoding, then hands it to
    /// [`Embedder::prepare_document_hook`] for the duration of this call.
    fn prepare_document(&self, text: &str) -> Result<PreparedEventSearchDocument> {
        require_event_search_capability(self)?;
        let profile = self.encoding_profile_id();
        let canonical_authority = verify_event_history_authority(self, &profile)?;
        let seal = EventSearchSeal::new(profile, canonical_authority);
        self.prepare_document_hook(text, &seal)
    }

    /// Mirrors [`EventSearchEmbedder::prepare_document`] above for the query role.
    fn prepare_query(&self, text: &str) -> Result<PreparedEventSearchQuery> {
        require_event_search_capability(self)?;
        let profile = self.encoding_profile_id();
        let canonical_authority = verify_event_history_authority(self, &profile)?;
        let seal = EventSearchSeal::new(profile, canonical_authority);
        self.prepare_query_hook(text, &seal)
    }

    /// Refuses, with [`CgError::EventSearchPreparationProfileMismatch`], a prepared document
    /// sealed under a different encoding profile than this embedder's own — checked inline, here,
    /// before [`Embedder::embed_prepared_document_hook`] ever runs. No `Embedder` implementation
    /// can skip or weaken this check: this trait is sealed and blanket-implemented, so there is
    /// nothing for an implementation to override.
    fn embed_prepared_document(
        &self,
        prepared: &PreparedEventSearchDocument,
    ) -> Result<EmbeddingOutput> {
        let capability = require_event_search_capability(self)?;
        let expected = self.encoding_profile_id();
        verify_event_history_identity(
            self,
            &expected,
            prepared.encoding_profile_id(),
            prepared.has_canonical_authority(),
        )?;
        if prepared.encoding_profile_id() != expected {
            return Err(CgError::EventSearchPreparationProfileMismatch {
                expected_profile: expected,
                actual_profile: prepared.encoding_profile_id().to_string(),
            });
        }
        verify_prepared_token_count(capability, prepared.token_count())?;
        let inference = EventSearchInference::new(
            prepared.payload.clone(),
            EventSearchRole::Document,
            expected,
            prepared.has_canonical_authority(),
        );
        self.embed_prepared_document_hook(&inference)
    }

    /// Embeds several prepared documents in one hook call, checking each exactly as
    /// [`EventSearchEmbedder::embed_prepared_document`] does before any inference runs, and
    /// refusing an implementation that answers a different number of outputs than it was given.
    fn embed_prepared_documents(
        &self,
        prepared: &[PreparedEventSearchDocument],
    ) -> Result<Vec<EmbeddingOutput>> {
        let capability = require_event_search_capability(self)?;
        let expected = self.encoding_profile_id();
        let mut inferences = Vec::with_capacity(prepared.len());
        for document in prepared {
            verify_event_history_identity(
                self,
                &expected,
                document.encoding_profile_id(),
                document.has_canonical_authority(),
            )?;
            if document.encoding_profile_id() != expected {
                return Err(CgError::EventSearchPreparationProfileMismatch {
                    expected_profile: expected,
                    actual_profile: document.encoding_profile_id().to_string(),
                });
            }
            verify_prepared_token_count(capability, document.token_count())?;
            inferences.push(EventSearchInference::new(
                document.payload.clone(),
                EventSearchRole::Document,
                expected.clone(),
                document.has_canonical_authority(),
            ));
        }
        if inferences.is_empty() {
            return Ok(Vec::new());
        }
        let outputs = self.embed_prepared_documents_hook(&inferences)?;
        if outputs.len() != inferences.len() {
            return Err(CgError::Engine(format!(
                "a batched event-search embed answered {} outputs for {} documents",
                outputs.len(),
                inferences.len()
            )));
        }
        Ok(outputs)
    }

    /// Mirrors [`EventSearchEmbedder::embed_prepared_document`] above for the query role.
    fn embed_prepared_query(&self, prepared: &PreparedEventSearchQuery) -> Result<EmbeddingOutput> {
        let capability = require_event_search_capability(self)?;
        let expected = self.encoding_profile_id();
        verify_event_history_identity(
            self,
            &expected,
            prepared.encoding_profile_id(),
            prepared.has_canonical_authority(),
        )?;
        if prepared.encoding_profile_id() != expected {
            return Err(CgError::EventSearchPreparationProfileMismatch {
                expected_profile: expected,
                actual_profile: prepared.encoding_profile_id().to_string(),
            });
        }
        verify_prepared_token_count(capability, prepared.token_count())?;
        let inference = EventSearchInference::new(
            prepared.payload.clone(),
            EventSearchRole::Query,
            expected,
            prepared.has_canonical_authority(),
        );
        self.embed_prepared_query_hook(&inference)
    }
}

impl<T: Embedder + ?Sized> EventSearchEmbedder for T {}

/// Whether this embedder is the library's own runtime type for the reserved event-history
/// encoding, and never merely something that declared that profile id. The concrete type is read
/// with [`Any::type_id`] on the value itself — through the vtable when the caller is holding a
/// `dyn Embedder`, and the concrete type directly when it is holding one — so an outside wrapper
/// cannot inherit the authority from an inner Nomic field.
pub(crate) fn verify_event_history_authority<E: Embedder + ?Sized>(
    embedder: &E,
    profile: &str,
) -> Result<bool> {
    if profile != BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID {
        return Ok(false);
    }
    #[cfg(feature = "nomic-embedder")]
    let verified = {
        let concrete_type = Any::type_id(embedder);
        let is_library_type = concrete_type == std::any::TypeId::of::<nomic::NomicTextEmbedder>()
            || concrete_type == std::any::TypeId::of::<nomic::EventHistoryTextEmbedder>();
        is_library_type
            && embedder.embedding_space().dimension
                == crate::embedding_space::event_history_embedding_space().dimension
    };
    #[cfg(not(feature = "nomic-embedder"))]
    let verified = {
        // With no bundled embedder compiled in, no type in this library can hold the canonical
        // event-history identity, so there is nothing to read the handle for.
        let _ = embedder;
        false
    };
    if !verified {
        return Err(CgError::EventHistoryProfileIdentityUnverified {
            declared_profile: profile.to_string(),
        });
    }
    Ok(true)
}

/// Refuses a claim to the canonical event-history identity that this library did not itself vouch
/// for — on either side of the comparison, the embedder about to run inference or the value it was
/// handed. The identity names one exact encoding; a vector produced by anything else under that
/// name would be searched as if it were that encoding.
fn verify_event_history_identity<E: Embedder + ?Sized>(
    embedder: &E,
    expected_profile: &str,
    sealed_profile: &str,
    sealed_has_canonical_authority: bool,
) -> Result<()> {
    verify_event_history_authority(embedder, expected_profile)?;
    if sealed_profile == BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID
        && !sealed_has_canonical_authority
    {
        return Err(CgError::EventHistoryProfileIdentityUnverified {
            declared_profile: sealed_profile.to_string(),
        });
    }
    Ok(())
}

fn require_event_search_capability<E: Embedder + ?Sized>(
    embedder: &E,
) -> Result<EventSearchCapability> {
    embedder
        .event_search_capability()
        .ok_or_else(|| CgError::EventSearchPreparationUnsupported {
            embedding_space_id: embedder.embedding_space().embedding_space_id,
        })
}

/// Refuses a prepared value whose exact token count exceeds the embedder's declared event-search
/// capability before any inference hook runs.
fn verify_prepared_token_count(
    capability: EventSearchCapability,
    token_count: usize,
) -> Result<()> {
    let limit = capability.token_limit();
    if token_count > limit {
        return Err(CgError::PreparedInputExceedsModelLimit { token_count, limit });
    }
    Ok(())
}

#[derive(Clone)]
pub struct EmbedderRegistration {
    pub space: EmbeddingSpaceDecl,
    pub embedder: Arc<dyn Embedder>,
}

impl std::fmt::Debug for EmbedderRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedderRegistration")
            .field("space", &self.space)
            .field(
                "embedder_declared_space",
                &self.embedder.embedding_space().embedding_space_id,
            )
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSource {
    HuggingFace { repo: String },
    LocalPath(PathBuf),
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderConfig {
    pub source: ModelSource,
    pub offline: bool,
    pub cache_dir: Option<PathBuf>,
    pub pinned_revision: Option<String>,
    pub network_policy: String,
}

impl EmbedderConfig {
    pub fn default_nomic() -> Self {
        Self {
            source: ModelSource::HuggingFace {
                repo: NOMIC_DEFAULT_REPOSITORY.to_string(),
            },
            offline: false,
            cache_dir: None,
            pinned_revision: Some(NOMIC_DEFAULT_REVISION.to_string()),
            network_policy: "allow-download".to_string(),
        }
    }

    pub fn default_nomic_offline() -> Self {
        Self {
            source: ModelSource::HuggingFace {
                repo: NOMIC_DEFAULT_REPOSITORY.to_string(),
            },
            offline: true,
            cache_dir: None,
            pinned_revision: Some(NOMIC_DEFAULT_REVISION.to_string()),
            network_policy: "deny-download".to_string(),
        }
    }

    pub fn disabled() -> Self {
        Self {
            source: ModelSource::Disabled,
            offline: true,
            cache_dir: None,
            pinned_revision: None,
            network_policy: "disabled".to_string(),
        }
    }
}
