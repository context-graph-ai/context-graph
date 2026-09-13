use crate::{DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind};

pub type EmbeddingSpaceDecl = EmbeddingSpace;

pub struct BuiltinSpaceRegistry;

impl BuiltinSpaceRegistry {
    pub const TEXT_DEFAULT_ID: &'static str = "text_default_nomic_v1_5_768";
    pub const TEXT_DEFAULT_MODEL_FAMILY: &'static str = "text";
    pub const TEXT_DEFAULT_MODEL_NAME: &'static str = "nomic-embed-text-v1.5";
    pub const TEXT_DEFAULT_MODEL_VERSION: &'static str = "1.5";
    pub const TEXT_DEFAULT_DIMENSION: usize = 768;
    /// The dedicated event-history search space: role-aware (document/query) Nomic preparation,
    /// never compared with [`Self::TEXT_DEFAULT_ID`].
    pub const EVENT_HISTORY_DOCUMENT_QUERY_ID: &'static str =
        "event_history_nomic_v1_5_document_query_v1_768";

    pub(crate) fn text_space_id_for_dimension(dimension: usize) -> String {
        if dimension == Self::TEXT_DEFAULT_DIMENSION {
            Self::TEXT_DEFAULT_ID.to_string()
        } else {
            format!("text_default_nomic_v1_5_{dimension}")
        }
    }
}

pub fn default_text_embedding_space() -> EmbeddingSpace {
    text_nomic_space(BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION, true)
}

/// The canonical, production declaration of the dedicated event-history search space: 768
/// dimensions, cosine metric, the same Nomic text-embedding model identity as
/// [`default_text_embedding_space`], the `Text` evidence kind, and no default claim on any
/// evidence kind (`default_for_evidence_kind: None`) — it is never the implicit destination for
/// plain `Embedder::embed` calls, only for the role-aware event-search preparation surface.
///
/// Its sole binding is the dedicated event-history passage vector. It never contends with the
/// Digest/Evidence [`default_text_embedding_space`] for `evidence.vector_text` or
/// `digests.vector_text`.
pub fn event_history_embedding_space() -> EmbeddingSpace {
    let id = BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.to_string();
    EmbeddingSpace {
        embedding_space_id: id.clone(),
        dimension: BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_FAMILY.to_string(),
        model_name: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_NAME.to_string(),
        model_version: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_VERSION.to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: id,
            table_name: "event_search_passages".to_string(),
            vector_column: "vector_text".to_string(),
        }],
    }
}

pub(crate) fn text_nomic_space(dimension: usize, default_for_text: bool) -> EmbeddingSpace {
    let id = BuiltinSpaceRegistry::text_space_id_for_dimension(dimension);
    let mut bindings = vec![EmbeddingSpaceBinding {
        embedding_space_id: id.clone(),
        table_name: "evidence".to_string(),
        vector_column: "vector_text".to_string(),
    }];
    if dimension == BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION {
        bindings.push(EmbeddingSpaceBinding {
            embedding_space_id: id.clone(),
            table_name: "digests".to_string(),
            vector_column: "vector_text".to_string(),
        });
    }
    EmbeddingSpace {
        embedding_space_id: id,
        dimension,
        metric: DistanceMetric::Cosine,
        model_family: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_FAMILY.to_string(),
        model_name: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_NAME.to_string(),
        model_version: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_VERSION.to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
        default_for_evidence_kind: default_for_text.then_some(EvidenceKind::Text),
        bindings,
    }
}
