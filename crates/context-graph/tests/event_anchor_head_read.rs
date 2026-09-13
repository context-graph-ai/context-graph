//! The public exact-anchor read follows the active interpretation head.
//!
//! Captured `event_anchors` answer for every event no active interpretation covers; once a
//! released interpretation is activated, its complete anchor set answers for the events it covers.
//! Matches are ordered by connector, source stream and sequence; the range's sequence bounds apply
//! inside each stream and its limit applies once to the whole ordered answer.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AnchorKey, AnchorValue, AppendEventBatch, CapturedEventAndAnchors, CgError,
    ConnectorInterpretationRunner, ContextId, CreateContext, DistanceMetric, EffectiveEventFields,
    Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace,
    EmbeddingSpaceBinding, EventAdapterDeclaration, EventCoverage, EventDraft, EventRange,
    EventRenderRole, EventSearchProfile, EventSearchRebuildRequest, EvidenceKind, EvidenceProducer,
    InstalledInterpretationContract, InterpretationActivationRequest, InterpretationBuildRequest,
    ReleasedConnectorInterpretation, ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use serde_json::json;

const CONNECTOR: &str = "anchor-head";
const OTHER_CONNECTOR: &str = "anchor-other";
const SEARCH_SPACE: &str = "anchor-head-768";

struct DeterministicEmbedder;

impl DeterministicEmbedder {
    fn output() -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SEARCH_SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "deterministic-768".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![1.0; 768],
        }
    }
}

impl Embedder for DeterministicEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search uses prepared inputs")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: SEARCH_SPACE.into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "deterministic-768".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: SEARCH_SPACE.into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
        }
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(1_536)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        seal.document_builder()
            .token_count(text.len())
            .byte_boundaries((0..text.len()).map(|index| (index, index + 1)).collect())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        seal.query_builder()
            .token_count(text.len())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }

    fn embed_prepared_document_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }

    fn embed_prepared_query_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }
}

fn declaration(connector: &str) -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: connector.into(),
        adapter_version: "capture-v1".into(),
        anchor_declaration_version: "anchors-v1".into(),
        anchor_schema: json!({"type": "object", "properties": {"file": {"type": "string"}}}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(key: &str, ordinal: u64, file: &str) -> EventDraft {
    EventDraft {
        session_ref: "anchor-session".into(),
        source_event_key: key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "captured-author".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("captured {key}")),
        payload: json!({}),
        anchors: json!({"file": file}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

/// The value text the read matches: the canonical JSON text capture writes for a string anchor.
fn file(path: &str) -> AnchorValue {
    AnchorValue(format!("\"{path}\""))
}

fn all(limit: u32) -> EventRange {
    EventRange::Count {
        after_seq: None,
        limit: NonZeroU32::new(limit).expect("nonzero limit"),
    }
}

fn by_file(
    store: &Store,
    context_id: ContextId,
    path: &str,
    range: EventRange,
) -> Vec<(String, String, u64)> {
    store
        .list_events_by_anchor(context_id, AnchorKey("file".into()), file(path), range)
        .expect("exact anchor read")
        .into_iter()
        .map(|event| (event.connector_id, event.source_instance_ref, event.seq))
        .collect()
}

fn open(config: &StoreConfig) -> Store {
    Store::open_with_embedder_registrations(
        config.clone(),
        vec![EmbedderRegistration {
            space: DeterministicEmbedder.embedding_space(),
            embedder: Arc::new(DeterministicEmbedder),
        }],
    )
    .expect("open fixture store")
}

fn fixture() -> (tempfile::TempDir, StoreConfig, ContextId, ContextId) {
    let root = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: root.path().join("anchor-head.db"),
        extra_embedding_spaces: vec![DeterministicEmbedder.embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store = open(&config);
    let create = |name: &str| {
        store
            .create_context(CreateContext {
                name: name.into(),
                labels: Vec::new(),
                properties: Default::default(),
            })
            .expect("create Context")
            .id
    };
    let context_id = create("anchor head");
    let other_context_id = create("anchor elsewhere");
    // Appended stream `b-stream` first so the answer's order cannot be append order.
    for (source, prefix) in [("b-stream", "b"), ("a-stream", "a")] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(CONNECTOR),
                source,
                vec![
                    draft(&format!("{prefix}1"), 1, "src/old.rs"),
                    draft(&format!("{prefix}2"), 2, "src/old.rs"),
                ],
            ))
            .expect("retain connector events");
    }
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(OTHER_CONNECTOR),
            "other-stream",
            vec![draft("o1", 1, "src/old.rs")],
        ))
        .expect("retain another connector's event");
    store
        .append_event_batch(AppendEventBatch::live(
            other_context_id,
            declaration(CONNECTOR),
            "elsewhere-stream",
            vec![draft("e1", 1, "src/elsewhere.rs")],
        ))
        .expect("retain an event in another Context");
    drop(store);
    (root, config, context_id, other_context_id)
}

fn release() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        target_adapter_version: "capture-v1".into(),
        target_anchor_declaration_version: "anchors-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "capture-v1".into(),
            anchor_declaration_version: "anchors-v1".into(),
        }],
        mapping_contract_id: "anchor-head-contract".into(),
        mapping_contract_sha256: "anchor-head-contract-v1".into(),
    }
}

/// Keeps every captured field and replaces the complete anchor set with `file = src/new.rs`.
struct RenamingMapper(ReleasedInterpretationDeclaration);

impl ReleasedConnectorInterpretation for RenamingMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.0
    }

    fn interpret_event(
        &self,
        captured: &context_graph::ConversationEvent,
    ) -> Result<EffectiveEventFields> {
        Ok(EffectiveEventFields {
            occurred_at: captured.occurred_at,
            author: captured.author.clone(),
            kind: captured.kind.clone(),
            render_role: captured.render_role,
            interaction_ref: captured.interaction_ref.clone(),
            text: captured.text.clone(),
            payload: captured.payload.clone(),
            anchors: json!([{
                "key": "file",
                "canonical_value": "src/new.rs",
                "original_value": "src/new.rs"
            }]),
            coverage: captured.coverage,
        })
    }
}

struct PreservingContract(ReleasedInterpretationDeclaration);

impl InstalledInterpretationContract for PreservingContract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.0
    }

    fn validate(
        &self,
        captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        if candidate.author == captured.event.author
            && candidate.kind == captured.event.kind
            && candidate.render_role == captured.event.render_role
            && candidate.text == captured.event.text
            && candidate.payload == captured.event.payload
            && candidate.coverage == captured.event.coverage
        {
            Ok(())
        } else {
            Err(CgError::Engine(
                "anchor fixture refused a changed capture".into(),
            ))
        }
    }
}

fn activate_renaming_release(config: &StoreConfig, context_id: ContextId) {
    let mapper: Arc<dyn ReleasedConnectorInterpretation> = Arc::new(RenamingMapper(release()));
    let contract: Arc<dyn InstalledInterpretationContract> =
        Arc::new(PreservingContract(release()));
    let interpretation = ConnectorInterpretationRunner::build(InterpretationBuildRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        mapper: mapper.clone(),
        contract: contract.clone(),
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero bytes"),
        lease_duration_ms: NonZeroU64::new(10_000).expect("nonzero lease"),
    })
    .expect("build the renaming interpretation");
    let store = open(config);
    let search = store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: Some(interpretation.generation_id),
            embedder: EmbedderRegistration {
                space: DeterministicEmbedder.embedding_space(),
                embedder: Arc::new(DeterministicEmbedder),
            },
            profile: EventSearchProfile::default(),
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero bytes"),
            lease_duration_ms: NonZeroU64::new(10_000).expect("nonzero lease"),
        })
        .expect("build the staged replacement search");
    assert_eq!(
        by_file(&store, context_id, "src/new.rs", all(10)),
        Vec::new(),
        "a built but unactivated interpretation does not answer the exact read"
    );
    drop(store);
    ConnectorInterpretationRunner::activate(InterpretationActivationRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        mapper,
        contract,
        interpretation_generation_id: interpretation.generation_id,
        replacement_search_generation_id: search.generation_id,
    })
    .expect("activate the renaming interpretation");
}

#[test]
fn captured_anchors_answer_when_no_interpretation_is_active() {
    let (_root, config, context_id, _) = fixture();
    let store = open(&config);
    assert_eq!(
        by_file(&store, context_id, "src/old.rs", all(10)),
        vec![
            (CONNECTOR.to_string(), "a-stream".to_string(), 1),
            (CONNECTOR.to_string(), "a-stream".to_string(), 2),
            (CONNECTOR.to_string(), "b-stream".to_string(), 1),
            (CONNECTOR.to_string(), "b-stream".to_string(), 2),
            (OTHER_CONNECTOR.to_string(), "other-stream".to_string(), 1),
        ],
        "every captured match, ordered by connector, source stream and sequence"
    );
    assert_eq!(
        by_file(&store, context_id, "src/new.rs", all(10)),
        Vec::new()
    );
}

#[test]
fn the_active_interpretation_replaces_the_anchor_set_of_the_events_it_covers() {
    let (_root, config, context_id, _) = fixture();
    activate_renaming_release(&config, context_id);
    let store = open(&config);
    assert_eq!(
        by_file(&store, context_id, "src/new.rs", all(10)),
        vec![
            (CONNECTOR.to_string(), "a-stream".to_string(), 1),
            (CONNECTOR.to_string(), "a-stream".to_string(), 2),
            (CONNECTOR.to_string(), "b-stream".to_string(), 1),
            (CONNECTOR.to_string(), "b-stream".to_string(), 2),
        ],
        "covered events answer with the active interpretation's anchors"
    );
    assert_eq!(
        by_file(&store, context_id, "src/old.rs", all(10)),
        vec![(OTHER_CONNECTOR.to_string(), "other-stream".to_string(), 1)],
        "captured anchors still answer for the connector without an active interpretation, \
         and no longer answer for the covered events"
    );
}

#[test]
fn a_cross_context_guess_returns_nothing() {
    let (_root, config, context_id, other_context_id) = fixture();
    let store = open(&config);
    assert_eq!(
        by_file(&store, other_context_id, "src/old.rs", all(10)),
        Vec::new(),
        "a value retained only in another Context is not visible from this one"
    );
    assert_eq!(
        by_file(&store, context_id, "src/elsewhere.rs", all(10)),
        Vec::new(),
        "and the other Context's value is not visible from here"
    );
    assert_eq!(
        by_file(&store, other_context_id, "src/elsewhere.rs", all(10)),
        vec![(CONNECTOR.to_string(), "elsewhere-stream".to_string(), 1)]
    );
}

#[test]
fn sequence_bounds_apply_per_stream_and_the_limit_applies_once_over_the_ordered_answer() {
    let (_root, config, context_id, _) = fixture();
    let store = open(&config);
    assert_eq!(
        by_file(&store, context_id, "src/old.rs", all(3)),
        vec![
            (CONNECTOR.to_string(), "a-stream".to_string(), 1),
            (CONNECTOR.to_string(), "a-stream".to_string(), 2),
            (CONNECTOR.to_string(), "b-stream".to_string(), 1),
        ],
        "the limit cuts the whole ordered answer once"
    );
    assert_eq!(
        by_file(
            &store,
            context_id,
            "src/old.rs",
            EventRange::Count {
                after_seq: Some(1),
                limit: NonZeroU32::new(10).expect("nonzero limit"),
            },
        ),
        vec![
            (CONNECTOR.to_string(), "a-stream".to_string(), 2),
            (CONNECTOR.to_string(), "b-stream".to_string(), 2),
        ],
        "after_seq filters inside every stream"
    );
    assert_eq!(
        by_file(
            &store,
            context_id,
            "src/old.rs",
            EventRange::Sequence {
                first_seq: 1,
                last_seq_inclusive: 1,
                limit: NonZeroU32::new(10).expect("nonzero limit"),
            },
        ),
        vec![
            (CONNECTOR.to_string(), "a-stream".to_string(), 1),
            (CONNECTOR.to_string(), "b-stream".to_string(), 1),
            (OTHER_CONNECTOR.to_string(), "other-stream".to_string(), 1),
        ],
        "sequence bounds filter inside every stream"
    );
}
