//! regression controls for refusing invalid atomic interpretation/search activation pairs.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CapturedEventAndAnchors, CgError, ConnectorInterpretationRunner, ContextId,
    CreateContext, DistanceMetric, EffectiveEventFields, Embedder, EmbedderRegistration,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchProfile,
    EventSearchRebuildRequest, EvidenceKind, EvidenceProducer, InstalledInterpretationContract,
    InterpretationActivationRefusalReason, InterpretationActivationRequest,
    InterpretationBuildReceipt, InterpretationBuildRequest, ReleasedConnectorInterpretation,
    ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "activation-red";
const SOURCE: &str = "activation-red-source";
const SEARCH_SPACE: &str = "activation-red-768";

struct ActivationEmbedder;

impl ActivationEmbedder {
    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

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

impl Embedder for ActivationEmbedder {
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

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "capture-v1".into(),
        anchor_declaration_version: "anchors-v1".into(),
        anchor_schema: json!({"type": "object", "properties": {"file": {"type": "string"}}}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(key: &str, ordinal: u64) -> EventDraft {
    EventDraft {
        session_ref: "activation-red-session".into(),
        source_event_key: key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "captured-author".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("captured {key}")),
        payload: json!({"captured": key}),
        anchors: json!({"file": "src/lib.rs"}),
        source_remainder: json!({"retained": key}),
        coverage: EventCoverage::Complete,
    }
}

fn release(target: &str) -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: target.into(),
        target_adapter_version: "capture-v1".into(),
        target_anchor_declaration_version: "anchors-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "capture-v1".into(),
            anchor_declaration_version: "anchors-v1".into(),
        }],
        mapping_contract_id: "activation-red-contract".into(),
        mapping_contract_sha256: format!("activation-red-contract-{target}"),
    }
}

struct Mapper {
    declaration: ReleasedInterpretationDeclaration,
}

impl ReleasedConnectorInterpretation for Mapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
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
                "canonical_value": "src/lib.rs",
                "original_value": "src/lib.rs"
            }]),
            coverage: captured.coverage,
        })
    }
}

struct Contract {
    declaration: ReleasedInterpretationDeclaration,
}

impl InstalledInterpretationContract for Contract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn validate(
        &self,
        captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        if candidate.occurred_at == captured.event.occurred_at
            && candidate.author == captured.event.author
            && candidate.kind == captured.event.kind
            && candidate.render_role == captured.event.render_role
            && candidate.interaction_ref == captured.event.interaction_ref
            && candidate.text == captured.event.text
            && candidate.payload == captured.event.payload
            && candidate.coverage == captured.event.coverage
        {
            Ok(())
        } else {
            Err(CgError::Engine(
                "activation fixture refused changed capture".into(),
            ))
        }
    }
}

fn installed(
    target: &str,
) -> (
    Arc<dyn ReleasedConnectorInterpretation>,
    Arc<dyn InstalledInterpretationContract>,
) {
    let declaration = release(target);
    (
        Arc::new(Mapper {
            declaration: declaration.clone(),
        }),
        Arc::new(Contract { declaration }),
    )
}

fn build(
    config: StoreConfig,
    context_id: ContextId,
    target: &str,
    mapper: Arc<dyn ReleasedConnectorInterpretation>,
    contract: Arc<dyn InstalledInterpretationContract>,
) -> InterpretationBuildReceipt {
    ConnectorInterpretationRunner::build(InterpretationBuildRequest {
        store: config,
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: target.into(),
        mapper,
        contract,
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero bytes"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease"),
    })
    .expect("build a ready staged interpretation")
}

fn rebuild(
    store: &Store,
    context_id: ContextId,
    interpretation_generation_id: context_graph::InterpretationGenerationId,
    registration: EmbedderRegistration,
) -> context_graph::EventSearchBuildReceipt {
    store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: Some(interpretation_generation_id),
            embedder: registration,
            profile: EventSearchProfile::default(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero bytes"),
            lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease"),
        })
        .expect("build the explicit staged search generation")
}

fn heads(store: &Store, context_id: ContextId) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let interpretation = store
        .execute_sql_for_test(
            "SELECT generation_id, activated_at FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("connector_id".into(), Value::Text(CONNECTOR.into())),
            ]),
        )
        .expect("snapshot interpretation head")
        .rows;
    let search = store
        .execute_sql_for_test(
            "SELECT generation_id, activated_at FROM event_search_heads WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("snapshot search head")
        .rows;
    (interpretation, search)
}

fn fixture() -> (
    tempfile::TempDir,
    StoreConfig,
    ContextId,
    Arc<ActivationEmbedder>,
    EmbedderRegistration,
) {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(ActivationEmbedder);
    let registration = embedder.registration();
    let config = StoreConfig {
        db_path: root.path().join("activation.db"),
        extra_embedding_spaces: vec![embedder.embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store = Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
        .expect("open fixture store");
    let context_id = store
        .create_context(CreateContext {
            name: "activation red".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![draft("initial", 1)],
        ))
        .expect("retain initial event");
    drop(store);
    (root, config, context_id, embedder, registration)
}

#[test]
fn activation_refuses_a_staged_pair_after_the_source_advances_without_moving_heads() {
    let (_root, config, context_id, _embedder, registration) = fixture();
    let (mapper, contract) = installed("released-v1");
    let interpretation = build(
        config.clone(),
        context_id,
        "released-v1",
        mapper.clone(),
        contract.clone(),
    );
    let store = Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
        .expect("open staged search store");
    let search = rebuild(
        &store,
        context_id,
        interpretation.generation_id,
        registration.clone(),
    );
    assert_eq!(
        search.interpretation_generation_ids,
        vec![interpretation.generation_id]
    );
    assert!(!search.active_generation_changed);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![draft("late", 2)],
        ))
        .expect("append after both staged generations finish");
    let before = heads(&store, context_id);
    assert_eq!(before, (Vec::new(), Vec::new()));
    drop(store);

    let error = ConnectorInterpretationRunner::activate(InterpretationActivationRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        mapper,
        contract,
        interpretation_generation_id: interpretation.generation_id,
        replacement_search_generation_id: search.generation_id,
    })
    .expect_err("the stale staged pair must be refused by the activation boundary");
    assert!(matches!(
        error,
        CgError::InterpretationActivationRefused {
            reason: InterpretationActivationRefusalReason::SourceWatermarkMismatch,
            ..
        }
    ));

    let store = Store::open_with_embedder_registrations(config, vec![registration])
        .expect("reopen after refused activation");
    assert_eq!(
        heads(&store, context_id),
        before,
        "refusal leaves both heads unchanged"
    );
}

#[test]
fn activation_refuses_a_search_generation_from_the_wrong_interpretation_snapshot_without_moving_heads()
 {
    let (_root, config, context_id, _embedder, registration) = fixture();
    let (mapper_v1, contract_v1) = installed("released-v1");
    let interpretation_v1 = build(
        config.clone(),
        context_id,
        "released-v1",
        mapper_v1.clone(),
        contract_v1.clone(),
    );
    let (mapper_v2, contract_v2) = installed("released-v2");
    let interpretation_v2 = build(
        config.clone(),
        context_id,
        "released-v2",
        mapper_v2,
        contract_v2,
    );
    let store = Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
        .expect("open staged search store");
    let wrong_search = rebuild(
        &store,
        context_id,
        interpretation_v2.generation_id,
        registration.clone(),
    );
    assert_eq!(
        wrong_search.interpretation_generation_ids,
        vec![interpretation_v2.generation_id]
    );
    assert!(!wrong_search.active_generation_changed);
    let before = heads(&store, context_id);
    assert_eq!(before, (Vec::new(), Vec::new()));
    drop(store);

    let error = ConnectorInterpretationRunner::activate(InterpretationActivationRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        mapper: mapper_v1,
        contract: contract_v1,
        interpretation_generation_id: interpretation_v1.generation_id,
        replacement_search_generation_id: wrong_search.generation_id,
    })
    .expect_err("a search snapshot for another interpretation must be refused");
    assert!(matches!(
        error,
        CgError::InterpretationActivationRefused {
            reason: InterpretationActivationRefusalReason::SearchInterpretationMismatch,
            ..
        }
    ));

    let store = Store::open_with_embedder_registrations(config, vec![registration])
        .expect("reopen after refused activation");
    assert_eq!(
        heads(&store, context_id),
        before,
        "refusal leaves both heads unchanged"
    );
}
