//! Regression for selecting captured versus explicitly staged interpretation input in search.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CapturedEventAndAnchors, CgError, ConnectorInterpretationRunner,
    ContextSelector, CreateContext, DistanceMetric, EffectiveEventFields, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchBuildRequest,
    EventSearchFilter, EventSearchProfile, EventSearchRebuildRequest, EvidenceKind,
    EvidenceProducer, InstalledInterpretationContract, InterpretationBuildRequest,
    InterpretationGenerationId, ReleasedConnectorInterpretation, ReleasedInterpretationDeclaration,
    Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "search-interpretation-selection";
const SOURCE: &str = "search-interpretation-selection-source";
const SESSION: &str = "search-interpretation-selection-session";
const SEARCH_SPACE: &str = "search-interpretation-selection-768";
const CAPTURED_TEXT: &str = "retained capture: original source wording";
const DERIVED_TEXT: &str = "released interpretation: original source wording";

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "capture-v1".into(),
        anchor_declaration_version: "anchors-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
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
        mapping_contract_id: "search-interpretation-selection-contract".into(),
        mapping_contract_sha256: "search-interpretation-selection-contract-sha256".into(),
    }
}

fn retained_draft() -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: "retained-event".into(),
        source_position: json!({"ordinal": 1}),
        occurred_at: None,
        author: "captured-author".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(CAPTURED_TEXT.into()),
        payload: json!({"retained": true}),
        anchors: json!({"file": "src/lib.rs"}),
        source_remainder: json!({"source_only": "retained"}),
        coverage: EventCoverage::Complete,
    }
}

fn effective_anchors() -> serde_json::Value {
    json!([{
        "key": "file",
        "canonical_value": "src/lib.rs",
        "original_value": "src/lib.rs"
    }])
}

struct ReleasedMapper {
    declaration: ReleasedInterpretationDeclaration,
}

impl ReleasedConnectorInterpretation for ReleasedMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn interpret_event(
        &self,
        captured: &context_graph::ConversationEvent,
    ) -> Result<EffectiveEventFields> {
        let captured_text = captured.text.as_deref().ok_or_else(|| {
            CgError::Engine("selection fixture requires retained captured text".into())
        })?;
        assert_eq!(
            captured_text, CAPTURED_TEXT,
            "runner supplies retained capture"
        );
        Ok(EffectiveEventFields {
            occurred_at: captured.occurred_at,
            author: captured.author.clone(),
            kind: captured.kind.clone(),
            render_role: captured.render_role,
            interaction_ref: captured.interaction_ref.clone(),
            text: Some(format!("released interpretation: {}", &captured_text[18..])),
            payload: captured.payload.clone(),
            anchors: effective_anchors(),
            coverage: captured.coverage,
        })
    }
}

struct PreservationContract {
    declaration: ReleasedInterpretationDeclaration,
}

impl InstalledInterpretationContract for PreservationContract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn validate(
        &self,
        captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        if captured.event.text.as_deref() != Some(CAPTURED_TEXT)
            || captured.anchors.len() != 1
            || candidate.text.as_deref() != Some(DERIVED_TEXT)
            || candidate.occurred_at != captured.event.occurred_at
            || candidate.author != captured.event.author
            || candidate.kind != captured.event.kind
            || candidate.render_role != captured.event.render_role
            || candidate.interaction_ref != captured.event.interaction_ref
            || candidate.payload != captured.event.payload
            || candidate.anchors != effective_anchors()
            || candidate.coverage != captured.event.coverage
        {
            return Err(CgError::Engine(
                "selection fixture preservation contract refused derived value".into(),
            ));
        }
        Ok(())
    }
}

struct SelectionEmbedder;

impl SelectionEmbedder {
    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output() -> EmbeddingOutput {
        let mut vector = vec![0.0; 768];
        vector[0] = 1.0;
        EmbeddingOutput {
            embedding_space_id: SEARCH_SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "search-interpretation-selection".into(),
                model_name: "deterministic-768".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }
}

impl Embedder for SelectionEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search uses explicit document/query preparation")
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
            model_name: "selection".into(),
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

    fn encoding_profile_id(&self) -> String {
        "search-interpretation-selection-v1".into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(4_096)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let payload: Vec<u32> = text.bytes().map(u32::from).collect();
        seal.document_builder()
            .token_count(payload.len())
            .byte_boundaries((0..text.len()).map(|index| (index, index + 1)).collect())
            .payload(payload)
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        let payload: Vec<u32> = text.bytes().map(u32::from).collect();
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }

    fn embed_prepared_query_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }
}

fn search_build_request(
    context_id: context_graph::ContextId,
    embedder: EmbedderRegistration,
) -> EventSearchBuildRequest {
    EventSearchBuildRequest {
        context_id,
        embedder,
        profile: EventSearchProfile::default(),
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch size"),
        max_passages_per_batch: NonZeroU32::new(8).expect("nonzero batch size"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte size"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease duration"),
    }
}

fn staged_rebuild_request(
    context_id: context_graph::ContextId,
    interpretation_generation_id: InterpretationGenerationId,
    embedder: EmbedderRegistration,
) -> EventSearchRebuildRequest {
    EventSearchRebuildRequest {
        context_id,
        interpretation_generation_id: Some(interpretation_generation_id),
        embedder,
        profile: EventSearchProfile::default(),
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch size"),
        max_passages_per_batch: NonZeroU32::new(8).expect("nonzero batch size"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte size"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease duration"),
    }
}

fn params(context_id: context_graph::ContextId) -> HashMap<String, Value> {
    HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))])
}

#[test]
fn ready_unactivated_interpretation_changes_only_the_explicit_staged_search_generation() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(SelectionEmbedder);
    let registration = embedder.registration();
    let config = StoreConfig {
        db_path: root.path().join("selection.db"),
        extra_embedding_spaces: vec![embedder.embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store = Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
        .expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "search interpretation selection".into(),
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
            vec![retained_draft()],
        ))
        .expect("retain captured source event");
    drop(store);

    let mapper: Arc<dyn ReleasedConnectorInterpretation> = Arc::new(ReleasedMapper {
        declaration: release(),
    });
    let contract: Arc<dyn InstalledInterpretationContract> = Arc::new(PreservationContract {
        declaration: release(),
    });
    let interpretation = ConnectorInterpretationRunner::build(InterpretationBuildRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        mapper,
        contract,
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch size"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte size"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease duration"),
    })
    .expect("released runner builds a ready interpretation from retained capture");
    assert_eq!(
        interpretation.state,
        context_graph::InterpretationGenerationState::Ready
    );
    assert_eq!(interpretation.interpreted_event_count, 1);
    assert!(interpretation.completed_at.is_some());

    let store = Store::open_with_embedder_registrations(config, vec![registration.clone()])
        .expect("reopen public search store");
    let before_status = store
        .interpretation_status(ContextSelector::one(context_id), CONNECTOR.into())
        .expect("read interpretation head");
    assert_eq!(before_status.contexts[0].active_generation_id, None);

    let ordinary = store
        .catch_up_event_search(search_build_request(context_id, registration.clone()))
        .expect("ordinary catch-up uses captured base while the interpretation is unactivated");
    assert_eq!(ordinary.interpretation_generation_ids, Vec::new());
    assert!(ordinary.active_generation_changed);
    assert_eq!(ordinary.source_watermark.event_count, 1);
    assert_eq!(ordinary.rendered_event_count, 1);
    assert_eq!(ordinary.embedded_event_count, 1);

    let before = store
        .find_event_history(
            ContextSelector::one(context_id),
            "original source wording",
            EventSearchFilter::default(),
        )
        .expect("public search sees ordinary active generation");
    assert_eq!(before.hits.len(), 1);
    assert_eq!(before.hits[0].generation_id, ordinary.generation_id);
    assert!(before.hits[0].text.contains(CAPTURED_TEXT));
    assert!(!before.hits[0].text.contains(DERIVED_TEXT));

    let ordinary_head = store
        .execute_sql_for_test(
            "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
            &params(context_id),
        )
        .expect("inspect ordinary search head");
    assert_eq!(
        ordinary_head.rows,
        vec![vec![Value::Uuid(ordinary.generation_id.as_uuid())]]
    );
    let interpretation_head = store
        .execute_sql_for_test(
            "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("connector_id".into(), Value::Text(CONNECTOR.into())),
            ]),
        )
        .expect("inspect unactivated interpretation head");
    assert!(interpretation_head.rows.is_empty());

    let staged = store
        .rebuild_event_search(staged_rebuild_request(
            context_id,
            interpretation.generation_id,
            registration,
        ))
        .expect("explicit staged rebuild accepts the exact ready interpretation generation");
    assert_ne!(staged.generation_id, ordinary.generation_id);
    assert_eq!(
        staged.interpretation_generation_ids,
        vec![interpretation.generation_id]
    );
    assert!(!staged.active_generation_changed);
    assert_eq!(staged.source_watermark, ordinary.source_watermark);
    assert_eq!(staged.rendered_event_count, 1);
    assert_eq!(staged.embedded_event_count, 1);

    let staged_membership = store
        .execute_sql_for_test(
            "SELECT effective_interpretation_generation_id FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(staged.generation_id.as_uuid())),
            ]),
        )
        .expect("inspect staged sealed membership");
    assert_eq!(
        staged_membership.rows,
        vec![vec![Value::Uuid(interpretation.generation_id.as_uuid())]]
    );
    let staged_text = store
        .execute_sql_for_test(
            "SELECT rendered_text FROM event_search_passages WHERE context_id = $context_id AND generation_id = $generation_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(staged.generation_id.as_uuid())),
            ]),
        )
        .expect("inspect staged derived passage");
    assert_eq!(
        staged_text.rows,
        vec![vec![Value::Text(format!(
            "User [author=\"captured-author\"]: {DERIVED_TEXT}"
        ))]]
    );

    let after = store
        .find_event_history(
            ContextSelector::one(context_id),
            "original source wording",
            EventSearchFilter::default(),
        )
        .expect("public search remains on ordinary active generation");
    assert_eq!(after.hits.len(), 1);
    assert_eq!(after.hits[0].generation_id, ordinary.generation_id);
    assert_eq!(after.hits[0].text, before.hits[0].text);
    assert!(after.hits[0].text.contains(CAPTURED_TEXT));
    assert!(!after.hits[0].text.contains(DERIVED_TEXT));
    assert_eq!(
        store
            .interpretation_status(ContextSelector::one(context_id), CONNECTOR.into())
            .expect("read interpretation head after staged rebuild")
            .contexts[0]
            .active_generation_id,
        None
    );
    assert_eq!(
        store
            .execute_sql_for_test(
                "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
                &params(context_id),
            )
            .expect("inspect search head after staged rebuild")
            .rows,
        ordinary_head.rows
    );
}
