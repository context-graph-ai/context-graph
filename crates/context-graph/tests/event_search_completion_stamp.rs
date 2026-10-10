//! Completion reuses the run's certified membership capture only while the Context's event revision
//! and the bound interpretation stamp both still match what the capture was certified against.
//! This row isolates the stamp: after certification the bound interpretation advances while it stays
//! a completely sealed ready input, no event is appended and the binding is unchanged. Completion
//! must capture again and publish the stamp it verified.
//!
//! Authored by writecost-opus-01 (the separate test author was closed; the source reviewer specified
//! the expectation). Fixture shared with event_search_interpretation_drift.rs.

#![cfg(feature = "test-support")]
#![allow(dead_code)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::test_support::{
    advance_bound_interpretation_stamp_for_test, after_next_event_search_batch_commit_for_test,
    take_event_search_membership_captures_for_test,
};
use context_graph::{
    AppendEventBatch, CapturedEventAndAnchors, CgError, ConnectorInterpretationRunner, ContextId,
    ContextSelector, CreateContext, DistanceMetric, EffectiveEventFields, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchProfile,
    EventSearchProjectionRunner, EventSearchRebuildRequest, EventSearchRunRequest, EvidenceKind,
    EvidenceProducer, InstalledInterpretationContract, InterpretationActivationRequest,
    InterpretationBuildRequest, InterpretationGenerationId, ReleasedConnectorInterpretation,
    ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "wc-drift";
const SOURCE: &str = "wc-drift-source";
const SPACE: &str = "wc-drift-768";

struct DriftEmbedder;

impl DriftEmbedder {
    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output() -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
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

impl Embedder for DriftEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search uses prepared inputs")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "deterministic-768".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: SPACE.into(),
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
        session_ref: "wc-drift-session".into(),
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
        mapping_contract_id: "wc-drift-contract".into(),
        mapping_contract_sha256: "wc-drift-contract-released-v1".into(),
    }
}

struct Mapper(ReleasedInterpretationDeclaration);

impl ReleasedConnectorInterpretation for Mapper {
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
                "canonical_value": "src/lib.rs",
                "original_value": "src/lib.rs"
            }]),
            coverage: captured.coverage,
        })
    }
}

struct Contract(ReleasedInterpretationDeclaration);

impl InstalledInterpretationContract for Contract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.0
    }

    fn validate(
        &self,
        captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        if candidate.text == captured.event.text && candidate.payload == captured.event.payload {
            Ok(())
        } else {
            Err(CgError::Engine("drift fixture refused changed capture".into()))
        }
    }
}

fn installed() -> (
    Arc<dyn ReleasedConnectorInterpretation>,
    Arc<dyn InstalledInterpretationContract>,
) {
    (Arc::new(Mapper(release())), Arc::new(Contract(release())))
}

fn sql(store: &Store, statement: &str, params: Vec<(&str, Value)>) -> contextdb_engine::QueryResult {
    let params = params
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect::<HashMap<_, _>>();
    store
        .execute_sql_for_test(statement, &params)
        .unwrap_or_else(|error| panic!("`{statement}` failed: {error:?}"))
}

fn run(config: &StoreConfig, context_id: ContextId) -> Result<context_graph::EventSearchBuildReceipt> {
    let embedder = Arc::new(DriftEmbedder);
    EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: EventSearchProfile::default(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(2).expect("nonzero batch"),
        max_passages_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero bytes"),
        lease_duration_ms: NonZeroU64::new(30_000).expect("nonzero lease"),
    })
}

struct Fixture {
    _root: tempfile::TempDir,
    config: StoreConfig,
    context_id: ContextId,
    interpretation: InterpretationGenerationId,
}

fn open(config: &StoreConfig) -> Store {
    let embedder = Arc::new(DriftEmbedder);
    Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
        .expect("open Store")
}

/// A Context whose head search generation is bound to an activated, ready interpretation.
fn bound_fixture(events: u64) -> Fixture {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(DriftEmbedder);
    let config = StoreConfig {
        db_path: root.path().join("drift.db"),
        extra_embedding_spaces: vec![embedder.embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store = open(&config);
    let context_id = store
        .create_context(CreateContext {
            name: "interpretation drift".into(),
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
            (1..=events).map(|n| draft(&format!("initial-{n}"), n)).collect(),
        ))
        .expect("retain initial events");
    drop(store);
    let (mapper, contract) = installed();
    let interpretation = ConnectorInterpretationRunner::build(InterpretationBuildRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        mapper: mapper.clone(),
        contract: contract.clone(),
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero"),
        lease_duration_ms: NonZeroU64::new(30_000).expect("nonzero"),
    })
    .expect("build a ready interpretation");
    let store = open(&config);
    let staged = store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: Some(interpretation.generation_id),
            embedder: embedder.registration(),
            profile: EventSearchProfile::default(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero"),
            max_passages_per_batch: NonZeroU32::new(8).expect("nonzero"),
            max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero"),
            lease_duration_ms: NonZeroU64::new(30_000).expect("nonzero"),
        })
        .expect("stage the interpretation-bound search generation");
    drop(store);
    ConnectorInterpretationRunner::activate(InterpretationActivationRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "released-v1".into(),
        mapper,
        contract,
        interpretation_generation_id: interpretation.generation_id,
        replacement_search_generation_id: staged.generation_id,
    })
    .expect("activate the interpretation with its search generation");
    Fixture { _root: root, config, context_id, interpretation: interpretation.generation_id }
}

fn head_generation(store: &Store, context_id: ContextId) -> uuid::Uuid {
    match &sql(
        store,
        "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
        vec![("context_id", Value::Uuid(context_id.as_uuid()))],
    )
    .rows[0][0]
    {
        Value::Uuid(id) => *id,
        other => panic!("{other:?}"),
    }
}

fn stamp(store: &Store, context_id: ContextId, generation: uuid::Uuid) -> Option<serde_json::Value> {
    match &sql(
        store,
        "SELECT interpretation_verified_stamp FROM event_search_generations WHERE context_id = $context_id AND id = $id",
        vec![("context_id", Value::Uuid(context_id.as_uuid())), ("id", Value::Uuid(generation))],
    )
    .rows[0][0]
    {
        Value::Json(value) => Some(value.clone()),
        Value::Null => None,
        other => panic!("{other:?}"),
    }
}

fn interpretation_progress(store: &Store, context_id: ContextId, id: InterpretationGenerationId) -> (i64, i64) {
    let rows = sql(
        store,
        "SELECT capture_ordinal, build_ordinal FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
        vec![("context_id", Value::Uuid(context_id.as_uuid())), ("id", Value::Uuid(id.as_uuid()))],
    );
    match (&rows.rows[0][0], &rows.rows[0][1]) {
        (Value::Int64(capture), Value::Int64(build)) => (*capture, *build),
        other => panic!("{other:?}"),
    }
}

fn advance_interpretation(store: &Store, context_id: ContextId, id: InterpretationGenerationId) {
    sql(
        store,
        "UPDATE event_interpretation_generations SET build_ordinal = build_ordinal + 1 WHERE context_id = $context_id AND id = $id",
        vec![("context_id", Value::Uuid(context_id.as_uuid())), ("id", Value::Uuid(id.as_uuid()))],
    );
}

/// Re-verification, or the existing typed refusal. Nothing else is an acceptable answer to a
/// moved interpretation.
fn assert_verified_or_typed_refusal(result: &Result<context_graph::EventSearchBuildReceipt>, label: &str) -> bool {
    match result {
        Ok(_) => true,
        Err(CgError::Engine(message))
            if message.contains("not a completely sealed ready input")
                || message.contains("bound interpretation advanced before capture certification")
                || message.contains("changed while event-search membership was paged") =>
        {
            false
        }
        Err(other) => panic!("{label}: neither verification nor the typed refusal: {other:?}"),
    }
}

fn expected_stamp(id: InterpretationGenerationId, progress: (i64, i64)) -> serde_json::Value {
    json!({CONNECTOR: {
        "build_ordinal": progress.1,
        "capture_ordinal": progress.0,
        "generation_id": id.to_string(),
    }})
}

#[test]
fn a_stamp_that_moves_after_certification_makes_completion_capture_again() {
    let f = bound_fixture(3);
    let generation = {
        let store = open(&f.config);
        head_generation(&store, f.context_id)
    };
    {
        // New events give the run batches to commit, so the advance lands after its certified capture.
        let store = open(&f.config);
        store
            .append_event_batch(AppendEventBatch::live(
                f.context_id,
                declaration(),
                SOURCE,
                (4..=7).map(|n| draft(&format!("later-{n}"), n)).collect(),
            ))
            .expect("append events before the run");
    }
    let context_id = f.context_id;
    after_next_event_search_batch_commit_for_test(move |store| {
        advance_bound_interpretation_stamp_for_test(store, context_id, CONNECTOR)
            .expect("advance the bound interpretation without an event");
        // Everything captured before this point is the run's certified capture.
        take_event_search_membership_captures_for_test();
    });
    take_event_search_membership_captures_for_test();
    run(&f.config, f.context_id).expect("the run completes");
    let after_advance = take_event_search_membership_captures_for_test();
    let store = open(&f.config);
    assert_eq!(head_generation(&store, f.context_id), generation, "the same generation continues");
    let progress = interpretation_progress(&store, f.context_id, f.interpretation);
    assert!(
        after_advance >= 1,
        "completion must capture again once the bound interpretation stamp moved (captures after the advance: {after_advance})"
    );
    assert_eq!(
        stamp(&store, f.context_id, generation),
        Some(expected_stamp(f.interpretation, progress)),
        "the published stamp is the one completion verified"
    );
}
