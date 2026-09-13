//! RED proof that an event-search generation consumes only its sealed source membership.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::test_support::store_database_is_live_for_test;
use context_graph::{
    AppendEventBatch, CreateContext, DistanceMetric, Embedder, EmbedderRegistration,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchFilter,
    EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest, EvidenceKind,
    EvidenceProducer, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "event-search-sealed-source-red";
const SOURCE: &str = "event-search-sealed-source-fixture";
const SESSION: &str = "event-search-sealed-source-session";
const SEARCH_SPACE: &str = "event-search-sealed-source-768";
const SEALED_TEXT: &str = "sealed event remains the only source";
const LATE_TEXT: &str = "late append must not enter this generation";

fn config(path: PathBuf, space: EmbeddingSpace) -> StoreConfig {
    StoreConfig {
        db_path: path,
        extra_embedding_spaces: vec![space],
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "sealed-source-fixture-v1".into(),
        anchor_declaration_version: "sealed-source-fixture-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(ordinal: u64, text: &str) -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: format!("sealed-source-{ordinal}"),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "fixture".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(text.into()),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

type EmbedControl = Box<dyn FnOnce() + Send>;

struct SourceAppendingEmbedder {
    control: Mutex<Option<EmbedControl>>,
    embedded_documents: Mutex<Vec<String>>,
}

impl SourceAppendingEmbedder {
    fn new(control: impl FnOnce() + Send + 'static) -> Self {
        Self {
            control: Mutex::new(Some(Box::new(control))),
            embedded_documents: Mutex::new(Vec::new()),
        }
    }

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
                system: "event-search-sealed-source-red".into(),
                model_name: "fixture".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }
}

impl Embedder for SourceAppendingEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event-search uses role-aware preparation")
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
            model_name: "sealed-source".into(),
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
        "sealed-source-fixture-v1".into()
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

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.embedded_documents
            .lock()
            .expect("embedded document log")
            .push(
                inference
                    .token_ids()
                    .iter()
                    .map(|token| char::from_u32(*token).expect("fixture tokens are bytes"))
                    .collect(),
            );
        if let Some(control) = self.control.lock().expect("embed control lock").take() {
            control();
        }
        Ok(Self::output())
    }

    fn embed_prepared_query_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }
}

#[test]
fn late_append_stays_out_of_the_already_embedded_passage_then_catches_up_before_activation() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("sealed-sources.db");
    let control_path = path.clone();
    let embedder = Arc::new(SourceAppendingEmbedder::new(move || {
        assert!(
            !store_database_is_live_for_test(&control_path),
            "the standalone runner must release its Store before the embedder callback"
        );
        let control = Store::open(StoreConfig {
            db_path: control_path,
            default_text_embedder: None,
            ..StoreConfig::default()
        })
        .expect("open released Store from embedder callback");
        let context_id = control
            .list_contexts()
            .expect("list the one test Context")
            .into_iter()
            .next()
            .expect("test Context exists")
            .id;
        control
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                SOURCE,
                vec![draft(2, LATE_TEXT)],
            ))
            .expect("append the event after source capture");
    }));
    let config = config(path.clone(), embedder.embedding_space());
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open capture Store");
    let context_id = store
        .create_context(CreateContext {
            name: "sealed source generation".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    let sealed_event = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![draft(1, SEALED_TEXT)],
        ))
        .expect("append event present at source capture")
        .events
        .into_iter()
        .next()
        .expect("one sealed event")
        .event;
    drop(store);

    let receipt = EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config.clone(),
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: EventSearchProfile::default(),
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch size"),
        max_passages_per_batch: NonZeroU32::new(8).expect("nonzero batch size"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte size"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease duration"),
    })
    .expect("the sealed generation must complete after the late append");

    let embedded_documents = embedder
        .embedded_documents
        .lock()
        .expect("embedded document log")
        .clone();
    assert!(
        embedded_documents
            .first()
            .expect("the original sealed passage was embedded")
            .contains(SEALED_TEXT),
        "the original embedded passage contains its sealed event"
    );
    assert!(
        !embedded_documents
            .first()
            .expect("the original sealed passage was embedded")
            .contains(LATE_TEXT),
        "the late append cannot enter the passage that was already embedded"
    );
    assert_eq!(receipt.source_watermark.event_count, 2);
    assert_eq!(receipt.eligible_event_count, 2);
    assert_eq!(receipt.rendered_event_count, 2);
    assert_eq!(receipt.embedded_event_count, 2);

    let reopened = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("open public read Store");
    let found = reopened
        .find_event_history(
            context_graph::ContextSelector::one(context_id),
            "sealed",
            EventSearchFilter::default(),
        )
        .expect("public search read");
    assert_eq!(
        found.hits.len(),
        1,
        "catch-up replaces the original passage"
    );
    assert!(found.hits[0].text.contains(SEALED_TEXT));
    assert!(found.hits[0].text.contains(LATE_TEXT));
    assert_eq!(
        found.hits[0].sources.len(),
        2,
        "the caught-up passage retains a link for each sealed source event"
    );
    assert_eq!(found.hits[0].sources[0].event_id, sealed_event.id);
    assert_ne!(
        found.hits[0].sources[1].event_id, sealed_event.id,
        "the second link belongs to the late sealed member"
    );

    let members = reopened
        .execute_sql_for_test(
            "SELECT event_id FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(receipt.generation_id.as_uuid())),
            ]),
        )
        .expect("inspect sealed source membership");
    assert_eq!(
        members.rows,
        vec![
            vec![Value::Uuid(sealed_event.id.as_uuid())],
            vec![Value::Uuid(found.hits[0].sources[1].event_id.as_uuid())],
        ]
    );
}
