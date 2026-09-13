//! RED proofs for incremental event-search edges that are not covered by the ordinary tail case.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, ContextId, CreateContext, DistanceMetric, Embedder, EmbedderRegistration,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchFilter,
    EventSearchPassage, EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "event-search-incremental-edge-red";
const SESSION: &str = "event-search-incremental-edge-session";
const SOURCE_A: &str = "incremental-edge-source-a";
const SOURCE_B: &str = "incremental-edge-source-b";
const SPACE: &str = "event-search-incremental-edge-768";

struct EdgeEmbedder;

impl EdgeEmbedder {
    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output(&self, inference: &EventSearchInference) -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "edge-fixture".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![inference.token_count() as f32; 768],
        }
    }
}

impl Embedder for EdgeEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search uses role-aware preparation")
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
            model_name: "edge-fixture".into(),
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

    fn encoding_profile_id(&self) -> String {
        "incremental-edge-byte-v1".into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(256)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let payload = text.bytes().map(u32::from).collect::<Vec<_>>();
        seal.document_builder()
            .token_count(payload.len())
            .byte_boundaries((0..text.len()).map(|byte| (byte, byte + 1)).collect())
            .payload(payload)
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        let payload = text.bytes().map(u32::from).collect::<Vec<_>>();
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference))
    }
}

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
        adapter_version: "incremental-edge-v1".into(),
        anchor_declaration_version: "incremental-edge-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(key: &str, ordinal: u64, role: EventRenderRole, text: &str) -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "fixture".into(),
        kind: "message".into(),
        render_role: role,
        interaction_ref: Some("edge-pair".into()),
        text: Some(text.into()),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn profile() -> EventSearchProfile {
    EventSearchProfile {
        // The ordinary incremental-tail RED fixture declares these binding renderer limits.
        target_passage_tokens: NonZeroU32::new(177).expect("nonzero target"),
        hard_passage_tokens: NonZeroU32::new(256).expect("nonzero hard limit"),
        ordinary_overlap_events: NonZeroU32::new(1).expect("nonzero overlap"),
        oversized_event_overlap_tokens: NonZeroU32::new(4).expect("nonzero oversized overlap"),
        ..EventSearchProfile::default()
    }
}

fn run(
    config: StoreConfig,
    context_id: ContextId,
    embedder: &Arc<EdgeEmbedder>,
) -> context_graph::EventSearchBuildReceipt {
    EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config,
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: profile(),
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero event batch"),
        max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease"),
    })
    .expect("public projection runner completes")
}

fn hits(store: &Store, context_id: ContextId) -> Vec<EventSearchPassage> {
    store
        .find_event_history(
            context_graph::ContextSelector::one(context_id),
            "incremental edge",
            EventSearchFilter {
                limit: NonZeroU32::new(16).expect("nonzero limit"),
                ..EventSearchFilter::default()
            },
        )
        .expect("public search read")
        .hits
        .into_iter()
        .map(|hit| {
            store
                .get_event_search_passage(
                    context_graph::ContextSelector::one(context_id),
                    hit.passage_id,
                )
                .expect("public show read")
                .expect("search hit has passage")
        })
        .collect()
}

fn memberships(
    store: &Store,
    context_id: ContextId,
    generation_id: context_graph::EventSearchGenerationId,
) -> Vec<Vec<Value>> {
    store.execute_sql_for_test(
        "SELECT ordinal, event_id, source_instance_ref FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal",
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
        ]),
    ).expect("inspect persisted source membership").rows
}

#[test]
fn append_on_one_of_two_sources_preserves_ready_membership_ordinals_and_projects_the_append() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(EdgeEmbedder);
    let config = config(
        root.path().join("incremental-sources.db"),
        embedder.embedding_space(),
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open Store");
    let context_id = store
        .create_context(CreateContext {
            name: "incremental edge sources".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    let a1 = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_A,
            vec![draft(
                "a1",
                1,
                EventRenderRole::HumanMessage,
                "A1 incremental edge",
            )],
        ))
        .expect("append A1")
        .events
        .remove(0)
        .event;
    let b1 = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_B,
            vec![draft(
                "b1",
                1,
                EventRenderRole::HumanMessage,
                "B1 incremental edge",
            )],
        ))
        .expect("append B1")
        .events
        .remove(0)
        .event;
    drop(store);

    let first = run(config.clone(), context_id, &embedder);
    let before =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("reopen ready projection");
    let old_memberships = memberships(&before, context_id, first.generation_id);
    assert_eq!(
        old_memberships.len(),
        2,
        "A1 and B1 are retained ready members"
    );
    assert!(
        old_memberships
            .iter()
            .any(|row| row[1] == Value::Uuid(a1.id.as_uuid()))
    );
    assert!(
        old_memberships
            .iter()
            .any(|row| row[1] == Value::Uuid(b1.id.as_uuid()))
    );
    let a2 = before
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_A,
            vec![draft(
                "a2",
                2,
                EventRenderRole::HumanMessage,
                "A2 incremental edge",
            )],
        ))
        .expect("append A2")
        .events
        .remove(0)
        .event;
    drop(before);

    let second = run(config.clone(), context_id, &embedder);
    assert_eq!(
        second.generation_id, first.generation_id,
        "an ordinary append extends the ready head"
    );
    let after = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen extended projection");
    let after_memberships = memberships(&after, context_id, second.generation_id);
    assert_eq!(
        &after_memberships[..2],
        old_memberships.as_slice(),
        "A1/B1 keep their persisted membership rows and ordinals"
    );
    assert!(
        after_memberships
            .iter()
            .any(|row| row[1] == Value::Uuid(a2.id.as_uuid())),
        "A2 joins the ready generation"
    );
    assert!(
        hits(&after, context_id)
            .iter()
            .any(|passage| passage.text.contains("A2 incremental edge")
                && passage.source_instance_ref == SOURCE_A),
        "public search projects A2 from source A"
    );
}

#[test]
fn excluded_metadata_does_not_leave_a_duplicate_current_tail_after_a_renderable_append() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(EdgeEmbedder);
    let config = config(
        root.path().join("incremental-metadata.db"),
        embedder.embedding_space(),
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open Store");
    let context_id = store
        .create_context(CreateContext {
            name: "incremental edge metadata".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_A,
            vec![
                draft(
                    "renderable-1",
                    1,
                    EventRenderRole::HumanMessage,
                    "renderable one incremental edge",
                ),
                EventDraft {
                    session_ref: SESSION.into(),
                    source_event_key: "metadata-2".into(),
                    source_position: json!({"ordinal": 2}),
                    occurred_at: None,
                    author: "fixture".into(),
                    kind: "message".into(),
                    render_role: EventRenderRole::Metadata,
                    interaction_ref: Some("edge-pair".into()),
                    text: None,
                    payload: json!({"structural": "metadata two stays captured"}),
                    anchors: json!({}),
                    source_remainder: json!({}),
                    coverage: EventCoverage::Complete,
                },
            ],
        ))
        .expect("append renderable and metadata");
    drop(store);

    let first = run(config.clone(), context_id, &embedder);
    assert_eq!(
        first.excluded_event_count, 1,
        "metadata is excluded from rendering"
    );
    let before =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("reopen ready projection");
    let old_tail = hits(&before, context_id)
        .into_iter()
        .next()
        .expect("renderable event made one current tail");
    before
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE_A,
            vec![draft(
                "renderable-3",
                3,
                EventRenderRole::HumanMessage,
                "renderable three incremental edge",
            )],
        ))
        .expect("append next renderable event");
    drop(before);

    let second = run(config.clone(), context_id, &embedder);
    assert_eq!(
        second.generation_id, first.generation_id,
        "the ready generation catches up in place"
    );
    let after = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen extended projection");
    let current = hits(&after, context_id);
    assert_eq!(
        current.len(),
        1,
        "public search has no duplicate current tail"
    );
    assert!(
        current[0]
            .text
            .contains("renderable three incremental edge"),
        "the replacement tail projects the append"
    );
    assert!(
        !current[0].text.contains("metadata two"),
        "metadata remains excluded from the rendered tail"
    );
    let shown_old_tail = after
        .get_event_search_passage(context_graph::ContextSelector::one(context_id), old_tail.id)
        .expect("public old-tail show")
        .expect("retired tail remains addressable");
    assert!(
        !shown_old_tail.is_current,
        "the old tail show result reports false for currentness"
    );
}
