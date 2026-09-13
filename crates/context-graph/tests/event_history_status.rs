//! Event-history status: one read-only fold, answered identically by the writable Store and the
//! read-only consumer reader.
//!
//! Every expected count below comes from what the fixture captured through the public append and
//! projection surfaces, never from a status helper.

use std::collections::BTreeMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CgError, ConsumerReader, ContextBindingKey, ContextId, ContextSelector,
    CreateContext, DistanceMetric, Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput,
    EmbeddingSpace, EmbeddingSpaceBinding, EventAdapterDeclaration, EventCoverage,
    EventDeliveryState, EventDraft, EventHistorySearchState, EventHistoryStatusSet,
    EventPlacementState, EventRenderRole, EventSearchBuildRequest, EventSearchProfile,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig, event_history_reader_declaration,
};
use serde_json::json;

const SPACE: &str = "event-history-status-768";
const CONNECTOR: &str = "event-history-status-connector";

struct FixedEmbedder;

impl FixedEmbedder {
    fn registration() -> EmbedderRegistration {
        let embedder = Arc::new(FixedEmbedder);
        EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }
    }

    fn output() -> EmbeddingOutput {
        let mut vector = vec![0.0; 768];
        vector[0] = 1.0;
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "event-history-status".into(),
                model_name: "fixed".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }
}

impl Embedder for FixedEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event-history projection uses role-aware preparation")
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
            model_name: "fixed".into(),
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
        "event-history-status-profile".into()
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
        let boundaries = (0..text.len()).map(|index| (index, index + 1)).collect();
        seal.document_builder()
            .token_count(payload.len())
            .byte_boundaries(boundaries)
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
        _inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }

    fn embed_prepared_query_hook(
        &self,
        _inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }
}

fn open(path: &Path) -> Store {
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: path.to_path_buf(),
            extra_embedding_spaces: vec![FixedEmbedder.embedding_space()],
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        vec![FixedEmbedder::registration()],
    )
    .expect("open store")
}

fn context(store: &Store, name: &str) -> ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "status-fixture-v1".into(),
        anchor_declaration_version: "status-fixture-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn append(
    store: &Store,
    context_id: ContextId,
    session: &str,
    source: &str,
    key: &str,
    ordinal: u64,
    coverage: EventCoverage,
) {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            source,
            vec![EventDraft {
                session_ref: session.into(),
                source_event_key: key.into(),
                source_position: json!({"ordinal": ordinal}),
                occurred_at: None,
                author: "fixture-user".into(),
                kind: "message".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some(format!("status fixture text {key}")),
                payload: json!({}),
                anchors: json!({}),
                source_remainder: json!({}),
                coverage,
            }],
        ))
        .expect("capture event");
}

fn project(store: &Store, context_id: ContextId) {
    store
        .catch_up_event_search(EventSearchBuildRequest {
            context_id,
            embedder: FixedEmbedder::registration(),
            profile: EventSearchProfile::default(),
            max_events_per_batch: NonZeroU32::new(8).unwrap(),
            max_passages_per_batch: NonZeroU32::new(8).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(64 * 1024).unwrap(),
            lease_duration_ms: NonZeroU64::new(5_000).unwrap(),
        })
        .expect("projection completes");
}

fn by_context(
    set: &EventHistoryStatusSet,
) -> BTreeMap<ContextId, &context_graph::EventHistoryContextStatus> {
    set.contexts
        .iter()
        .map(|status| (status.context_id, status))
        .collect()
}

fn directory_image(path: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut pending = vec![path.to_path_buf()];
    let mut image = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).expect("read directory") {
            let entry = entry.expect("directory entry");
            let entry_path = entry.path();
            if entry.file_type().expect("entry type").is_dir() {
                pending.push(entry_path);
            } else {
                let bytes = std::fs::read(&entry_path).expect("read file");
                image.push((entry_path, bytes));
            }
        }
    }
    image.sort_by(|left, right| left.0.cmp(&right.0));
    image
}

#[test]
fn status_separates_capture_from_search_coverage_per_selected_context() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("status.db");
    let store = open(&database);
    let alpha = context(&store, "status-alpha");
    let beta = context(&store, "status-beta");
    let empty = context(&store, "status-empty");

    append(
        &store,
        alpha,
        "session-a1",
        "source-a1",
        "a1-1",
        1,
        EventCoverage::Complete,
    );
    append(
        &store,
        alpha,
        "session-a1",
        "source-a1",
        "a1-2",
        2,
        EventCoverage::Opaque,
    );
    append(
        &store,
        alpha,
        "session-a2",
        "source-a2",
        "a2-1",
        1,
        EventCoverage::PayloadRefused,
    );
    append(
        &store,
        beta,
        "session-b1",
        "source-b1",
        "b1-1",
        1,
        EventCoverage::Complete,
    );

    let selector = ContextSelector::new([alpha, beta, empty]).expect("three contexts");
    let status = store
        .event_history_status(selector.clone())
        .expect("status reads captured history");
    assert_eq!(status.placement, EventPlacementState::LocalOnly);
    assert_eq!(
        status.contexts.len(),
        3,
        "one tagged status per selected Context"
    );
    let contexts = by_context(&status);

    let alpha_status = contexts[&alpha];
    assert_eq!(alpha_status.context_name, "status-alpha");
    assert_eq!(alpha_status.captured_events, 3);
    assert_eq!(alpha_status.sessions, 2);
    assert_eq!(alpha_status.source_streams.len(), 2);
    let first_stream = alpha_status
        .source_streams
        .iter()
        .find(|stream| stream.source_instance_ref == "source-a1")
        .expect("source-a1 is reported");
    assert_eq!(first_stream.session_ref, "session-a1");
    assert_eq!(first_stream.connector_id, CONNECTOR);
    assert_eq!(first_stream.adapter_version, "status-fixture-v1");
    assert_eq!(first_stream.event_count, 2);
    assert_eq!(first_stream.latest_event_seq, Some(2));
    assert_eq!(alpha_status.coverage.complete, 1);
    assert_eq!(alpha_status.coverage.opaque, 1);
    assert_eq!(alpha_status.coverage.payload_refused, 1);
    assert_eq!(alpha_status.coverage.source_declared_missing, 0);
    assert_eq!(alpha_status.memory.goals, 0);
    assert_eq!(alpha_status.search.state, EventHistorySearchState::Pending);
    assert_eq!(alpha_status.search.covered_events, 0);
    assert_eq!(alpha_status.search.pending_events, 3);
    assert_eq!(alpha_status.search.active_generation_id, None);
    let delivery = alpha_status
        .delivery
        .as_ref()
        .expect("a local-only Store reports its delivery state");
    assert_eq!(delivery.state, EventDeliveryState::Disabled);
    assert_eq!(delivery.eligible, 0);

    assert_eq!(contexts[&beta].captured_events, 1);
    assert_eq!(contexts[&beta].search.pending_events, 1);

    let empty_status = contexts[&empty];
    assert_eq!(empty_status.captured_events, 0);
    assert_eq!(empty_status.sessions, 0);
    assert!(empty_status.source_streams.is_empty());
    assert_eq!(
        empty_status.search.state,
        EventHistorySearchState::Ready,
        "a Context with nothing captured has no uncovered work"
    );

    project(&store, alpha);
    let projected = store
        .event_history_status(selector.clone())
        .expect("status after projection");
    let projected_alpha = by_context(&projected)[&alpha];
    assert_eq!(projected_alpha.search.state, EventHistorySearchState::Ready);
    assert_eq!(projected_alpha.search.covered_events, 3);
    assert_eq!(projected_alpha.search.pending_events, 0);
    assert!(projected_alpha.search.active_generation_id.is_some());
    assert_eq!(
        projected_alpha.search.embedding_space_id.as_deref(),
        Some(SPACE)
    );
    assert_eq!(
        by_context(&projected)[&beta].search.state,
        EventHistorySearchState::Pending,
        "projecting one Context never reports another as searchable"
    );

    append(
        &store,
        alpha,
        "session-a2",
        "source-a2",
        "a2-2",
        2,
        EventCoverage::Complete,
    );
    let stale = store
        .event_history_status(ContextSelector::one(alpha))
        .expect("status after a later capture");
    let stale_alpha = &stale.contexts[0];
    assert_eq!(stale_alpha.captured_events, 4);
    assert_eq!(stale_alpha.search.covered_events, 3);
    assert_eq!(stale_alpha.search.pending_events, 1);
    assert_eq!(
        stale_alpha.search.state,
        EventHistorySearchState::Pending,
        "newer captured events without a live projection lease are pending, never running"
    );

    let blocked = store
        .event_history_status(selector.clone())
        .expect("status")
        .with_search_artifact_available(false);
    let blocked_contexts = by_context(&blocked);
    assert_eq!(
        blocked_contexts[&alpha].search.state,
        EventHistorySearchState::Blocked
    );
    assert_eq!(
        blocked_contexts[&beta].search.state,
        EventHistorySearchState::Blocked
    );
    assert_eq!(
        blocked_contexts[&empty].search.state,
        EventHistorySearchState::Ready,
        "caught-up coverage stays ready without a model"
    );
    let available = store
        .event_history_status(selector.clone())
        .expect("status")
        .with_search_artifact_available(true);
    assert_eq!(
        by_context(&available)[&alpha].search.state,
        EventHistorySearchState::Pending
    );

    let expected = store
        .event_history_status(selector.clone())
        .expect("Store answer");
    drop(store);

    let before = directory_image(root.path());
    let reader = ConsumerReader::open(&database, event_history_reader_declaration())
        .expect("the published read declaration opens the existing Store");
    let actual = reader
        .event_history_status(selector)
        .expect("the read-only reader answers the same fold");
    drop(reader);
    assert_eq!(actual, expected, "one fold answers both doors identically");
    assert_eq!(
        directory_image(root.path()),
        before,
        "the read-only status leaves every database and companion byte unchanged"
    );
}

#[test]
fn status_refuses_an_unknown_context_instead_of_reporting_zero_counts() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("unknown.db");
    let store = open(&database);
    let known = context(&store, "known");
    let unknown = ContextId::new_v7();
    let error = store
        .event_history_status(ContextSelector::new([known, unknown]).expect("two ids"))
        .expect_err("an unknown selected Context is refused");
    assert!(
        matches!(error, CgError::NotFound { .. }),
        "unknown Context is a typed refusal: {error:?}"
    );
    drop(store);

    let reader =
        ConsumerReader::open(&database, event_history_reader_declaration()).expect("open reader");
    let error = reader
        .event_history_status(ContextSelector::one(unknown))
        .expect_err("the reader refuses the same unknown Context");
    assert!(matches!(error, CgError::NotFound { .. }), "{error:?}");
}

#[test]
fn read_only_reader_resolves_a_binding_without_writing() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("binding.db");
    let store = open(&database);
    let key = ContextBindingKey {
        namespace: "status-test-workspace/v1".into(),
        opaque_reference: "opaque-project-reference".into(),
    };
    let bound = store
        .create_context_with_binding(
            CreateContext {
                name: "bound-project".into(),
                labels: Vec::new(),
                properties: Default::default(),
            },
            key.clone(),
        )
        .expect("create and bind")
        .context_id;
    drop(store);

    let before = directory_image(root.path());
    let reader =
        ConsumerReader::open(&database, event_history_reader_declaration()).expect("open reader");
    assert_eq!(
        reader
            .resolve_context_binding(&key)
            .expect("resolve binding"),
        Some(bound)
    );
    assert_eq!(
        reader
            .resolve_context_binding(&ContextBindingKey {
                namespace: key.namespace.clone(),
                opaque_reference: "another-reference".into(),
            })
            .expect("resolve missing binding"),
        None,
        "an unbound reference is an answered absence"
    );
    drop(reader);
    assert_eq!(directory_image(root.path()), before);
}

fn released() -> context_graph::ReleasedInterpretationDeclaration {
    context_graph::ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: "status-released-v1".into(),
        target_adapter_version: "status-fixture-v1".into(),
        target_anchor_declaration_version: "status-fixture-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "status-fixture-v1".into(),
            anchor_declaration_version: "status-fixture-v1".into(),
        }],
        mapping_contract_id: "status-anchor-contract".into(),
        mapping_contract_sha256: "status-anchor-contract-v1".into(),
    }
}

/// Keeps every captured field and replaces the anchor set with `file = src/new.rs`.
struct RenamingMapper(context_graph::ReleasedInterpretationDeclaration);

impl context_graph::ReleasedConnectorInterpretation for RenamingMapper {
    fn declaration(&self) -> &context_graph::ReleasedInterpretationDeclaration {
        &self.0
    }

    fn interpret_event(
        &self,
        captured: &context_graph::ConversationEvent,
    ) -> Result<context_graph::EffectiveEventFields> {
        Ok(context_graph::EffectiveEventFields {
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

struct PreservingContract(context_graph::ReleasedInterpretationDeclaration);

impl context_graph::InstalledInterpretationContract for PreservingContract {
    fn declaration(&self) -> &context_graph::ReleasedInterpretationDeclaration {
        &self.0
    }

    fn validate(
        &self,
        captured: &context_graph::CapturedEventAndAnchors,
        candidate: &context_graph::EffectiveEventFields,
    ) -> Result<()> {
        if candidate.text == captured.event.text && candidate.author == captured.event.author {
            Ok(())
        } else {
            Err(CgError::Engine("fixture refused a changed capture".into()))
        }
    }
}

#[test]
fn the_published_read_declaration_serves_the_exact_anchor_event_read() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("anchors.db");
    let config = StoreConfig {
        db_path: database.clone(),
        extra_embedding_spaces: vec![FixedEmbedder.embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store = open(&database);
    let context_id = context(&store, "anchored");
    let captured = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "anchored-source",
            vec![EventDraft {
                session_ref: "anchored-session".into(),
                source_event_key: "anchored-1".into(),
                source_position: json!({"ordinal": 1}),
                occurred_at: None,
                author: "fixture-user".into(),
                kind: "message".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some("anchored fixture text".into()),
                payload: json!({}),
                anchors: json!({"file": "src/old.rs"}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }],
        ))
        .expect("capture anchored event")
        .events
        .remove(0)
        .event;
    drop(store);

    // Activate a released interpretation that replaces the anchor, so the exact read follows
    // the active head through the interpretation tables.
    let mapper: Arc<dyn context_graph::ReleasedConnectorInterpretation> =
        Arc::new(RenamingMapper(released()));
    let contract: Arc<dyn context_graph::InstalledInterpretationContract> =
        Arc::new(PreservingContract(released()));
    let interpretation = context_graph::ConnectorInterpretationRunner::build(
        context_graph::InterpretationBuildRequest {
            store: config.clone(),
            context_id,
            connector_id: CONNECTOR.into(),
            target_interpretation_version: "status-released-v1".into(),
            mapper: mapper.clone(),
            contract: contract.clone(),
            max_events_per_batch: NonZeroU32::new(8).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(4_096).unwrap(),
            lease_duration_ms: NonZeroU64::new(10_000).unwrap(),
        },
    )
    .expect("build the renaming interpretation");
    let store = open(&database);
    let search = store
        .rebuild_event_search(context_graph::EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: Some(interpretation.generation_id),
            embedder: FixedEmbedder::registration(),
            profile: EventSearchProfile::default(),
            max_events_per_batch: NonZeroU32::new(8).unwrap(),
            max_passages_per_batch: NonZeroU32::new(8).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(4_096).unwrap(),
            lease_duration_ms: NonZeroU64::new(10_000).unwrap(),
        })
        .expect("build the staged replacement search");
    drop(store);
    context_graph::ConnectorInterpretationRunner::activate(
        context_graph::InterpretationActivationRequest {
            store: config.clone(),
            context_id,
            connector_id: CONNECTOR.into(),
            target_interpretation_version: "status-released-v1".into(),
            mapper,
            contract,
            interpretation_generation_id: interpretation.generation_id,
            replacement_search_generation_id: search.generation_id,
        },
    )
    .expect("activate the renaming interpretation");

    let range = context_graph::EventRange::Count {
        after_seq: None,
        limit: NonZeroU32::new(10).unwrap(),
    };
    let store = open(&database);
    let expected = store
        .list_events_by_anchor(
            context_id,
            context_graph::AnchorKey("file".into()),
            context_graph::AnchorValue("\"src/new.rs\"".into()),
            range.clone(),
        )
        .expect("Store anchor read");
    assert_eq!(
        expected,
        vec![captured],
        "the active interpretation's anchor answers the exact read"
    );
    drop(store);

    let reader =
        ConsumerReader::open(&database, event_history_reader_declaration()).expect("open reader");
    let actual = reader
        .graph()
        .list_events_by_anchor(
            context_id,
            context_graph::AnchorKey("file".into()),
            context_graph::AnchorValue("\"src/new.rs\"".into()),
            range,
        )
        .expect("the published declaration covers every table the anchor read needs");
    assert_eq!(actual, expected);
}
