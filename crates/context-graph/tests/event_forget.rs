//! Forgetting imported history is one authoritative erasure per selection.
//!
//! The fixtures capture events through the public append surface, build the public search
//! projection, and then read the Store back through the published read-only reader. Expected counts
//! come from the capture receipts, never from the forget implementation.

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process;
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CgError, ConsumerReader, ContextBindingKey, ContextId, ContextSelector,
    CreateContext, DistanceMetric, Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput,
    EmbeddingSpace, EmbeddingSpaceBinding, EventAdapterDeclaration, EventCoverage, EventDraft,
    EventHistorySearchState, EventPlacementPolicy, EventRenderRole, EventSearchBuildRequest,
    EventSearchFilter, EventSearchProfile, EvidenceKind, EvidenceProducer, ForgetSelection, Result,
    Store, StoreConfig, event_history_reader_declaration,
};
use contextdb_core::Value;
use serde_json::json;

const SPACE: &str = "event-forget-768";
const CONNECTOR: &str = "event-forget-connector";

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
                system: "event-forget".into(),
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
        "event-forget-profile".into()
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

fn config(path: &Path, placement: Option<EventPlacementPolicy>) -> StoreConfig {
    StoreConfig {
        db_path: path.to_path_buf(),
        extra_embedding_spaces: vec![FixedEmbedder.embedding_space()],
        default_text_embedder: None,
        event_placement: placement,
        ..StoreConfig::default()
    }
}

fn open(path: &Path) -> Store {
    Store::open_with_embedder_registrations(config(path, None), vec![FixedEmbedder::registration()])
        .expect("open store")
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "forget-fixture-v1".into(),
        anchor_declaration_version: "forget-fixture-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

/// Capture one event and return its captured anchor-row count.
fn append(
    store: &Store,
    context_id: ContextId,
    session: &str,
    source: &str,
    key: &str,
    ordinal: u64,
) -> u64 {
    let receipt = store
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
                text: Some(format!("forget fixture text {key}")),
                payload: json!({"detail": key}),
                anchors: json!({"file": format!("/workspace/{key}.md")}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }],
        ))
        .expect("capture event");
    receipt
        .events
        .iter()
        .map(|result| result.event.anchor_count)
        .sum()
}

fn project(store: &Store, context_id: ContextId) {
    store
        .catch_up_event_search(EventSearchBuildRequest {
            context_id,
            embedder: FixedEmbedder::registration(),
            profile: EventSearchProfile::default(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).unwrap(),
            max_passages_per_batch: NonZeroU32::new(8).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(64 * 1024).unwrap(),
            lease_duration_ms: NonZeroU64::new(5_000).unwrap(),
        })
        .expect("projection completes");
}

fn count(reader: &ConsumerReader, sql: &str, context_id: ContextId) -> u64 {
    let answer = reader
        .execute(
            sql,
            &HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("count rows");
    match answer.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => u64::try_from(*value).expect("non-negative count"),
        other => panic!("count query returned {other:?}"),
    }
}

/// One context's pending-work markers as (generation, connector, source, session, pending),
/// sorted, so a before/after pair compares as a relational control.
fn pending_streams(
    reader: &ConsumerReader,
    context_id: ContextId,
) -> Vec<(String, String, String, String, bool)> {
    let answer = reader
        .execute(
            "SELECT generation_id, connector_id, source_instance_ref, session_ref, pending FROM event_search_pending_streams WHERE context_id = $context_id",
            &HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("read pending-stream markers");
    let mut rows = answer
        .rows
        .iter()
        .map(|row| match row.as_slice() {
            [
                Value::Uuid(generation),
                Value::Text(connector),
                Value::Text(source),
                Value::Text(session),
                Value::Bool(pending),
            ] => (
                generation.to_string(),
                connector.clone(),
                source.clone(),
                session.clone(),
                *pending,
            ),
            other => panic!("pending-stream marker row {other:?}"),
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

fn reader(path: &Path) -> ConsumerReader {
    ConsumerReader::open(path, event_history_reader_declaration()).expect("open reader")
}

#[test]
fn forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-session.db");
    let store = open(&database);
    let key = ContextBindingKey {
        namespace: "forget-test-workspace/v1".into(),
        opaque_reference: "forget-project".into(),
    };
    let context_id = store
        .create_context_with_binding(
            CreateContext {
                name: "forget-project".into(),
                labels: Vec::new(),
                properties: Default::default(),
            },
            key.clone(),
        )
        .expect("create bound context")
        .context_id;

    let forgotten_anchors = append(
        &store,
        context_id,
        "session-gone",
        "source-gone",
        "gone-1",
        1,
    ) + append(
        &store,
        context_id,
        "session-gone",
        "source-gone",
        "gone-2",
        2,
    );
    let kept_anchors = append(
        &store,
        context_id,
        "session-kept",
        "source-kept",
        "kept-1",
        1,
    );
    assert!(forgotten_anchors > 0, "the fixture captured exact anchors");
    project(&store, context_id);

    let before_find = store
        .find_event_history(
            ContextSelector::one(context_id),
            "forget fixture",
            EventSearchFilter::default(),
        )
        .expect("history is searchable before forget");
    let forgotten_passages = before_find
        .hits
        .iter()
        .filter(|hit| hit.session_ref == "session-gone")
        .map(|hit| hit.passage_id)
        .collect::<Vec<_>>();
    assert!(
        !forgotten_passages.is_empty(),
        "the forgotten session was searchable"
    );
    // Projection clears every marker, so one more capture to the session being forgotten leaves
    // it pending work: the erasure is then proven on a live marker, not a cleared one.
    let pending_anchors = append(
        &store,
        context_id,
        "session-gone",
        "source-gone",
        "gone-3",
        3,
    );
    let markers_before = pending_streams(&reader(&database), context_id);
    assert!(
        markers_before
            .iter()
            .any(|(_, _, _, session, pending)| session == "session-gone" && *pending),
        "the fixture leaves a pending marker for the session to forget: {markers_before:?}"
    );
    assert!(
        markers_before
            .iter()
            .any(|(_, _, _, session, _)| session == "session-kept"),
        "the fixture leaves a marker for the retained session: {markers_before:?}"
    );

    let receipt = store
        .forget_event_history(ForgetSelection::Session {
            context_id,
            connector_id: CONNECTOR.into(),
            session_ref: "session-gone".into(),
        })
        .expect("forget one session");
    assert!(receipt.changed);
    assert_eq!(receipt.sessions, 1);
    assert_eq!(receipt.events, 3);
    assert_eq!(receipt.anchors, forgotten_anchors + pending_anchors);
    assert!(receipt.passages > 0, "search passages were erased");
    assert_eq!(
        receipt.vectors, receipt.passages,
        "every erased passage carried one embedding handle"
    );

    for passage_id in &forgotten_passages {
        assert_eq!(
            store
                .get_event_search_passage(ContextSelector::one(context_id), *passage_id)
                .expect("read old handle"),
            None,
            "an old handle of forgotten history no longer resolves"
        );
    }
    let status = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("status after forget");
    assert_eq!(status.contexts[0].captured_events, 1);
    assert_eq!(status.contexts[0].sessions, 1);
    assert_eq!(
        status.contexts[0].source_streams.len(),
        1,
        "the forgotten stream's checkpoint is gone"
    );
    assert_eq!(
        status.contexts[0].source_streams[0].event_count, 1,
        "the surviving capture stream summary remains exact"
    );
    assert_eq!(
        status.contexts[0].source_streams[0].latest_event_seq,
        Some(1)
    );
    assert_eq!(
        status.contexts[0].search.state,
        EventHistorySearchState::Pending,
        "the remaining history reads as pending re-indexing, never as an error or as ready"
    );
    assert_eq!(status.contexts[0].search.pending_events, 1);
    assert_eq!(
        store
            .resolve_context_binding(&key)
            .expect("binding survives"),
        Some(context_id),
        "the project binding remains"
    );

    let derived_reader = reader(&database);
    let markers_after = pending_streams(&derived_reader, context_id);
    assert!(
        markers_after
            .iter()
            .all(|(_, _, _, session, _)| session != "session-gone"),
        "no pending-work marker of the forgotten session survives, in any generation: {markers_after:?}"
    );
    assert_eq!(
        markers_after,
        markers_before
            .iter()
            .filter(|(_, _, _, session, _)| session == "session-kept")
            .cloned()
            .collect::<Vec<_>>(),
        "the retained session's markers are exactly what they were before the forget"
    );
    assert_eq!(
        count(
            &derived_reader,
            "SELECT COUNT(*) AS n FROM event_context_sessions WHERE context_id = $context_id AND event_count = 1",
            context_id
        ),
        1,
        "the capture-time session summary is deleted with the forgotten session and retains the kept session's exact count"
    );
    assert_eq!(
        count(
            &derived_reader,
            "SELECT COUNT(*) AS n FROM event_search_session_progress WHERE context_id = $context_id",
            context_id
        ),
        0,
        "forget purges the generation progress in the same transaction before the retained history is rebuilt"
    );
    drop(derived_reader);

    project(&store, context_id);
    let after_find = store
        .find_event_history(
            ContextSelector::one(context_id),
            "forget fixture",
            EventSearchFilter::default(),
        )
        .expect("the remaining history is searchable again");
    assert!(!after_find.hits.is_empty());
    assert!(
        after_find
            .hits
            .iter()
            .all(|hit| hit.session_ref == "session-kept"),
        "find returns nothing from the forgotten session"
    );
    drop(store);

    let reader = reader(&database);
    assert_eq!(
        count(
            &reader,
            "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
            context_id
        ),
        1
    );
    assert_eq!(
        count(
            &reader,
            "SELECT COUNT(*) AS n FROM event_anchors WHERE context_id = $context_id",
            context_id
        ),
        kept_anchors
    );
    assert_eq!(
        count(
            &reader,
            "SELECT COUNT(*) AS n FROM event_source_checkpoints WHERE context_id = $context_id",
            context_id
        ),
        0,
        "live appends carry no file checkpoint"
    );
    assert_eq!(
        count(
            &reader,
            "SELECT COUNT(*) AS n FROM contexts WHERE id = $context_id",
            context_id
        ),
        1,
        "the Context row remains"
    );
}

#[test]
fn forgetting_a_context_erases_all_of_its_history_and_the_next_capture_lands_in_the_same_context() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-context.db");
    let store = open(&database);
    let forgotten = store
        .create_context(CreateContext {
            name: "forgotten".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    let untouched = store
        .create_context(CreateContext {
            name: "untouched".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append(&store, forgotten, "s1", "src1", "f-1", 1);
    append(&store, forgotten, "s2", "src2", "f-2", 1);
    append(&store, untouched, "s1", "src1", "u-1", 1);
    project(&store, forgotten);
    project(&store, untouched);

    let selection_reader = reader(&database);
    assert_eq!(
        selection_reader
            .forget_selection_session_count(&ForgetSelection::Context {
                context_id: forgotten,
            })
            .expect("count selected Context sessions"),
        2,
        "events, checkpoints, and locators for the same sessions count once"
    );
    assert_eq!(
        selection_reader
            .forget_selection_session_count(&ForgetSelection::Session {
                context_id: forgotten,
                connector_id: CONNECTOR.into(),
                session_ref: "s1".into(),
            })
            .expect("count selected session"),
        1
    );
    drop(selection_reader);

    let receipt = store
        .forget_event_history(ForgetSelection::Context {
            context_id: forgotten,
        })
        .expect("forget a Context's history");
    assert!(receipt.changed);
    assert_eq!(receipt.sessions, 2);
    assert_eq!(receipt.events, 2);
    assert_eq!(receipt.anchors, 2);
    assert!(receipt.passages > 0);
    assert_eq!(receipt.vectors, receipt.passages);

    let status = store
        .event_history_status(ContextSelector::new([forgotten, untouched]).expect("two"))
        .expect("status after forget");
    let forgotten_status = status
        .contexts
        .iter()
        .find(|status| status.context_id == forgotten)
        .expect("the forgotten Context is still listed");
    assert_eq!(forgotten_status.captured_events, 0);
    assert_eq!(forgotten_status.search.covered_events, 0);
    assert_eq!(forgotten_status.search.active_generation_id, None);
    let untouched_status = status
        .contexts
        .iter()
        .find(|status| status.context_id == untouched)
        .expect("the other Context");
    assert_eq!(untouched_status.captured_events, 1);
    assert_eq!(
        untouched_status.search.state,
        EventHistorySearchState::Ready
    );

    let again = store
        .forget_event_history(ForgetSelection::Context {
            context_id: forgotten,
        })
        .expect("forgetting nothing succeeds");
    assert!(!again.changed, "nothing left to forget changes nothing");
    assert_eq!(again.events, 0);

    append(&store, forgotten, "s1", "src1", "f-1", 1);
    project(&store, forgotten);
    let status = store
        .event_history_status(ContextSelector::one(forgotten))
        .expect("status after recapture");
    assert_eq!(status.contexts[0].captured_events, 1);
    assert_eq!(
        status.contexts[0].search.state,
        EventHistorySearchState::Ready
    );
}

#[test]
fn forgetting_an_unknown_session_changes_nothing() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-nothing.db");
    let store = open(&database);
    let context_id = store
        .create_context(CreateContext {
            name: "nothing".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append(&store, context_id, "present", "src", "p-1", 1);
    let receipt = store
        .forget_event_history(ForgetSelection::Session {
            context_id,
            connector_id: CONNECTOR.into(),
            session_ref: "absent".into(),
        })
        .expect("an unknown session is nothing to forget");
    assert!(!receipt.changed);
    assert_eq!(receipt.sessions, 0);
    assert_eq!(receipt.events, 0);
    let status = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("status");
    assert_eq!(status.contexts[0].captured_events, 1);
}

#[test]
fn a_tenant_hub_store_is_refused_rather_than_purged_from_an_edge() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-tenant.db");
    let store = Store::open_with_embedder_registrations(
        config(&database, Some(EventPlacementPolicy::TenantHub)),
        vec![FixedEmbedder::registration()],
    )
    .expect("open unbound tenant store");
    let error = store
        .forget_event_history(ForgetSelection::Context {
            context_id: ContextId::new_v7(),
        })
        .expect_err("an edge never issues an authoritative purge");
    assert!(
        matches!(error, CgError::EventHistoryForgetRequiresEdgeDiscard { .. }),
        "tenant-hub forget is a typed refusal: {error:?}"
    );
}

#[test]
fn a_context_can_be_forgotten_after_a_session_forget_and_rebuild() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-twice.db");
    let store = open(&database);
    let context_id = store
        .create_context(CreateContext {
            name: "twice".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append(&store, context_id, "s1", "src1", "t-1", 1);
    append(&store, context_id, "s2", "src2", "t-2", 1);
    project(&store, context_id);
    store
        .forget_event_history(ForgetSelection::Session {
            context_id,
            connector_id: CONNECTOR.into(),
            session_ref: "s1".into(),
        })
        .expect("forget one session");
    project(&store, context_id);
    let markers_before = pending_streams(&reader(&database), context_id);
    assert!(
        !markers_before.is_empty(),
        "the rebuilt Context carries pending-work markers before its forget"
    );
    let receipt = store
        .forget_event_history(ForgetSelection::Context { context_id })
        .expect("forget the Context after its search index was rebuilt");
    assert_eq!(receipt.events, 1);
    assert_eq!(
        pending_streams(&reader(&database), context_id),
        Vec::new(),
        "forgetting the Context erases every one of its pending-work markers"
    );
}

#[test]
fn history_captured_again_after_a_forget_can_be_forgotten_again() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-recapture.db");
    let store = open(&database);
    let context_id = store
        .create_context(CreateContext {
            name: "recapture".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append(&store, context_id, "s1", "src1", "r-1", 1);
    store
        .forget_event_history(ForgetSelection::Context { context_id })
        .expect("first forget");
    append(&store, context_id, "s1", "src1", "r-1", 1);
    let receipt = store
        .forget_event_history(ForgetSelection::Context { context_id })
        .expect("the same history captured again is forgotten again");
    assert_eq!(receipt.events, 1);
}

const INTERPRETATION: &str = "forget-released-v1";

fn released() -> context_graph::ReleasedInterpretationDeclaration {
    context_graph::ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: INTERPRETATION.into(),
        target_adapter_version: "forget-fixture-v1".into(),
        target_anchor_declaration_version: "forget-fixture-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "forget-fixture-v1".into(),
            anchor_declaration_version: "forget-fixture-v1".into(),
        }],
        mapping_contract_id: "forget-anchor-contract".into(),
        mapping_contract_sha256: "forget-anchor-contract-v1".into(),
    }
}

/// Keeps every captured field and replaces the anchor set with `file = src/interpreted.rs`.
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
                "canonical_value": "src/interpreted.rs",
                "original_value": "src/interpreted.rs"
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

/// The host's journey for a released interpretation: build it over the Context's current history,
/// stage the matching search generation, and activate both. The Store must be closed.
fn build_and_activate_interpretation(database: &Path, context_id: ContextId) {
    let mapper: Arc<dyn context_graph::ReleasedConnectorInterpretation> =
        Arc::new(RenamingMapper(released()));
    let contract: Arc<dyn context_graph::InstalledInterpretationContract> =
        Arc::new(PreservingContract(released()));
    let interpretation = context_graph::ConnectorInterpretationRunner::build(
        context_graph::InterpretationBuildRequest {
            store: config(database, None),
            context_id,
            connector_id: CONNECTOR.into(),
            target_interpretation_version: INTERPRETATION.into(),
            mapper: mapper.clone(),
            contract: contract.clone(),
            max_events_per_batch: NonZeroU32::new(8).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(64 * 1024).unwrap(),
            lease_duration_ms: NonZeroU64::new(10_000).unwrap(),
        },
    )
    .expect("build the released interpretation");
    let store = open(database);
    let search = store
        .rebuild_event_search(context_graph::EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: Some(interpretation.generation_id),
            embedder: FixedEmbedder::registration(),
            profile: EventSearchProfile::default(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).unwrap(),
            max_passages_per_batch: NonZeroU32::new(8).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(64 * 1024).unwrap(),
            lease_duration_ms: NonZeroU64::new(10_000).unwrap(),
        })
        .expect("stage the interpreted search generation");
    drop(store);
    context_graph::ConnectorInterpretationRunner::activate(
        context_graph::InterpretationActivationRequest {
            store: config(database, None),
            context_id,
            connector_id: CONNECTOR.into(),
            target_interpretation_version: INTERPRETATION.into(),
            mapper,
            contract,
            interpretation_generation_id: interpretation.generation_id,
            replacement_search_generation_id: search.generation_id,
        },
    )
    .expect("activate the released interpretation");
}

fn sessions_by_interpreted_anchor(store: &Store, context_id: ContextId) -> Vec<String> {
    store
        .list_events_by_anchor(
            context_id,
            context_graph::AnchorKey("file".into()),
            context_graph::AnchorValue("\"src/interpreted.rs\"".into()),
            context_graph::EventRange::Count {
                after_seq: None,
                limit: NonZeroU32::new(10).unwrap(),
            },
        )
        .expect("anchor read")
        .into_iter()
        .map(|event| event.session_ref)
        .collect()
}

/// Forgetting one session erases the Context's activated interpretation with the rest of its
/// derived rows; the receipt names that interpretation so the host can build and activate it
/// again over the retained sessions, which then read through it exactly as before.
#[test]
fn a_session_forget_names_the_retired_interpretation_and_the_host_can_restore_it_for_the_retained_sessions()
 {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-interpretation.db");
    let store = open(&database);
    let context_id = store
        .create_context(CreateContext {
            name: "interpreted".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append(&store, context_id, "gone", "src-gone", "i-gone", 1);
    append(&store, context_id, "kept", "src-kept", "i-kept", 1);
    drop(store);
    build_and_activate_interpretation(&database, context_id);

    let store = open(&database);
    let mut before = sessions_by_interpreted_anchor(&store, context_id);
    before.sort();
    assert_eq!(
        before,
        vec!["gone".to_string(), "kept".to_string()],
        "both sessions read through the activated interpretation"
    );

    let receipt = store
        .forget_event_history(ForgetSelection::Session {
            context_id,
            connector_id: CONNECTOR.into(),
            session_ref: "gone".into(),
        })
        .expect("forget one session");
    assert!(receipt.changed);
    assert_eq!(receipt.events, 1);
    assert_eq!(
        receipt.retired_interpretations,
        vec![context_graph::RetiredInterpretation {
            connector_id: CONNECTOR.into(),
            target_interpretation_version: INTERPRETATION.into(),
            mapping_contract_id: "forget-anchor-contract".into(),
            mapping_contract_sha256: "forget-anchor-contract-v1".into(),
        }],
        "the receipt names the interpretation the host must rebuild"
    );
    assert!(
        sessions_by_interpreted_anchor(&store, context_id).is_empty(),
        "until it is rebuilt, the retained session reads with its captured fields"
    );
    drop(store);

    build_and_activate_interpretation(&database, context_id);
    let store = open(&database);
    assert_eq!(
        sessions_by_interpreted_anchor(&store, context_id),
        vec!["kept".to_string()],
        "the retained session reads through the restored interpretation; the forgotten one is gone"
    );
    let status = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("status after the host restored the interpretation");
    assert_eq!(status.contexts[0].captured_events, 1);
    assert_eq!(
        status.contexts[0].search.state,
        EventHistorySearchState::Ready
    );

    let nothing = store
        .forget_event_history(ForgetSelection::Session {
            context_id,
            connector_id: CONNECTOR.into(),
            session_ref: "absent".into(),
        })
        .expect("an unknown session is nothing to forget");
    assert!(
        nothing.retired_interpretations.is_empty(),
        "a forget that changes nothing retires nothing"
    );
}

/// A refused erasure changes nothing: the active search head, every passage and a find answer are
/// exactly what they were. The refusal is the engine's own: its one-shot durable purge fault fails
/// the erasure after admission and before publication, on history captured again after a forget.
/// The fault is process-wide, so the body runs alone in a child test process where no other
/// test's erasure can consume it.
#[test]
fn a_refused_forget_leaves_the_search_head_passages_and_find_answer_unchanged() {
    const CHILD: &str = "CG_EVENT_FORGET_REFUSAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "a_refused_forget_leaves_the_search_head_passages_and_find_answer_unchanged",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .expect("run the refusal in its own test process");
        assert!(status.success(), "the refusal proof failed: {status}");
        return;
    }
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-refused.db");
    let store = open(&database);
    let context_id = store
        .create_context(CreateContext {
            name: "refused".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append(&store, context_id, "s1", "src1", "x-1", 1);
    store
        .forget_event_history(ForgetSelection::Context { context_id })
        .expect("first forget");
    append(&store, context_id, "s1", "src1", "x-1", 1);
    project(&store, context_id);
    let status_before = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("status before");
    let find_before = store
        .find_event_history(
            ContextSelector::one(context_id),
            "forget fixture",
            EventSearchFilter::default(),
        )
        .expect("find before");
    drop(store);
    let before_reader = reader(&database);
    let heads = |reader: &ConsumerReader| {
        reader
            .execute(
                "SELECT context_id, generation_id, activated_at FROM event_search_heads",
                &HashMap::new(),
            )
            .expect("read heads")
            .rows
    };
    let heads_before = heads(&before_reader);
    let passages_before = count(
        &before_reader,
        "SELECT COUNT(*) AS n FROM event_search_passages WHERE context_id = $context_id",
        context_id,
    );
    drop(before_reader);

    let store = open(&database);
    let fault = contextdb_engine::Database::open_memory();
    fault.arm_vector_durable_purge_failure_for_test();
    let refused = store.forget_event_history(ForgetSelection::Context { context_id });
    assert!(
        fault.vector_durable_purge_failure_consumed_for_test(),
        "the erasure reached the engine's purge commit"
    );
    assert!(
        refused.is_err(),
        "the engine refuses this erasure: {refused:?}"
    );
    assert_eq!(
        store
            .event_history_status(ContextSelector::one(context_id))
            .expect("status after"),
        status_before
    );
    assert_eq!(
        store
            .find_event_history(
                ContextSelector::one(context_id),
                "forget fixture",
                EventSearchFilter::default(),
            )
            .expect("find after"),
        find_before
    );
    drop(store);
    let after_reader = reader(&database);
    assert_eq!(
        heads(&after_reader),
        heads_before,
        "the search head is exactly restored"
    );
    assert_eq!(
        count(
            &after_reader,
            "SELECT COUNT(*) AS n FROM event_search_passages WHERE context_id = $context_id",
            context_id,
        ),
        passages_before
    );
}

/// A crash after the head pointer is retired and before the erasure leaves history intact and a
/// Context that status and find report as not yet searchable, never as an error; the next
/// projection makes it searchable again.
#[cfg(context_graph_internal_test_support)]
#[test]
fn a_crash_between_retiring_the_head_and_the_erasure_reads_honestly_and_the_next_projection_repairs_it()
 {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("forget-crash.db");
    let store = open(&database);
    let context_id = store
        .create_context(CreateContext {
            name: "crash".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append(&store, context_id, "s1", "src1", "c-1", 1);
    project(&store, context_id);
    store
        .retire_event_search_head_for_test(context_id)
        .expect("stop after the first forget statement");

    let status = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("status reads the interrupted state");
    assert_eq!(status.contexts[0].captured_events, 1, "history is intact");
    assert_eq!(
        status.contexts[0].search.state,
        EventHistorySearchState::Pending
    );
    assert!(matches!(
        store.find_event_history(
            ContextSelector::one(context_id),
            "forget fixture",
            EventSearchFilter::default(),
        ),
        Err(CgError::HistorySearchNotReady { .. })
    ));

    project(&store, context_id);
    let repaired = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("status after the next projection");
    assert_eq!(
        repaired.contexts[0].search.state,
        EventHistorySearchState::Ready
    );
}
