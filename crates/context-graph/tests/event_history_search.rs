//! RED proof for public event-history retrieval.  The Store API is the only writable read path;
//! the database plugin observes its SQL after the public call.

#![cfg(context_graph_internal_test_support)]

use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::schema::TABLES;
use context_graph::{
    AppendEventBatch, CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable,
    ContextId, ContextSelector, CreateContext, DistanceMetric, Embedder, EmbedderRegistration,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchBuildRequest,
    EventSearchContextState, EventSearchFilter, EventSearchPassageId, EventSearchProfile,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
};
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
use serde_json::json;

#[derive(Default)]
struct QueryRecorder {
    enabled: AtomicBool,
    sql: Mutex<Vec<String>>,
}

impl QueryRecorder {
    fn enable_for_next_call(&self) {
        self.sql.lock().expect("query recorder lock").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn queries(&self) -> Vec<String> {
        self.sql.lock().expect("query recorder lock").clone()
    }
}

impl DatabasePlugin for QueryRecorder {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.sql
                .lock()
                .expect("query recorder lock")
                .push(sql.into());
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _elapsed: Duration, _outcome: &QueryOutcome) {}
}

fn store_with_recorder() -> (tempfile::TempDir, Store, Arc<QueryRecorder>) {
    let root = tempfile::tempdir().expect("tempdir");
    let recorder = Arc::new(QueryRecorder::default());
    let store = Store::open_with_database_plugin(
        StoreConfig {
            db_path: root.path().join("event-history-search.db"),
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        recorder.clone(),
    )
    .expect("open Store with query recorder");
    (root, store, recorder)
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

fn filter(limit: u32) -> EventSearchFilter {
    EventSearchFilter {
        limit: NonZeroU32::new(limit).expect("non-zero limit"),
        ..EventSearchFilter::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: "event-history-retrieval-red".into(),
        adapter_version: "fixture-v1".into(),
        anchor_declaration_version: "fixture-v1".into(),
        anchor_schema: json!({"type":"object", "additionalProperties":true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn append_captured_event(store: &Store, context_id: ContextId) {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "event-history-retrieval-source",
            vec![EventDraft {
                session_ref: "event-history-retrieval-session".into(),
                source_event_key: "not-ready-event".into(),
                source_position: json!({"ordinal": 1}),
                occurred_at: None,
                author: "Renée".into(),
                kind: "message".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some("incident rollback remains pending".into()),
                payload: json!({}),
                anchors: json!({}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }],
        ))
        .expect("append captured event");
}

const SEARCH_SPACE: &str = "event-history-readiness-768";

struct ReadinessEmbedder;

impl ReadinessEmbedder {
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
                system: "event-history-readiness".into(),
                model_name: "test".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }
}

impl Embedder for ReadinessEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event-history search uses role-aware preparation")
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
            model_name: "readiness".into(),
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
        "readiness-v1".into()
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

fn search_declaration() -> ConsumerDeclaration {
    const EVENTS: &str = "CREATE TABLE events (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), connector_id TEXT NOT NULL, adapter_version TEXT NOT NULL, anchor_declaration_version TEXT NOT NULL, source_instance_ref TEXT NOT NULL, session_ref TEXT NOT NULL, source_event_key TEXT NOT NULL, source_position JSON NOT NULL, seq INTEGER NOT NULL, content_hash TEXT NOT NULL, anchor_count INTEGER NOT NULL, anchor_set_hash TEXT NOT NULL, occurred_at TIMESTAMP, author TEXT NOT NULL, kind TEXT NOT NULL, render_role TEXT NOT NULL, interaction_ref TEXT, text TEXT, payload JSON NOT NULL DEFAULT '{}', anchors JSON NOT NULL DEFAULT '{}', source_remainder JSON NOT NULL DEFAULT '{}', coverage TEXT NOT NULL, UNIQUE (context_id, id), UNIQUE (context_id, connector_id, session_ref, source_event_key), UNIQUE (context_id, connector_id, source_instance_ref, seq)) IMMUTABLE SYNC OFF SYNC CONFLICT KEEP FIRST";
    const GENERATIONS: &str = "CREATE TABLE event_search_generations (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), interpretation_generation_id UUID, interpretation_snapshot JSON NOT NULL, interpretation_snapshot_sha256 TEXT NOT NULL, event_count INTEGER NOT NULL, ordered_source_set_sha256 TEXT NOT NULL, renderer_id TEXT NOT NULL, renderer_version TEXT NOT NULL, renderer_profile JSON NOT NULL, embedding_space_id TEXT NOT NULL REFERENCES embedding_spaces(embedding_space_id), encoding_profile TEXT NOT NULL, state TEXT NOT NULL, eligible_event_count INTEGER NOT NULL DEFAULT 0, rendered_event_count INTEGER NOT NULL DEFAULT 0, embedded_event_count INTEGER NOT NULL DEFAULT 0, excluded_event_count INTEGER NOT NULL DEFAULT 0, failed_event_count INTEGER NOT NULL DEFAULT 0, passage_count INTEGER NOT NULL DEFAULT 0, started_at TXID NOT NULL, completed_at TXID, sanitized_failure TEXT, UNIQUE (context_id, id), FOREIGN KEY (context_id, interpretation_generation_id) REFERENCES event_interpretation_generations(context_id, id)) SYNC OFF";
    const PASSAGES: &str = "CREATE TABLE event_search_passages (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), connector_id TEXT NOT NULL, session_ref TEXT NOT NULL, source_instance_ref TEXT NOT NULL, passage_ordinal INTEGER NOT NULL, first_event_sequence INTEGER NOT NULL, last_event_sequence INTEGER NOT NULL, rendered_text TEXT NOT NULL, content_sha256 TEXT NOT NULL, vector_text VECTOR(768), dimension INTEGER NOT NULL, is_current BOOLEAN NOT NULL, created_at TXID NOT NULL, UNIQUE (context_id, id), UNIQUE (context_id, generation_id, id), UNIQUE (context_id, generation_id, connector_id, source_instance_ref, passage_ordinal), FOREIGN KEY (context_id, generation_id) REFERENCES event_search_generations(context_id, generation_id)) SYNC OFF";
    const SOURCES: &str = "CREATE TABLE event_search_passage_sources (context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), passage_id UUID NOT NULL REFERENCES event_search_passages(id), event_id UUID NOT NULL REFERENCES events(id), connector_id TEXT NOT NULL, session_ref TEXT NOT NULL, source_instance_ref TEXT NOT NULL, source_order INTEGER NOT NULL, event_start_byte INTEGER, event_end_byte INTEGER, passage_start_byte INTEGER NOT NULL, passage_end_byte INTEGER NOT NULL, PRIMARY KEY (context_id, passage_id, source_order), FOREIGN KEY (context_id, generation_id, passage_id) REFERENCES event_search_passages(context_id, generation_id, id), FOREIGN KEY (context_id, event_id) REFERENCES events(context_id, id)) IMMUTABLE SYNC OFF";
    const HEADS: &str = "CREATE TABLE event_search_heads (context_id UUID CONTEXT_ID PRIMARY KEY REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), activated_at TXID NOT NULL, FOREIGN KEY (context_id, generation_id) REFERENCES event_search_generations(context_id, id)) SYNC OFF";
    let local = [
        ("events", EVENTS),
        ("event_search_generations", GENERATIONS),
        ("event_search_passages", PASSAGES),
        ("event_search_passage_sources", SOURCES),
        ("event_search_heads", HEADS),
    ];
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "readiness".into(),
            tables: std::iter::once(published_table("contexts"))
                .chain(local.into_iter().map(|(name, ddl)| ConsumerTable {
                    name: name.into(),
                    ddl: ddl.into(),
                }))
                .collect(),
        },
        scope_labels: BTreeSet::new(),
    }
}

fn published_table(name: &str) -> ConsumerTable {
    let table = TABLES
        .iter()
        .find(|table| table.name == name)
        .unwrap_or_else(|| panic!("published schema contains {name}"));
    ConsumerTable {
        name: table.name.into(),
        ddl: table.ddl.into(),
    }
}

fn project_ready(store: &Store, context_id: ContextId, embedder: &Arc<ReadinessEmbedder>) {
    store
        .catch_up_event_search(EventSearchBuildRequest {
            context_id,
            embedder: embedder.registration(),
            profile: EventSearchProfile::default(),
            max_events_per_batch: NonZeroU32::new(1).unwrap(),
            max_passages_per_batch: NonZeroU32::new(1).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(4_096).unwrap(),
            lease_duration_ms: NonZeroU64::new(1_000).unwrap(),
        })
        .expect("project captured history");
}

#[test]
fn no_ready_captured_history_refuses_but_all_empty_history_succeeds_for_store_and_consumer() {
    let (root, store, recorder) = store_with_recorder();
    let empty = context(&store, "empty");
    let not_ready = context(&store, "not-ready");
    append_captured_event(&store, not_ready);
    recorder.enable_for_next_call();
    let error = store
        .find_event_history(
            ContextSelector::new([not_ready, empty]).expect("selector"),
            "incident rollback",
            filter(3),
        )
        .expect_err("captured history without a ready generation refuses");
    assert!(
        matches!(error, CgError::HistorySearchNotReady { context_ids } if context_ids == vec![empty, not_ready])
    );
    let all_empty = store
        .find_event_history(ContextSelector::one(empty), "incident rollback", filter(3))
        .expect("an all-empty selection succeeds with zero hits");
    assert_eq!(all_empty.coverage[0].state, EventSearchContextState::Empty);
    assert!(all_empty.hits.is_empty());
    let sql = recorder.queries();
    assert!(
        !sql.is_empty(),
        "coverage comes from the Store, never a source-file fallback"
    );
    assert!(
        sql.iter().all(|statement| {
            let command = statement.trim_start().to_ascii_uppercase();
            !command.starts_with("INSERT")
                && !command.starts_with("UPDATE")
                && !command.starts_with("DELETE")
                && !command.starts_with("CREATE")
        }),
        "the public history query must not write"
    );
    let database = store.db_path().to_path_buf();
    drop(store);

    let reader = ConsumerReader::open(&database, search_declaration()).expect("open consumer");
    let error = reader
        .find_event_history(
            ContextSelector::new([not_ready, empty]).expect("selector"),
            "incident rollback",
            filter(3),
        )
        .expect_err("consumer refuses captured history without a ready generation");
    assert!(
        matches!(error, CgError::HistorySearchNotReady { context_ids } if context_ids == vec![empty, not_ready])
    );
    let all_empty = reader
        .find_event_history(ContextSelector::one(empty), "incident rollback", filter(3))
        .expect("consumer all-empty selection succeeds");
    assert_eq!(all_empty.coverage[0].state, EventSearchContextState::Empty);
    assert!(all_empty.hits.is_empty());
    drop(reader);
    drop(root);
}

#[test]
fn mixed_ready_and_not_ready_contexts_search_the_ready_head_for_store_and_consumer() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("mixed-readiness.db");
    let embedder = Arc::new(ReadinessEmbedder);
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: database.clone(),
            extra_embedding_spaces: vec![embedder.embedding_space()],
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        vec![embedder.registration()],
    )
    .expect("open store");
    let ready = context(&store, "ready");
    let not_ready = context(&store, "not-ready");
    append_captured_event(&store, ready);
    append_captured_event(&store, not_ready);
    project_ready(&store, ready, &embedder);

    let expected = store
        .find_event_history(
            ContextSelector::new([ready, not_ready]).expect("selector"),
            "incident rollback",
            filter(3),
        )
        .expect("ready selected Context remains searchable");
    assert_eq!(expected.searched_context_ids, vec![ready]);
    assert_eq!(expected.hits.len(), 1);
    assert!(expected.coverage.iter().any(|coverage| {
        coverage.context_id == not_ready && coverage.state == EventSearchContextState::NotReady
    }));
    drop(store);

    let mut reader = ConsumerReader::open(&database, search_declaration()).expect("open consumer");
    reader
        .attach_event_search_embedder_registrations(vec![embedder.registration()])
        .expect("attach search embedder");
    let actual = reader
        .find_event_history(
            ContextSelector::new([ready, not_ready]).expect("selector"),
            "incident rollback",
            filter(3),
        )
        .expect("consumer keeps the ready Context searchable");
    assert_eq!(actual, expected);
}

#[test]
fn show_and_source_links_are_database_only_and_missing_passages_are_typed_empty_results() {
    let (_root, store, _recorder) = store_with_recorder();
    let selected = context(&store, "selected");
    let other = context(&store, "other");
    let absent = EventSearchPassageId::new_v7();
    assert_eq!(
        store
            .get_event_search_passage(ContextSelector::one(selected), absent)
            .expect("database-only show of an absent passage"),
        None
    );
    assert_eq!(
        store
            .list_events_for_passage(ContextSelector::one(other), absent)
            .expect("database-only source expansion of an absent passage"),
        Vec::new()
    );
}

#[test]
fn empty_selector_is_refused_while_a_valid_selector_is_constructible() {
    let (_root, store, _recorder) = store_with_recorder();
    let error = ContextSelector::new(Vec::<ContextId>::new()).expect_err("empty selector refuses");
    assert!(matches!(error, CgError::EmptyContextSelector));
    let selector = ContextSelector::one(context(&store, "one"));
    assert_eq!(selector.context_ids().len(), 1);
}

#[test]
fn read_only_missing_store_does_not_create_a_database() {
    let root = tempfile::tempdir().expect("tempdir");
    let absent = root.path().join("missing-event-history.db");
    let error = ConsumerReader::open(&absent, ConsumerDeclaration::default())
        .expect_err("read-only opening a missing database refuses");
    assert!(matches!(error, CgError::StoreMissing { .. }));
    assert!(
        !absent.exists(),
        "read-only retrieval never creates a database"
    );
}
