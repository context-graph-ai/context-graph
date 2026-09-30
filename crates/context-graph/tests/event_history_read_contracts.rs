//! Independent retrieval contracts for event-history passages.
//!
//! Every fixture captures immutable events, builds the public projection, and then calls the
//! public Store or ConsumerReader retrieval surface. Expected event ids and ordering come from
//! the capture receipt; the deterministic role-aware embedder is test support, never a production
//! ranking helper.

#![cfg(context_graph_internal_test_support)]

use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
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
    EventSearchContextState, EventSearchFilter, EventSearchProfile, EvidenceKind, EvidenceProducer,
    Result, Store, StoreConfig,
};
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
use serde_json::json;

const SPACE: &str = "event-history-read-contracts-768";
const CONNECTOR: &str = "event-history-read-contracts";
const SESSION: &str = "event-history-read-session";
const SOURCE: &str = "event-history-read-source";
const EVENT_CONTEXT_SESSIONS_DDL: &str = "CREATE TABLE event_context_sessions (context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), connector_id TEXT NOT NULL, session_ref TEXT NOT NULL, event_count INTEGER NOT NULL, latest_occurred_at TIMESTAMP, PRIMARY KEY (context_id, connector_id, session_ref)) SYNC OFF";
const EVENT_CONTEXT_COUNTS_DDL: &str = "CREATE TABLE event_context_counts (context_id UUID CONTEXT_ID PRIMARY KEY REFERENCES contexts(id), captured_events INTEGER NOT NULL, complete_events INTEGER NOT NULL, opaque_events INTEGER NOT NULL, payload_refused_events INTEGER NOT NULL, source_declared_missing_events INTEGER NOT NULL) SYNC OFF";
const EVENT_SEARCH_SESSION_PROGRESS_DDL: &str = "CREATE TABLE event_search_session_progress (context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), connector_id TEXT NOT NULL, session_ref TEXT NOT NULL, event_count INTEGER NOT NULL, committed_event_count INTEGER NOT NULL DEFAULT 0, eligible_event_count INTEGER NOT NULL DEFAULT 0, excluded_event_count INTEGER NOT NULL DEFAULT 0, searchable_event_count INTEGER NOT NULL DEFAULT 0, no_searchable_text_event_count INTEGER NOT NULL DEFAULT 0, is_required BOOLEAN NOT NULL DEFAULT false, is_published BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (context_id, generation_id, connector_id, session_ref), FOREIGN KEY (context_id, generation_id) REFERENCES event_search_generations(context_id, id)) SYNC OFF";
const EVENT_SEARCH_WORK_POLICIES_DDL: &str = "CREATE TABLE event_search_work_policies (context_id UUID CONTEXT_ID PRIMARY KEY REFERENCES contexts(id), search_order TEXT NOT NULL, search_sessions INTEGER) SYNC OFF";
const EVENT_SEARCH_CURSORS_DDL: &str = "CREATE TABLE event_search_cursors (context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), connector_id TEXT NOT NULL, source_instance_ref TEXT NOT NULL, session_ref TEXT NOT NULL, committed_sequence INTEGER NOT NULL, open_tail_start_sequence INTEGER, open_tail_members JSON NOT NULL, open_tail_passage_id UUID, watermark_sha256 TEXT NOT NULL, updated_at TXID NOT NULL, PRIMARY KEY (context_id, generation_id, connector_id, source_instance_ref, session_ref), FOREIGN KEY (context_id, generation_id) REFERENCES event_search_generations(context_id, id)) SYNC OFF";

#[derive(Default)]
struct QueryRecorder {
    enabled: AtomicBool,
    sql: Mutex<Vec<String>>,
}

impl QueryRecorder {
    fn enable(&self) {
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

/// Test-only role-aware embedder. The public sealed entrypoints still prepare and verify both
/// document and query values; this hook maps fixture markers to deterministic directions.
struct DeterministicRoleAwareEmbedder {
    profile: &'static str,
}

impl DeterministicRoleAwareEmbedder {
    fn new(profile: &'static str) -> Arc<Self> {
        Arc::new(Self { profile })
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn vector_for(text: &str) -> Vec<f32> {
        let (first, second) = if text.contains("rank-first") {
            (1.0, 0.0)
        } else if text.contains("rank-middle") {
            (0.8, 0.6)
        } else if text.contains("rank-last") {
            (0.6, 0.8)
        } else {
            // Every ordinary fixture searches in the same direction. The document's marker is
            // intentionally the only ranking input, so the expected order below is authored from
            // capture ids rather than from a production vector helper.
            (1.0, 0.0)
        };
        let mut vector = vec![0.0; 768];
        vector[0] = first;
        vector[1] = second;
        vector
    }

    fn output(&self, text: &str) -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "event-history-read-contracts".into(),
                model_name: "deterministic-role-aware".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: Self::vector_for(text),
        }
    }

    fn inference_text(inference: &EventSearchInference) -> String {
        inference
            .token_ids()
            .iter()
            .map(|token| u8::try_from(*token).expect("test payload stays byte-sized"))
            .map(char::from)
            .collect()
    }
}

impl Embedder for DeterministicRoleAwareEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event-history projection must use role-aware preparation")
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
            model_name: "deterministic-role-aware".into(),
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
        self.profile.into()
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
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(&Self::inference_text(inference)))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(&Self::inference_text(inference)))
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

fn declaration(connector_id: &str) -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: connector_id.into(),
        adapter_version: "fixture-v1".into(),
        anchor_declaration_version: "fixture-v1".into(),
        anchor_schema: json!({"type":"object", "additionalProperties":true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

#[allow(clippy::too_many_arguments)]
fn append(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    session_ref: &str,
    source_instance_ref: &str,
    source_event_key: &str,
    ordinal: u64,
    text: &str,
) -> context_graph::ConversationEvent {
    append_at(
        store,
        context_id,
        connector_id,
        session_ref,
        source_instance_ref,
        source_event_key,
        ordinal,
        text,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_at(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    session_ref: &str,
    source_instance_ref: &str,
    source_event_key: &str,
    ordinal: u64,
    text: &str,
    occurred_at: Option<chrono::DateTime<chrono::Utc>>,
) -> context_graph::ConversationEvent {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(connector_id),
            source_instance_ref,
            vec![EventDraft {
                session_ref: session_ref.into(),
                source_event_key: source_event_key.into(),
                source_position: json!({"ordinal": ordinal}),
                occurred_at,
                author: "fixture-user".into(),
                kind: "message".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some(text.into()),
                payload: json!({}),
                anchors: json!({}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }],
        ))
        .expect("capture immutable event")
        .events
        .into_iter()
        .next()
        .expect("one captured event")
        .event
}

fn project(store: &Store, context_id: ContextId, embedder: &Arc<DeterministicRoleAwareEmbedder>) {
    store
        .catch_up_event_search(EventSearchBuildRequest {
            context_id,
            embedder: embedder.registration(),
            profile: EventSearchProfile::default(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(1).unwrap(),
            max_passages_per_batch: NonZeroU32::new(1).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(4_096).unwrap(),
            lease_duration_ms: NonZeroU64::new(1_000).unwrap(),
        })
        .expect("public projection completes");
}

fn filter(limit: u32) -> EventSearchFilter {
    EventSearchFilter {
        limit: NonZeroU32::new(limit).expect("non-zero limit"),
        ..EventSearchFilter::default()
    }
}

fn hit_event_ids(result: &context_graph::EventSearchResult) -> Vec<context_graph::EventId> {
    result
        .hits
        .iter()
        .map(|hit| {
            assert_eq!(
                hit.sources.len(),
                1,
                "each fixture capture is one passage source"
            );
            hit.sources[0].event_id
        })
        .collect()
}

#[test]
fn stale_ready_head_keeps_old_hits_and_reports_pending_capture_coverage() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = DeterministicRoleAwareEmbedder::new("stale-ready-profile");
    let store = Store::open_with_embedder_registrations(
        config(
            root.path().join("stale-ready.db"),
            embedder.embedding_space(),
        ),
        vec![embedder.registration()],
    )
    .expect("open store");
    let selected = context(&store, "stale-ready");
    let old = append(
        &store,
        selected,
        CONNECTOR,
        SESSION,
        SOURCE,
        "old",
        1,
        "retained old hit",
    );
    project(&store, selected, &embedder);
    let pending = append(
        &store,
        selected,
        CONNECTOR,
        SESSION,
        SOURCE,
        "pending",
        2,
        "new capture remains pending",
    );

    let result = store
        .find_event_history(ContextSelector::one(selected), "history query", filter(3))
        .expect("a stale ready head still answers its completed projection");

    assert_eq!(hit_event_ids(&result), vec![old.id]);
    assert_eq!(result.searched_context_ids, vec![selected]);
    assert_eq!(result.coverage.len(), 1);
    let coverage = &result.coverage[0];
    assert_eq!(coverage.context_id, selected);
    assert_eq!(coverage.state, EventSearchContextState::Ready);
    assert_eq!(coverage.captured_event_count, 2);
    assert_eq!(coverage.searchable_event_count, 1);
    assert_eq!(coverage.pending_event_count, 1);
    assert_ne!(
        old.id, pending.id,
        "the post-projection append is a distinct capture"
    );
}

#[test]
fn incompatible_ready_profiles_refuse_multi_context_retrieval_without_merging_hits() {
    let root = tempfile::tempdir().expect("tempdir");
    let first_embedder = DeterministicRoleAwareEmbedder::new("profile-a");
    let second_embedder = DeterministicRoleAwareEmbedder::new("profile-b");
    let store = Store::open_with_embedder_registrations(
        config(
            root.path().join("incompatible-profiles.db"),
            first_embedder.embedding_space(),
        ),
        vec![first_embedder.registration()],
    )
    .expect("open store");
    let first = context(&store, "profile-a");
    let second = context(&store, "profile-b");
    append(
        &store,
        first,
        CONNECTOR,
        SESSION,
        "source-a",
        "a",
        1,
        "rank-first",
    );
    append(
        &store,
        second,
        CONNECTOR,
        SESSION,
        "source-b",
        "b",
        1,
        "rank-middle",
    );
    project(&store, first, &first_embedder);
    project(&store, second, &second_embedder);

    let error = store
        .find_event_history(
            ContextSelector::new([first, second]).expect("selector"),
            "history query",
            filter(2),
        )
        .expect_err("ready heads with incompatible encoding profiles must not merge");

    assert!(matches!(
        error,
        CgError::EmbeddingSpaceConfigMismatch {
            field: "encoding_profile",
            ..
        }
    ));
}

#[test]
fn filtered_selected_contexts_rank_one_global_top_k_before_returning_hits() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = DeterministicRoleAwareEmbedder::new("global-ranking-profile");
    let store = Store::open_with_embedder_registrations(
        config(
            root.path().join("global-ranking.db"),
            embedder.embedding_space(),
        ),
        vec![embedder.registration()],
    )
    .expect("open store");
    let first_context = context(&store, "rank-first-context");
    let second_context = context(&store, "rank-second-context");
    let first = append(
        &store,
        first_context,
        CONNECTOR,
        "rank-first-session",
        "rank-first-source",
        "first",
        1,
        "rank-first",
    );
    let last = append(
        &store,
        first_context,
        CONNECTOR,
        "rank-last-session",
        "rank-last-source",
        "last",
        2,
        "rank-last",
    );
    let middle = append(
        &store,
        second_context,
        CONNECTOR,
        "rank-middle-session",
        "rank-middle-source",
        "middle",
        1,
        "rank-middle",
    );
    // The distractors share the selected connector but rank below the expected global top two.
    append(
        &store,
        second_context,
        "other-connector",
        SESSION,
        SOURCE,
        "connector-distractor",
        2,
        "rank-last connector distractor",
    );
    append(
        &store,
        second_context,
        CONNECTOR,
        "other-session",
        SOURCE,
        "session-distractor",
        3,
        "rank-last session distractor",
    );
    append(
        &store,
        second_context,
        CONNECTOR,
        SESSION,
        "other-source",
        "source-distractor",
        4,
        "rank-last source distractor",
    );
    project(&store, first_context, &embedder);
    project(&store, second_context, &embedder);

    let filtered = EventSearchFilter {
        connector_id: Some(CONNECTOR.into()),
        limit: NonZeroU32::new(2).unwrap(),
        ..EventSearchFilter::default()
    };
    let result = store
        .find_event_history(
            ContextSelector::new([first_context, second_context]).expect("selector"),
            "rank-query",
            filtered.clone(),
        )
        .expect("selected contexts and the connector filter share one ranking query");

    assert_eq!(
        result.searched_context_ids,
        vec![first_context, second_context]
    );
    assert_eq!(hit_event_ids(&result), vec![first.id, middle.id]);
    assert!(
        !hit_event_ids(&result).contains(&last.id),
        "the lower-ranked passage from the first context loses to the middle-ranked cross-context passage"
    );
    assert!(result.hits.iter().all(|hit| hit.applied_filter == filtered));
}

fn published_table(name: &str) -> ConsumerTable {
    let spec = TABLES
        .iter()
        .find(|spec| spec.name == name)
        .unwrap_or_else(|| panic!("published core schema contains {name}"));
    ConsumerTable {
        name: spec.name.into(),
        ddl: spec.ddl.into(),
    }
}

/// The event-history projection tables are node-local. This consumer declaration is intentionally
/// a literal snapshot of their persisted DDL: drift must refuse the reader instead of widening it.
fn event_history_read_declaration() -> ConsumerDeclaration {
    const EVENT_DDL: &str = "CREATE TABLE events (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), connector_id TEXT NOT NULL, adapter_version TEXT NOT NULL, anchor_declaration_version TEXT NOT NULL, source_instance_ref TEXT NOT NULL, session_ref TEXT NOT NULL, source_event_key TEXT NOT NULL, source_position JSON NOT NULL, seq INTEGER NOT NULL, content_hash TEXT NOT NULL, anchor_count INTEGER NOT NULL, anchor_set_hash TEXT NOT NULL, occurred_at TIMESTAMP, author TEXT NOT NULL, kind TEXT NOT NULL, render_role TEXT NOT NULL, interaction_ref TEXT, text TEXT, payload JSON NOT NULL DEFAULT '{}', anchors JSON NOT NULL DEFAULT '{}', source_remainder JSON NOT NULL DEFAULT '{}', coverage TEXT NOT NULL, UNIQUE (context_id, id), UNIQUE (context_id, connector_id, session_ref, source_event_key), UNIQUE (context_id, connector_id, source_instance_ref, seq)) IMMUTABLE SYNC OFF SYNC CONFLICT KEEP FIRST";
    const GENERATIONS_DDL: &str = "CREATE TABLE event_search_generations (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), interpretation_generation_id UUID, interpretation_snapshot JSON NOT NULL, interpretation_snapshot_sha256 TEXT NOT NULL, event_count INTEGER NOT NULL, ordered_source_set_sha256 TEXT NOT NULL, source_content_sha256 TEXT NOT NULL DEFAULT '', capture_ordinal INTEGER NOT NULL DEFAULT 0, renderer_id TEXT NOT NULL, renderer_version TEXT NOT NULL, renderer_profile JSON NOT NULL, embedding_space_id TEXT NOT NULL REFERENCES embedding_spaces(embedding_space_id), encoding_profile TEXT NOT NULL, state TEXT NOT NULL, required_event_count INTEGER NOT NULL DEFAULT 0, searchable_event_count INTEGER NOT NULL DEFAULT 0, searchable_excluded_event_count INTEGER NOT NULL DEFAULT 0, eligible_event_count INTEGER NOT NULL DEFAULT 0, rendered_event_count INTEGER NOT NULL DEFAULT 0, embedded_event_count INTEGER NOT NULL DEFAULT 0, excluded_event_count INTEGER NOT NULL DEFAULT 0, failed_event_count INTEGER NOT NULL DEFAULT 0, passage_count INTEGER NOT NULL DEFAULT 0, started_at TXID NOT NULL, completed_at TXID, finalized_event_revision_tx TXID, last_attempt_at TXID, last_attempt_token UUID NOT NULL, sanitized_failure TEXT, UNIQUE (context_id, id), FOREIGN KEY (context_id, interpretation_generation_id) REFERENCES event_interpretation_generations(context_id, id)) SYNC OFF";
    const PASSAGES_DDL: &str = "CREATE TABLE event_search_passages (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), connector_id TEXT NOT NULL, session_ref TEXT NOT NULL, source_instance_ref TEXT NOT NULL, passage_ordinal INTEGER NOT NULL, first_event_sequence INTEGER NOT NULL, last_event_sequence INTEGER NOT NULL, rendered_text TEXT NOT NULL, content_sha256 TEXT NOT NULL, vector_text VECTOR(768), dimension INTEGER NOT NULL, is_current BOOLEAN NOT NULL, is_searchable BOOLEAN NOT NULL DEFAULT true, created_at TXID NOT NULL, UNIQUE (context_id, id), UNIQUE (context_id, generation_id, id), UNIQUE (context_id, generation_id, connector_id, source_instance_ref, passage_ordinal), FOREIGN KEY (context_id, generation_id) REFERENCES event_search_generations(context_id, id)) HISTORY CURRENT ONLY SYNC OFF";
    const SOURCES_DDL: &str = "CREATE TABLE event_search_passage_sources (context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), passage_id UUID NOT NULL REFERENCES event_search_passages(id), event_id UUID NOT NULL REFERENCES events(id), connector_id TEXT NOT NULL, session_ref TEXT NOT NULL, source_instance_ref TEXT NOT NULL, source_order INTEGER NOT NULL, event_start_byte INTEGER, event_end_byte INTEGER, passage_start_byte INTEGER NOT NULL, passage_end_byte INTEGER NOT NULL, PRIMARY KEY (context_id, passage_id, source_order), FOREIGN KEY (context_id, generation_id, passage_id) REFERENCES event_search_passages(context_id, generation_id, id), FOREIGN KEY (context_id, event_id) REFERENCES events(context_id, id)) IMMUTABLE SYNC OFF";
    const HEADS_DDL: &str = "CREATE TABLE event_search_heads (context_id UUID CONTEXT_ID PRIMARY KEY REFERENCES contexts(id), generation_id UUID NOT NULL REFERENCES event_search_generations(id), activated_at TXID NOT NULL, FOREIGN KEY (context_id, generation_id) REFERENCES event_search_generations(context_id, id)) SYNC OFF";
    let local = [
        ("events", EVENT_DDL),
        ("event_search_generations", GENERATIONS_DDL),
        ("event_search_passages", PASSAGES_DDL),
        ("event_search_passage_sources", SOURCES_DDL),
        ("event_search_heads", HEADS_DDL),
    ];
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "reader".into(),
            tables: std::iter::once(published_table("contexts"))
                .chain(local.into_iter().map(|(name, ddl)| ConsumerTable {
                    name: name.into(),
                    ddl: ddl.into(),
                }))
                .chain(
                    [
                        ("event_context_counts", EVENT_CONTEXT_COUNTS_DDL),
                        ("event_context_sessions", EVENT_CONTEXT_SESSIONS_DDL),
                        (
                            "event_search_session_progress",
                            EVENT_SEARCH_SESSION_PROGRESS_DDL,
                        ),
                        ("event_search_work_policies", EVENT_SEARCH_WORK_POLICIES_DDL),
                        ("event_search_cursors", EVENT_SEARCH_CURSORS_DDL),
                    ]
                    .into_iter()
                    .map(|(name, ddl)| ConsumerTable {
                        name: name.into(),
                        ddl: ddl.into(),
                    }),
                )
                .collect(),
        },
        scope_labels: BTreeSet::new(),
    }
}

fn directory_image(path: &Path) -> Vec<(String, Vec<u8>)> {
    let mut image = std::fs::read_dir(path)
        .expect("read fixture directory")
        .map(|entry| {
            let entry = entry.expect("directory entry");
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read(entry.path()).expect("fixture file bytes"),
            )
        })
        .collect::<Vec<_>>();
    image.sort_by(|left, right| left.0.cmp(&right.0));
    image
}

#[test]
fn ready_consumer_reader_matches_store_results_and_keeps_retrieval_read_only() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("consumer-reader.db");
    let embedder = DeterministicRoleAwareEmbedder::new("consumer-reader-profile");
    let recorder = Arc::new(QueryRecorder::default());
    let store = Store::open_with_database_plugin_and_embedder_registrations(
        config(database.clone(), embedder.embedding_space()),
        recorder.clone(),
        vec![embedder.registration()],
    )
    .expect("open store");
    let selected = context(&store, "consumer-reader");
    let captured = append(
        &store,
        selected,
        CONNECTOR,
        SESSION,
        SOURCE,
        "reader",
        1,
        "reader parity hit",
    );
    project(&store, selected, &embedder);
    recorder.enable();
    let expected = store
        .find_event_history(ContextSelector::one(selected), "reader query", filter(1))
        .expect("Store retrieval records the expected public answer");
    assert_eq!(hit_event_ids(&expected), vec![captured.id]);
    let store_queries = recorder.queries();
    assert!(
        !store_queries.is_empty(),
        "the Store retrieval records database queries"
    );
    assert!(
        store_queries.iter().all(|sql| {
            let command = sql.trim_start().to_ascii_uppercase();
            command.starts_with("SELECT")
        }),
        "the recorded public retrieval queries are read-only"
    );
    drop(store);

    let before = directory_image(root.path());
    let mut reader = ConsumerReader::open(&database, event_history_read_declaration())
        .expect("ready event-history tables satisfy the read-only declaration");
    reader
        .attach_event_search_embedder_registrations(vec![embedder.registration()])
        .expect("attach the same role-aware registration without opening a writer");
    let actual = reader
        .find_event_history(ContextSelector::one(selected), "reader query", filter(1))
        .expect("read-only query returns the Store answer");
    drop(reader);

    assert_eq!(actual, expected);
    assert_eq!(
        directory_image(root.path()),
        before,
        "the read-only query leaves every database and companion byte unchanged"
    );
}

#[test]
fn many_session_status_and_find_use_constant_coverage_queries_and_record_scan_cost() {
    let root = tempfile::tempdir().expect("tempdir");
    let database = root.path().join("many-session-read-cost.db");
    let embedder = DeterministicRoleAwareEmbedder::new("many-session-read-cost-profile");
    let recorder = Arc::new(QueryRecorder::default());
    let store = Store::open_with_database_plugin_and_embedder_registrations(
        config(database, embedder.embedding_space()),
        recorder.clone(),
        vec![embedder.registration()],
    )
    .expect("open measured store");
    let selected = context(&store, "many-session-read-cost");
    for index in 0..128_u64 {
        append_at(
            &store,
            selected,
            CONNECTOR,
            &format!("many-session-{index:03}"),
            &format!("many-source-{index:03}"),
            &format!("many-event-{index:03}"),
            1,
            &format!("many session searchable marker {index:03}"),
            chrono::DateTime::from_timestamp(1_800_000_000 + index as i64, 0),
        );
    }
    project(&store, selected, &embedder);

    recorder.enable();
    let status = store
        .event_history_status(ContextSelector::one(selected))
        .expect("status remains readable before the archive is projected");
    let search = &status.contexts[0].search;
    assert_eq!(search.covered_events, 10);
    assert_eq!(search.pending_events, 0);
    assert_eq!(search.outside_scope_events, 118);
    let status_queries = recorder.queries();

    recorder.enable();
    let found = store
        .find_event_history(ContextSelector::one(selected), "many session", filter(5))
        .expect("find remains readable before the archive is projected");
    assert!(!found.hits.is_empty());
    assert_eq!(found.coverage[0].searchable_event_count, 10);
    assert_eq!(found.coverage[0].outside_scope_event_count, 118);
    let find_queries = recorder.queries();

    for (name, queries) in [("status", &status_queries), ("find", &find_queries)] {
        let search_queries = queries
            .iter()
            .filter(|sql| {
                sql.contains("event_search_work_policies")
                    || sql.contains("event_context_sessions")
                    || sql.contains("searchable_event_count")
                    || sql.contains("event_search_session_progress")
            })
            .collect::<Vec<_>>();
        assert!(
            search_queries.len() <= 5,
            "{name} used {} coverage queries for one Context: {search_queries:#?}",
            search_queries.len()
        );
        assert!(
            search_queries
                .iter()
                .filter(|sql| sql.contains("FROM event_search_session_progress"))
                .count()
                <= 1,
            "{name} must fold selected coverage in one progress query: {search_queries:#?}"
        );
        assert!(
            search_queries
                .iter()
                .filter(|sql| sql.contains("FROM event_context_sessions"))
                .count()
                <= 2,
            "{name} may read only the timestamped selected-session window: {search_queries:#?}"
        );
        assert!(
            search_queries.iter().all(|sql| {
                !sql.contains("COUNT(DISTINCT")
                    && !sql.contains(
                        "event_search_passage_sources INNER JOIN event_search_source_events",
                    )
            }),
            "{name} must not reconstruct coverage from passage-source membership: {search_queries:#?}"
        );
    }

    let selected_window = store
        .execute_sql_for_test(
            "SELECT connector_id, session_ref, event_count FROM event_context_sessions WHERE context_id = $context_id AND latest_occurred_at IS NOT NULL ORDER BY latest_occurred_at DESC, connector_id ASC, session_ref ASC LIMIT 10",
            &std::collections::HashMap::from([(
                "context_id".into(),
                contextdb_core::Value::Uuid(selected.as_uuid()),
            )]),
        )
        .expect("measure the exact selected-session query");
    assert_eq!(selected_window.rows.len(), 10);
    assert!(selected_window.trace.rows_examined <= 300);
    let generation_id = search
        .active_generation_id
        .expect("bounded projection publishes a ready head");
    let mut progress_params = std::collections::HashMap::from([
        (
            "context_id".into(),
            contextdb_core::Value::Uuid(selected.as_uuid()),
        ),
        (
            "generation_id".into(),
            contextdb_core::Value::Uuid(generation_id.as_uuid()),
        ),
    ]);
    let progress_filter = (0..10)
        .map(|index| {
            let session = format!("session_{index}");
            progress_params.insert(
                session.clone(),
                contextdb_core::Value::Text(format!("many-session-{:03}", 127 - index)),
            );
            format!("(connector_id = $connector AND session_ref = ${session})")
        })
        .collect::<Vec<_>>()
        .join(" OR ");
    progress_params.insert(
        "connector".into(),
        contextdb_core::Value::Text(CONNECTOR.into()),
    );
    let progress = store
        .execute_sql_for_test(
            &format!(
                "SELECT searchable_event_count, no_searchable_text_event_count FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND ({progress_filter})"
            ),
            &progress_params,
        )
        .expect("measure the exact selected-progress query");
    assert_eq!(progress.rows.len(), 10);
    assert!(progress.trace.rows_examined <= 300);
    eprintln!(
        "many-session coverage benchmark: status_total_queries={}, find_total_queries={}, status_coverage_queries={}, find_coverage_queries={}, selected_session_rows_examined={}, selected_progress_rows_examined={}",
        status_queries.len(),
        find_queries.len(),
        status_queries
            .iter()
            .filter(|sql| {
                sql.contains("event_search_work_policies")
                    || sql.contains("event_context_sessions")
                    || sql.contains("searchable_event_count")
                    || sql.contains("event_search_session_progress")
            })
            .count(),
        find_queries
            .iter()
            .filter(|sql| {
                sql.contains("event_search_work_policies")
                    || sql.contains("event_context_sessions")
                    || sql.contains("searchable_event_count")
                    || sql.contains("event_search_session_progress")
            })
            .count(),
        selected_window.trace.rows_examined,
        progress.trace.rows_examined,
    );
}
