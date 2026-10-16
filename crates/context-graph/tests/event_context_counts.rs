//! Each Context's captured-event totals live in one node-local count row that capture and forget
//! keep current, so `find` classification and the status coverage line never count the events
//! table. Expected totals come from counting the events themselves, never from the row under test.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CgError, ConsumerReader, ContextId, ContextSelector, CreateContext,
    DistanceMetric, Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput,
    EmbeddingSpace, EmbeddingSpaceBinding, EventAdapterDeclaration, EventCoverage,
    EventCoverageCounts, EventDraft, EventRenderRole, EventSearchBuildRequest,
    EventSearchContextState, EventSearchFilter, EventSearchProfile, EvidenceKind, EvidenceProducer,
    ForgetSelection, Result, Store, StoreConfig, event_history_reader_declaration,
};
use contextdb_core::Value;
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
use serde_json::json;

const SPACE: &str = "event-context-counts-768";
const CONNECTOR: &str = "event-context-counts-connector";

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
                system: "event-context-counts".into(),
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
        "event-context-counts-profile".into()
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

/// Records every statement the Store's engine is asked to run while enabled.
#[derive(Default)]
struct QueryRecorder {
    enabled: AtomicBool,
    sql: Mutex<Vec<String>>,
}

impl QueryRecorder {
    fn start(&self) {
        self.sql.lock().expect("recorder lock").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn stop(&self) -> Vec<String> {
        self.enabled.store(false, Ordering::SeqCst);
        std::mem::take(&mut *self.sql.lock().expect("recorder lock"))
    }
}

impl DatabasePlugin for QueryRecorder {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.sql.lock().expect("recorder lock").push(sql.into());
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _elapsed: Duration, _outcome: &QueryOutcome) {}
}

fn open(path: &Path) -> (Store, Arc<QueryRecorder>) {
    let recorder = Arc::new(QueryRecorder::default());
    let store = Store::open_with_database_plugin_and_embedder_registrations(
        StoreConfig {
            db_path: path.to_path_buf(),
            extra_embedding_spaces: vec![FixedEmbedder.embedding_space()],
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        recorder.clone(),
        vec![FixedEmbedder::registration()],
    )
    .expect("open store");
    (store, recorder)
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
        adapter_version: "counts-fixture-v1".into(),
        anchor_declaration_version: "counts-fixture-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(session: &str, key: &str, ordinal: u64, coverage: EventCoverage) -> EventDraft {
    EventDraft {
        session_ref: session.into(),
        source_event_key: key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "fixture-user".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("count fixture text {key}")),
        payload: json!({"detail": key}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage,
    }
}

/// Capture `drafts` from one source file; returns how many were newly inserted.
fn capture(store: &Store, context_id: ContextId, source: &str, drafts: Vec<EventDraft>) -> u64 {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            source,
            drafts,
        ))
        .expect("capture events")
        .inserted
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

fn params(context_id: ContextId) -> HashMap<String, Value> {
    HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))])
}

fn int(value: &Value) -> u64 {
    match value {
        Value::Int64(value) => u64::try_from(*value).expect("non-negative"),
        other => panic!("expected an integer, got {other:?}"),
    }
}

/// Totals as (captured, per-coverage), read from the count row; `None` when the row is absent.
fn count_row(store: &Store, context_id: ContextId) -> Option<(u64, EventCoverageCounts)> {
    let answer = store
        .execute_sql_for_test(
            "SELECT captured_events, complete_events, opaque_events, payload_refused_events, source_declared_missing_events FROM event_context_counts WHERE context_id = $context_id",
            &params(context_id),
        )
        .expect("read the count row");
    assert!(answer.rows.len() <= 1, "one count row per Context");
    answer.rows.first().map(|row| {
        (
            int(&row[0]),
            EventCoverageCounts {
                complete: int(&row[1]),
                opaque: int(&row[2]),
                payload_refused: int(&row[3]),
                source_declared_missing: int(&row[4]),
            },
        )
    })
}

/// The same totals counted from the events themselves.
fn counted_from_events(store: &Store, context_id: ContextId) -> (u64, EventCoverageCounts) {
    let count = |sql: &str, params: HashMap<String, Value>| -> u64 {
        let answer = store
            .execute_sql_for_test(sql, &params)
            .expect("count events");
        int(&answer.rows[0][0])
    };
    let by_coverage = |coverage: &str| {
        let mut with = params(context_id);
        with.insert("coverage".into(), Value::Text(coverage.into()));
        count(
            "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id AND coverage = $coverage",
            with,
        )
    };
    (
        count(
            "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
            params(context_id),
        ),
        EventCoverageCounts {
            complete: by_coverage("complete"),
            opaque: by_coverage("opaque"),
            payload_refused: by_coverage("payload_refused"),
            source_declared_missing: by_coverage("source_declared_missing"),
        },
    )
}

fn assert_row_matches_events(store: &Store, context_id: ContextId, captured: u64) {
    let truth = counted_from_events(store, context_id);
    assert_eq!(truth.0, captured, "the fixture captured what it says");
    assert_eq!(
        count_row(store, context_id),
        Some(truth),
        "the count row equals the events it summarizes"
    );
}

/// Status reports the same captured total and coverage line through both doors.
fn assert_status_reports(path: &Path, store: &Store, context_id: ContextId, captured: u64) {
    let truth = counted_from_events(store, context_id);
    let status = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("store status");
    assert_eq!(status.contexts[0].captured_events, captured);
    assert_eq!(status.contexts[0].coverage, truth.1);
    let reader = ConsumerReader::open(path, event_history_reader_declaration()).expect("reader");
    let read = reader
        .event_history_status(ContextSelector::one(context_id))
        .expect("reader status");
    assert_eq!(read.contexts[0].captured_events, captured);
    assert_eq!(read.contexts[0].coverage, truth.1);
}

#[test]
fn the_count_row_tracks_appends_duplicate_reappends_and_a_second_file_in_the_same_session() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("counts-append.db");
    let (store, _recorder) = open(&path);
    let context_id = context(&store, "counted");
    let untouched = context(&store, "untouched");

    let first_file = || {
        vec![
            draft("s1", "a-1", 1, EventCoverage::Complete),
            draft("s1", "a-2", 2, EventCoverage::Opaque),
            draft("s1", "a-3", 3, EventCoverage::PayloadRefused),
        ]
    };
    assert_eq!(capture(&store, context_id, "file-1", first_file()), 3);
    assert_row_matches_events(&store, context_id, 3);

    assert_eq!(
        capture(&store, context_id, "file-1", first_file()),
        0,
        "the same records are already present"
    );
    assert_row_matches_events(&store, context_id, 3);

    assert_eq!(
        capture(
            &store,
            context_id,
            "file-2",
            vec![
                draft("s1", "b-1", 1, EventCoverage::Complete),
                draft("s1", "b-2", 2, EventCoverage::SourceDeclaredMissing),
            ],
        ),
        2
    );
    assert_row_matches_events(&store, context_id, 5);
    assert_eq!(
        count_row(&store, context_id).expect("row").1,
        EventCoverageCounts {
            complete: 2,
            opaque: 1,
            payload_refused: 1,
            source_declared_missing: 1,
        }
    );
    assert_eq!(counted_from_events(&store, untouched).0, 0);

    assert_status_reports(&path, &store, context_id, 5);
    assert_status_reports(&path, &store, untouched, 0);
}

#[test]
fn forgetting_a_session_and_then_the_context_rewrites_the_count_row_from_what_remains() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("counts-forget.db");
    let (store, _recorder) = open(&path);
    let context_id = context(&store, "forgotten");
    capture(
        &store,
        context_id,
        "file-1",
        vec![
            draft("s1", "a-1", 1, EventCoverage::Complete),
            draft("s1", "a-2", 2, EventCoverage::Opaque),
        ],
    );
    capture(
        &store,
        context_id,
        "file-2",
        vec![draft("s2", "b-1", 1, EventCoverage::Complete)],
    );
    project(&store, context_id);
    assert_row_matches_events(&store, context_id, 3);

    let receipt = store
        .forget_event_history(ForgetSelection::Session {
            context_id,
            connector_id: CONNECTOR.into(),
            session_ref: "s1".into(),
        })
        .expect("forget one session");
    assert_eq!(receipt.events, 2);
    assert_row_matches_events(&store, context_id, 1);
    assert_status_reports(&path, &store, context_id, 1);

    project(&store, context_id);
    let receipt = store
        .forget_event_history(ForgetSelection::Context { context_id })
        .expect("forget the Context");
    assert_eq!(receipt.events, 1);
    assert_row_matches_events(&store, context_id, 0);
    assert_status_reports(&path, &store, context_id, 0);
    let found = store
        .find_event_history(
            ContextSelector::one(context_id),
            "count fixture",
            EventSearchFilter::default(),
        )
        .expect("an emptied Context finds nothing");
    assert_eq!(found.coverage[0].state, EventSearchContextState::Empty);
    assert_eq!(found.coverage[0].captured_event_count, 0);
}

/// A crash between retiring the count row and rewriting it after a forget leaves no row: status,
/// find and the next capture recount that Context from its events instead of trusting a stale total.
#[test]
fn a_missing_count_row_is_recounted_from_events_by_status_find_and_the_next_capture() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("counts-missing.db");
    let (store, _recorder) = open(&path);
    let context_id = context(&store, "interrupted");
    capture(
        &store,
        context_id,
        "file-1",
        vec![
            draft("s1", "a-1", 1, EventCoverage::Complete),
            draft("s1", "a-2", 2, EventCoverage::Opaque),
        ],
    );
    store
        .retire_event_context_counts_for_test(context_id)
        .expect("retire the count row as a forget does first");
    assert_eq!(count_row(&store, context_id), None);

    assert_status_reports(&path, &store, context_id, 2);
    assert!(matches!(
        store.find_event_history(
            ContextSelector::one(context_id),
            "count fixture",
            EventSearchFilter::default(),
        ),
        Err(CgError::HistorySearchNotReady { .. })
    ));

    capture(
        &store,
        context_id,
        "file-1",
        vec![draft("s1", "a-3", 3, EventCoverage::Complete)],
    );
    assert_row_matches_events(&store, context_id, 3);
}

/// Classification of every selected Context reads its count row; no statement counts events.
#[test]
fn find_and_status_read_the_count_row_instead_of_counting_events() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("counts-find.db");
    let (store, recorder) = open(&path);
    let contexts = ["one", "two", "three"]
        .into_iter()
        .map(|name| {
            let context_id = context(&store, name);
            capture(
                &store,
                context_id,
                &format!("file-{name}"),
                vec![
                    draft(name, &format!("{name}-1"), 1, EventCoverage::Complete),
                    draft(name, &format!("{name}-2"), 2, EventCoverage::Complete),
                ],
            );
            project(&store, context_id);
            context_id
        })
        .collect::<Vec<_>>();

    recorder.start();
    let found = store
        .find_event_history(
            ContextSelector::new(contexts.iter().copied()).expect("selector"),
            "count fixture",
            EventSearchFilter::default(),
        )
        .expect("find over three ready Contexts");
    let sql = recorder.stop();
    assert!(!found.hits.is_empty());
    assert!(found.coverage.iter().all(|coverage| {
        coverage.state == EventSearchContextState::Ready
            && coverage.captured_event_count == 2
            && coverage.pending_event_count == 0
    }));
    assert!(
        sql.iter()
            .any(|statement| statement.contains("FROM event_context_counts")),
        "classification reads the count rows: {sql:#?}"
    );
    assert!(
        !sql.iter()
            .any(|statement| statement.contains("COUNT(") && statement.contains("FROM events")),
        "find counts no events: {sql:#?}"
    );

    recorder.start();
    let status = store
        .event_history_status(ContextSelector::new(contexts.iter().copied()).expect("selector"))
        .expect("status");
    let sql = recorder.stop();
    assert!(
        status
            .contexts
            .iter()
            .all(|context| { context.captured_events == 2 && context.coverage.complete == 2 })
    );
    assert!(
        !sql.iter().any(|statement| {
            statement.contains("COUNT(")
                && statement.contains("FROM events")
                && !statement.contains("source_instance_ref")
        }),
        "the status capture and coverage lines count no events; only per-stream totals do: {sql:#?}"
    );
}

/// The source links of a find's hits are read one passage at a time by the link table's primary-key
/// prefix, never as one list over the Context.
#[test]
fn find_reads_each_hits_source_links_by_its_passage_key() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("counts-links.db");
    let (store, recorder) = open(&path);
    let context_id = context(&store, "linked");
    for file in 1..=4 {
        capture(
            &store,
            context_id,
            &format!("file-{file}"),
            (1..=3)
                .map(|ordinal| {
                    draft(
                        &format!("s{file}"),
                        &format!("f{file}-{ordinal}"),
                        ordinal,
                        EventCoverage::Complete,
                    )
                })
                .collect(),
        );
    }
    project(&store, context_id);

    recorder.start();
    let found = store
        .find_event_history(
            ContextSelector::one(context_id),
            "count fixture",
            EventSearchFilter::default(),
        )
        .expect("find");
    let sql = recorder.stop();
    assert!(found.hits.len() > 1, "more than one passage is found");
    let link_reads = sql
        .iter()
        .filter(|statement| statement.contains("FROM event_search_passage_sources"))
        .collect::<Vec<_>>();
    assert_eq!(
        link_reads.len(),
        found.hits.len(),
        "one link read per hit: {link_reads:#?}"
    );
    for statement in &link_reads {
        assert!(
            statement.contains("context_id = $")
                && statement.contains("passage_id = $")
                && statement.contains("ORDER BY source_order")
                && !statement.contains(" IN ("),
            "each link read is an equality read on the passage key: {statement}"
        );
    }
    for hit in &found.hits {
        assert!(!hit.sources.is_empty());
        assert!(
            hit.sources
                .windows(2)
                .all(|pair| pair[0].source_order < pair[1].source_order)
        );
        assert!(hit.sources.iter().all(|source| {
            source.passage_id == hit.passage_id && source.context_id == hit.context_id
        }));
    }
}

/// `cg show`'s passage read reads the passage's source links once, by the passage key, and the
/// engine answers that read through the passage-first link index rather than walking the Context's
/// links through a `context_id`-only index.
#[test]
fn show_reads_its_source_links_once_through_the_passage_first_index() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("counts-show.db");
    let (store, recorder) = open(&path);
    let context_id = context(&store, "shown");
    for file in 1..=3 {
        capture(
            &store,
            context_id,
            &format!("file-{file}"),
            (1..=3)
                .map(|ordinal| {
                    draft(
                        &format!("s{file}"),
                        &format!("f{file}-{ordinal}"),
                        ordinal,
                        EventCoverage::Complete,
                    )
                })
                .collect(),
        );
    }
    project(&store, context_id);
    let hit = store
        .find_event_history(
            ContextSelector::one(context_id),
            "count fixture",
            EventSearchFilter::default(),
        )
        .expect("find")
        .hits
        .into_iter()
        .next()
        .expect("a hit to show");

    recorder.start();
    let passage = store
        .get_event_search_passage(ContextSelector::one(context_id), hit.passage_id)
        .expect("show")
        .expect("the found passage exists");
    let sql = recorder.stop();
    assert_eq!(passage.sources, hit.sources, "show links what find linked");
    let link_reads = sql
        .iter()
        .filter(|statement| statement.contains("FROM event_search_passage_sources"))
        .collect::<Vec<_>>();
    assert_eq!(link_reads.len(), 1, "show reads its links once: {sql:#?}");

    let traced = store
        .execute_sql_for_test(
            link_reads[0],
            &HashMap::from([
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                (
                    "passage_id".to_string(),
                    Value::Uuid(hit.passage_id.as_uuid()),
                ),
            ]),
        )
        .expect("rerun the recorded link read");
    assert_eq!(
        traced.trace.physical_plan, "IndexScan",
        "{:#?}",
        traced.trace
    );
    assert_eq!(traced.rows.len(), passage.sources.len());
    assert_eq!(
        traced.trace.rows_examined as usize,
        passage.sources.len(),
        "show reads each returned source link once: {:#?}",
        traced.trace
    );
}

/// Releases a waiting appender the moment the forget's PURGE is issued — after the forget has read
/// its plan — and lets the purge run only once that append has committed.
#[derive(Default)]
struct InterleaveOnPurge {
    armed: AtomicBool,
    /// Also release at a standalone count-row DELETE, so moving retirement outside the purge
    /// cannot evade the interleaving.
    release_at_count_row_delete: AtomicBool,
    release: Mutex<Option<Sender<()>>>,
    committed: Mutex<Option<Receiver<()>>>,
    observe_armed: AtomicBool,
    observe: Mutex<Option<Sender<()>>>,
    observed: Mutex<Option<Receiver<()>>>,
}

impl DatabasePlugin for InterleaveOnPurge {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        // Whichever the forget reaches first: its purge or a standalone count-row DELETE.
        let at_release_point = sql.trim_start().starts_with("PURGE FROM")
            || (self.release_at_count_row_delete.load(Ordering::SeqCst)
                && sql.contains("DELETE FROM")
                && sql.contains("event_context_counts"));
        if at_release_point && self.armed.swap(false, Ordering::SeqCst) {
            let release = self
                .release
                .lock()
                .expect("release lock")
                .take()
                .expect("armed once");
            release.send(()).expect("release the concurrent append");
            let committed = self
                .committed
                .lock()
                .expect("committed lock")
                .take()
                .expect("armed once");
            committed
                .recv_timeout(Duration::from_secs(60))
                .expect("the concurrent append commits before the purge runs");
        }
        Ok(())
    }

    /// The purge has committed and the forget has not yet rewritten the count row: let a reader
    /// look at the row in exactly that window.
    fn post_query(&self, sql: &str, _elapsed: Duration, _outcome: &QueryOutcome) {
        if sql.trim_start().starts_with("PURGE FROM")
            && self.observe_armed.swap(false, Ordering::SeqCst)
        {
            let observe = self
                .observe
                .lock()
                .expect("observe lock")
                .take()
                .expect("armed once");
            observe.send(()).expect("release the observer");
            let observed = self
                .observed
                .lock()
                .expect("observed lock")
                .take()
                .expect("armed once");
            observed
                .recv_timeout(Duration::from_secs(60))
                .expect("the observer read the count row");
        }
    }
}

fn open_with_plugin(path: &Path, plugin: Arc<dyn DatabasePlugin>) -> Store {
    Store::open_with_database_plugin_and_embedder_registrations(
        StoreConfig {
            db_path: path.to_path_buf(),
            extra_embedding_spaces: vec![FixedEmbedder.embedding_space()],
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        plugin,
        vec![FixedEmbedder::registration()],
    )
    .expect("open store with an interleaving plugin")
}

/// Capture one event on `store` when `release` fires, and say so when it has committed.
fn append_on_release(
    store: Store,
    context_id: ContextId,
    source: &'static str,
    session: &'static str,
    key: &'static str,
    release: Receiver<()>,
    committed: Sender<()>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        release
            .recv_timeout(Duration::from_secs(60))
            .expect("the forget reached its purge");
        capture(
            &store,
            context_id,
            source,
            vec![draft(session, key, 1, EventCoverage::Complete)],
        );
        committed.send(()).expect("report the commit");
    })
}

/// An append that commits between a session forget's plan read and its purge cannot leave a stale
/// total behind: the count row is erased by the same purge as the history, and what the store
/// reports afterwards is what its surviving events say.
#[test]
fn an_append_committing_before_a_session_forgets_purge_leaves_no_stale_total() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("counts-race-before-purge.db");
    let plugin = Arc::new(InterleaveOnPurge::default());
    let store = open_with_plugin(&path, plugin.clone());
    let context_id = context(&store, "append before purge");
    capture(
        &store,
        context_id,
        "file-1",
        vec![
            draft("s1", "a-1", 1, EventCoverage::Complete),
            draft("s1", "a-2", 2, EventCoverage::Opaque),
        ],
    );
    assert_row_matches_events(&store, context_id, 2);

    let (release, release_rx) = channel();
    let (committed, committed_rx) = channel();
    *plugin.release.lock().expect("release lock") = Some(release);
    *plugin.committed.lock().expect("committed lock") = Some(committed_rx);
    plugin.armed.store(true, Ordering::SeqCst);
    let appending = append_on_release(
        store.clone(),
        context_id,
        "file-2",
        "s2",
        "b-1",
        release_rx,
        committed,
    );
    let (observe, observe_rx) = channel();
    let (observed, observed_rx) = channel();
    *plugin.observe.lock().expect("observe lock") = Some(observe);
    *plugin.observed.lock().expect("observed lock") = Some(observed_rx);
    plugin.observe_armed.store(true, Ordering::SeqCst);
    let between = Arc::new(Mutex::new(None));
    let observer = {
        let store = store.clone();
        let between = Arc::clone(&between);
        std::thread::spawn(move || {
            observe_rx
                .recv_timeout(Duration::from_secs(60))
                .expect("the purge committed");
            *between.lock().expect("observation lock") = Some(count_row(&store, context_id));
            observed.send(()).expect("report the observation");
        })
    };

    let receipt = store
        .forget_event_history(ForgetSelection::Session {
            context_id,
            connector_id: CONNECTOR.into(),
            session_ref: "s1".into(),
        })
        .expect("forget the first session");
    appending.join().expect("the concurrent append succeeded");
    observer.join().expect("the observer read the count row");
    assert!(
        !plugin.armed.load(Ordering::SeqCst),
        "the append ran inside the forget, between its plan read and its purge"
    );
    assert!(
        !plugin.observe_armed.load(Ordering::SeqCst),
        "the count row was read between the purge and the recount"
    );
    assert_eq!(
        receipt.events, 2,
        "the forget erased the session it planned"
    );

    assert_eq!(
        *between.lock().expect("observation lock"),
        Some(None),
        "the purge erased the count row with the history, so nothing between the purge and the \
         recount can report a pre-purge total"
    );

    drop(store);
    let (store, _recorder) = open(&path);
    let surviving = counted_from_events(&store, context_id);
    assert_eq!(
        surviving.0, 1,
        "the concurrently appended event survives a forget of the other session"
    );
    assert_row_matches_events(&store, context_id, surviving.0);
    assert_status_reports(&path, &store, context_id, surviving.0);
}

/// A refused purge keeps whatever the count row says at that moment, including an increment a
/// concurrent append made after the forget read its plan: the row is only ever erased by the purge
/// itself, so a refusal cannot roll it back to a pre-append copy.
#[test]
fn a_refused_purge_keeps_the_increment_a_concurrent_append_made() {
    const CHILD: &str = "CG_COUNTS_REFUSED_PURGE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "a_refused_purge_keeps_the_increment_a_concurrent_append_made",
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
    let path = root.path().join("counts-race-refused-purge.db");
    let plugin = Arc::new(InterleaveOnPurge::default());
    let store = open_with_plugin(&path, plugin.clone());
    let context_id = context(&store, "refused purge");
    capture(
        &store,
        context_id,
        "file-1",
        vec![
            draft("s1", "a-1", 1, EventCoverage::Complete),
            draft("s1", "a-2", 2, EventCoverage::Opaque),
        ],
    );
    project(&store, context_id);
    assert_row_matches_events(&store, context_id, 2);

    let (release, release_rx) = channel();
    let (committed, committed_rx) = channel();
    *plugin.release.lock().expect("release lock") = Some(release);
    *plugin.committed.lock().expect("committed lock") = Some(committed_rx);
    plugin.armed.store(true, Ordering::SeqCst);
    plugin
        .release_at_count_row_delete
        .store(true, Ordering::SeqCst);
    let appending = append_on_release(
        store.clone(),
        context_id,
        "file-2",
        "s2",
        "b-1",
        release_rx,
        committed,
    );

    let fault = contextdb_engine::Database::open_memory();
    fault.arm_vector_durable_purge_failure_for_test();
    let refused = store.forget_event_history(ForgetSelection::Context { context_id });
    appending.join().expect("the concurrent append succeeded");
    assert!(
        fault.vector_durable_purge_failure_consumed_for_test(),
        "the erasure reached the engine's purge commit"
    );
    assert!(
        refused.is_err(),
        "the engine refuses this erasure: {refused:?}"
    );

    let surviving = counted_from_events(&store, context_id);
    assert_eq!(
        surviving.0, 3,
        "a refused purge erases nothing, so all three events remain"
    );
    assert_row_matches_events(&store, context_id, surviving.0);
    assert_status_reports(&path, &store, context_id, surviving.0);

    drop(store);
    let (store, _recorder) = open(&path);
    assert_row_matches_events(&store, context_id, surviving.0);
    assert_status_reports(&path, &store, context_id, surviving.0);
}
