//! Projection correctness under interruption and concurrent readers (intent "Projection
//! recovery" and the incremental-tail snapshot contract).
//!
//! The standalone runner releases its Store around rendering and embedding, so the embedder hooks
//! below act at real claim boundaries: one simulates a worker killed after a committed passage
//! batch, the other reads the committed projection between every claim of an incremental tail
//! replacement. The finalization test pauses inside the full rebuild's head-switch transaction
//! through the internal test seam.

#![cfg(feature = "test-support")]

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc::channel,
};

use context_graph::ConsumerReader;
use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::owner_control::{ReaderRelease, wait_for_reader_release_within};
use context_graph::test_support::{
    ProjectionFold, after_next_event_search_attempt_commit_before_readback_for_test,
    after_next_event_search_completion_commit_before_readback_for_test,
    after_next_event_search_membership_read_page_for_test,
    before_next_event_search_attempt_commit_for_test,
    before_next_event_search_completion_commit_for_test,
    before_next_event_search_failure_record_for_test,
    clear_event_search_discovery_observations_for_test,
    event_search_discovery_observations_for_test,
    fail_next_event_search_membership_capture_for_test,
    pause_next_event_search_finalization_for_test, pause_projection_unit_before_commit,
    set_event_search_projection_read_deadline_ms_for_test, set_projection_unit_cut,
    take_event_search_membership_captures_for_test, take_projection_unit_receipts,
};
use context_graph::{
    AppendEventBatch, CgError, ContextId, ContextSelector, CreateContext, DistanceMetric, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchBuildReceipt,
    EventSearchBuildRequest, EventSearchFailureDisposition, EventSearchFilter, EventSearchOrder,
    EventSearchProfile, EventSearchProjectionRunner, EventSearchRebuildRequest,
    EventSearchRunRequest, EventSearchSessionScope, EventSearchWorkAssessment,
    EventSearchWorkPolicy, EventSearchWorkSpec, EvidenceKind, EvidenceProducer, ForgetSelection,
    Result, Store, StoreConfig, event_history_reader_declaration,
};
use contextdb_core::{Value, Wallclock};
use serde_json::json;
use uuid::Uuid;

const CONNECTOR: &str = "event-search-correctness";
const SESSION: &str = "event-search-correctness-session";
const SPACE: &str = "event-search-correctness-768";
/// Short enough that the resume test can outwait a dead worker's lease.
const LEASE_MS: u64 = 1_000;
/// Long enough that a reader opening a Store between claims never expires the live claim.
const OBSERVED_LEASE_MS: u64 = 60_000;

#[test]
fn slow_projection_work_does_not_keep_a_source_read_statement_open() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("slow-projection.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    let events = (0..6)
        .map(|ordinal| {
            draft(
                &format!("event-{ordinal}"),
                ordinal,
                EventRenderRole::HumanMessage,
                &format!("{}-{ordinal}", "passage content ".repeat(5)),
            )
        })
        .collect();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "slow-projection-stream",
            events,
        ))
        .expect("append source events");
    drop(store);

    let (keep_open, receiver) = channel::<()>();
    let receiver = Mutex::new(receiver);
    let wait: Observer = Arc::new(move || {
        let _ = &keep_open;
        let _ = receiver
            .lock()
            .expect("wait receiver")
            .recv_timeout(std::time::Duration::from_millis(700));
    });
    let embedder = ControlledEmbedder::new();
    *embedder.embed_observer.lock().expect("observer lock") = Some(wait.clone());
    after_next_event_search_membership_read_page_for_test(move || wait());
    set_event_search_projection_read_deadline_ms_for_test(Some(500));
    let result = run(&config, context_id, &embedder, 2, OBSERVED_LEASE_MS);
    set_event_search_projection_read_deadline_ms_for_test(None);
    let receipt = result.expect("bounded source pages survive slow work and slow embeddings");
    assert_eq!(receipt.embedded_event_count, 6);
    assert!(
        embedder.embedded().len() >= 2,
        "more than one passage was embedded"
    );
}

#[test]
fn a_typed_active_time_refusal_retains_committed_passages_and_resumes() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("read-time-recovery.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "read-time-stream",
            (1..=8)
                .map(|ordinal| {
                    draft(
                        &format!("read-event-{ordinal}"),
                        ordinal,
                        EventRenderRole::HumanMessage,
                        &format!(
                            "unique source event {ordinal}: {}",
                            format!("word{ordinal} ").repeat(12)
                        ),
                    )
                })
                .collect(),
        ))
        .expect("append source events");
    drop(store);
    let reader = ConsumerReader::open(&config.db_path, event_history_reader_declaration()).unwrap();
    let selected = reader
        .select_event_search_work(context_id, EventSearchWorkPolicy::default())
        .expect("select captured source frontiers");
    drop(reader);
    let embedder = ControlledEmbedder::new();
    let request = run_request(&config, context_id, &embedder, profile(), 2, LEASE_MS);
    let mut source_batches = 0;
    let stopped =
        EventSearchProjectionRunner::catch_up_observed_with_selected_work_and_batch_limits(
            request.clone(),
            EventSearchWorkPolicy::default(),
            selected.clone(),
            || NonZeroUsize::MIN,
            |batch| {
                if batch.is_some() {
                    source_batches += 1;
                    if source_batches == 2 {
                        return Err(CgError::ReadActiveTimeExceeded { limit_ms: 5_000 });
                    }
                }
                Ok(context_graph::EventSearchBatchWriteLimit {
                    memory_limit_bytes: config.write_memory_limit_bytes,
                    through_sequence: None,
                })
            },
        );
    assert!(matches!(
        stopped.result,
        Err(CgError::ReadActiveTimeExceeded { limit_ms: 5_000 })
    ));
    assert_eq!(
        stopped.failure_disposition,
        EventSearchFailureDisposition::Recorded
    );
    let prefix = embedder.embedded();
    assert!(
        !prefix.is_empty(),
        "the refused run committed its first batch"
    );
    let failure = consumer_search(&config, context_id)
        .last_failure
        .expect("visible refusal");
    assert!(failure.contains("5_000") || failure.contains("5000"));
    assert!(
        failure.contains("--search wait"),
        "status supplies a retry instruction"
    );
    let resumed =
        EventSearchProjectionRunner::catch_up_observed_with_selected_work_and_batch_limits(
            request,
            EventSearchWorkPolicy::default(),
            selected,
            || NonZeroUsize::MIN,
            |_| {
                Ok(context_graph::EventSearchBatchWriteLimit {
                    memory_limit_bytes: config.write_memory_limit_bytes,
                    through_sequence: None,
                })
            },
        );
    let completed = resumed
        .result
        .expect("the failed read released its claim for continuation");
    assert_eq!(completed.embedded_event_count, 8);
    let all = embedder.embedded();
    for document in &prefix {
        assert_eq!(
            all.iter()
                .filter(|candidate| *candidate == document)
                .count(),
            prefix
                .iter()
                .filter(|candidate| *candidate == document)
                .count(),
            "committed prefix is never embedded twice"
        );
    }
    let store = open(&config);
    let rows = store
        .execute_sql_for_test(
            "SELECT id FROM event_search_passages WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("read published passages")
        .rows;
    let ids = rows
        .iter()
        .map(|row| format!("{:?}", row[0]))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        ids.len(),
        rows.len(),
        "continuation publishes no duplicate passage identities"
    );
}

type Observer = Arc<dyn Fn() + Send + Sync>;

/// Role-aware fixture tokenization (five model tokens plus two per scalar), with two controls:
/// a one-shot failure for any document containing a marker, and an observer run at every
/// document preparation.
struct ControlledEmbedder {
    fail_once_on: Mutex<Option<char>>,
    observer: Mutex<Option<Observer>>,
    /// Run at every prepared-document embed call.
    embed_observer: Mutex<Option<Observer>>,
    embedded_documents: Mutex<Vec<String>>,
}

fn content_vector(inference: &EventSearchInference) -> Vec<f32> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for token in inference.token_ids() {
        hash.update(token.to_le_bytes());
    }
    let hash = hash.finalize();
    (0..768)
        .map(|i| f32::from(hash[i % hash.len()]) / 255.0)
        .collect()
}

impl ControlledEmbedder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            fail_once_on: Mutex::new(None),
            observer: Mutex::new(None),
            embed_observer: Mutex::new(None),
            embedded_documents: Mutex::new(Vec::new()),
        })
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn embedded(&self) -> Vec<String> {
        self.embedded_documents
            .lock()
            .expect("embedded document log")
            .clone()
    }
}

fn fixture_embedding_text(text: &str) -> String {
    text.chars().flat_map(|scalar| [scalar, scalar]).collect()
}

impl Embedder for ControlledEmbedder {
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
            model_name: "correctness-fixture".into(),
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
        "event-search-correctness-v1".into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(256)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let observer = self.observer.lock().expect("observer lock").clone();
        if let Some(observer) = observer {
            observer();
        }
        let mut payload = vec![0; 5];
        let mut boundaries = vec![(0, 0); 5];
        for (start, scalar) in text.char_indices() {
            let end = start + scalar.len_utf8();
            payload.extend([scalar as u32, scalar as u32]);
            boundaries.extend([(start, end), (start, end)]);
        }
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
        let payload = text
            .chars()
            .flat_map(|scalar| [scalar as u32, scalar as u32])
            .collect::<Vec<_>>();
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.observe_forward_call();
        self.embed_one(inference)
    }

    fn embed_prepared_documents_hook(
        &self,
        inferences: &[EventSearchInference],
    ) -> Result<Vec<EmbeddingOutput>> {
        self.observe_forward_call();
        inferences
            .iter()
            .map(|inference| self.embed_one(inference))
            .collect()
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "correctness-fixture".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: content_vector(inference),
        })
    }
}

impl ControlledEmbedder {
    /// Run once at the start of every forward call, single or batched.
    fn observe_forward_call(&self) {
        let observer = self.embed_observer.lock().expect("observer lock").clone();
        if let Some(observer) = observer {
            observer();
        }
    }

    fn embed_one(&self, inference: &EventSearchInference) -> Result<EmbeddingOutput> {
        let text: String = inference
            .token_ids()
            .iter()
            .filter(|token| **token != 0)
            .map(|token| char::from_u32(*token).expect("fixture token is Unicode"))
            .collect();
        let mut fail = self.fail_once_on.lock().expect("failure control lock");
        if fail.is_some_and(|marker| text.contains(marker)) {
            *fail = None;
            return Err(CgError::Engine(
                "fixture worker killed during embedding".into(),
            ));
        }
        drop(fail);
        self.embedded_documents
            .lock()
            .expect("embedded document log")
            .push(text);
        Ok(EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "correctness-fixture".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: content_vector(inference),
        })
    }
}

fn config(path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        extra_embedding_spaces: vec![ControlledEmbedder::new().embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "correctness-v1".into(),
        anchor_declaration_version: "correctness-v1".into(),
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
        author: "Renée".into(),
        kind: "message".into(),
        render_role: role,
        interaction_ref: Some("correctness-pair".into()),
        text: Some(text.into()),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn profile() -> EventSearchProfile {
    EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(177).expect("nonzero target"),
        hard_passage_tokens: NonZeroU32::new(256).expect("nonzero hard limit"),
        ordinary_overlap_events: NonZeroU32::new(1).expect("nonzero overlap"),
        oversized_event_overlap_tokens: NonZeroU32::new(4).expect("nonzero oversized overlap"),
        ..EventSearchProfile::default()
    }
}

fn run(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<ControlledEmbedder>,
    max_events_per_batch: u32,
    lease_ms: u64,
) -> Result<EventSearchBuildReceipt> {
    run_with_profile(
        config,
        context_id,
        embedder,
        profile(),
        max_events_per_batch,
        lease_ms,
    )
}

fn run_with_profile(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<ControlledEmbedder>,
    profile: EventSearchProfile,
    max_events_per_batch: u32,
    lease_ms: u64,
) -> Result<EventSearchBuildReceipt> {
    EventSearchProjectionRunner::catch_up(run_request(
        config,
        context_id,
        embedder,
        profile,
        max_events_per_batch,
        lease_ms,
    ))
}

fn run_request(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<ControlledEmbedder>,
    profile: EventSearchProfile,
    max_events_per_batch: u32,
    lease_ms: u64,
) -> EventSearchRunRequest {
    run_request_with_workers(
        config,
        context_id,
        embedder,
        profile,
        max_events_per_batch,
        lease_ms,
        NonZeroUsize::MIN,
    )
}

fn run_request_with_workers(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<ControlledEmbedder>,
    profile: EventSearchProfile,
    max_events_per_batch: u32,
    lease_ms: u64,
    embedding_workers: NonZeroUsize,
) -> EventSearchRunRequest {
    EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile,
        embedding_workers,
        max_events_per_batch: NonZeroU32::new(max_events_per_batch).expect("nonzero batch"),
        max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
        lease_duration_ms: NonZeroU64::new(lease_ms).expect("nonzero lease"),
    }
}

fn run_until_projection_completes(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<ControlledEmbedder>,
    max_events_per_batch: u32,
) -> EventSearchBuildReceipt {
    for _ in 0..64 {
        match run(
            config,
            context_id,
            embedder,
            max_events_per_batch,
            OBSERVED_LEASE_MS,
        ) {
            Ok(receipt) if receipt.completed_at.is_some() => return receipt,
            Ok(_) | Err(CgError::ReadActiveTimeExceeded { .. }) => {
                // The 5 s active-time ceiling ends a unit run. A fresh attempt resumes the
                // persisted frontier; it never changes the unit allowance.
            }
            Err(error) => panic!("projection completion failed: {error:?}"),
        }
    }
    panic!("projection did not finish after repeated bounded-unit resumptions");
}

fn commit_later_begun_audit_transaction(
    store: &Store,
    context_id: ContextId,
) -> context_graph::TxId {
    let db = store.sync_database();
    let tx = db.begin().expect("begin intervening transaction");
    let id = uuid::Uuid::now_v7();
    db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, reason, target_kind, target_id, context_id, outcome_recorded_at, payload) VALUES ($id, $txid, $operation, $actor, $reason, $target_kind, $target_id, $context_id, $outcome_recorded_at, $payload)",
        &HashMap::from([
            ("id".into(), Value::Uuid(id)),
            ("txid".into(), Value::TxId(tx)),
            ("operation".into(), Value::Text("projection_interleave".into())),
            ("actor".into(), Value::Text("test".into())),
            ("reason".into(), Value::Null),
            ("target_kind".into(), Value::Text("context".into())),
            ("target_id".into(), Value::Uuid(context_id.as_uuid())),
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("outcome_recorded_at".into(), Value::Null),
            ("payload".into(), Value::Json(json!(null))),
        ]),
    )
    .expect("stage intervening audit row");
    db.commit(tx).expect("commit intervening transaction");
    let committed = db
        .execute(
            "SELECT txid FROM audit_log WHERE id = $id",
            &HashMap::from([("id".into(), Value::Uuid(id))]),
        )
        .expect("read intervening transaction");
    match committed.rows.as_slice() {
        [row] => match row.as_slice() {
            [Value::TxId(committed)] => *committed,
            other => panic!("intervening row has no committed TXID: {other:?}"),
        },
        other => panic!("intervening transaction row is not unique: {other:?}"),
    }
}

fn attempt_owner_row(config: &StoreConfig, context_id: ContextId) -> Vec<Value> {
    let mut rows = open(config)
        .execute_sql_for_test(
            "SELECT id, started_at, last_attempt_at, last_attempt_token, completed_at, sanitized_failure FROM event_search_generations WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("read the competing owner's committed identity")
        .rows;
    assert_eq!(rows.len(), 1, "one generation owns this fixture Context");
    rows.remove(0)
}

#[test]
fn projection_attempts_and_completion_keep_their_committed_identity_when_commits_interleave() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("commit-identity.db"));
    let embedder = ControlledEmbedder::new();
    let store = open(&config);
    let context_id = create_context_named(&store, "new attempt commit identity");
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "new-attempt-stream",
            vec![draft(
                "new-attempt",
                1,
                EventRenderRole::HumanMessage,
                "new attempt survives an interleaved commit α",
            )],
        ))
        .expect("append new-attempt fixture");
    drop(store);

    let interleaved = Rc::new(RefCell::new(Vec::new()));
    let attempt_interleaved = Rc::clone(&interleaved);
    before_next_event_search_attempt_commit_for_test(move |store| {
        attempt_interleaved
            .borrow_mut()
            .push(commit_later_begun_audit_transaction(store, context_id));
    });
    let completion_interleaved = Rc::clone(&interleaved);
    before_next_event_search_completion_commit_for_test(move |store| {
        completion_interleaved
            .borrow_mut()
            .push(commit_later_begun_audit_transaction(store, context_id));
    });
    let receipt = run(&config, context_id, &embedder, 8, LEASE_MS)
        .expect("a new attempt still claims and completes after the later transaction commits");
    let row = attempt_rows(&config, context_id).remove(0);
    let (Value::TxId(started), Value::TxId(attempt)) = (&row[1], &row[2]) else {
        panic!("new generation exposes typed timestamps: {row:?}")
    };
    let completed = receipt.completed_at.expect("completion timestamp");
    assert_eq!(*started, receipt.started_at);
    assert_eq!(*attempt, receipt.started_at);
    let interleaved = interleaved.borrow();
    assert_eq!(interleaved.len(), 2);
    assert!(*started > interleaved[0]);
    assert!(completed > interleaved[1]);
    let stored = open(&config)
        .execute_sql_for_test(
            "SELECT completed_at FROM event_search_generations WHERE id = $id",
            &HashMap::from([("id".into(), Value::Uuid(receipt.generation_id.as_uuid()))]),
        )
        .expect("read committed completion");
    assert_eq!(stored.rows, vec![vec![Value::TxId(completed)]]);
    drop(interleaved);

    let store = open(&config);
    let failed_context = create_context_named(&store, "failed and resumed commit identity");
    store
        .append_event_batch(AppendEventBatch::live(
            failed_context,
            declaration(),
            "failed-attempt-stream",
            vec![draft(
                "failed-attempt",
                1,
                EventRenderRole::HumanMessage,
                "specific interleaved failure β",
            )],
        ))
        .expect("append failed-attempt fixture");
    drop(store);

    before_next_event_search_attempt_commit_for_test(move |store| {
        commit_later_begun_audit_transaction(store, failed_context);
    });
    *embedder.fail_once_on.lock().expect("failure control") = Some('β');
    let failed = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        failed_context,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert!(
        matches!(&failed.result, Err(CgError::Engine(reason)) if reason.contains("fixture worker killed")),
        "the forced failure remains the specific run result: {:?}",
        failed.result
    );
    assert_eq!(
        failed.failure_disposition,
        EventSearchFailureDisposition::Recorded
    );
    assert!(
        consumer_search(&config, failed_context)
            .last_failure
            .as_deref()
            .is_some_and(|reason| reason.contains("fixture worker killed")),
        "the matching attempt stores the specific reason"
    );

    let resumed_interleaved = Rc::new(RefCell::new(Vec::new()));
    let resume_attempt = Rc::clone(&resumed_interleaved);
    before_next_event_search_attempt_commit_for_test(move |store| {
        resume_attempt
            .borrow_mut()
            .push(commit_later_begun_audit_transaction(store, failed_context));
    });
    let resume_completion = Rc::clone(&resumed_interleaved);
    before_next_event_search_completion_commit_for_test(move |store| {
        resume_completion
            .borrow_mut()
            .push(commit_later_begun_audit_transaction(store, failed_context));
    });
    let resumed = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        failed_context,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert_eq!(
        resumed.failure_disposition,
        EventSearchFailureDisposition::NoFailure
    );
    let resumed = resumed
        .result
        .expect("the resumed attempt still claims and completes");
    let rows = attempt_rows(&config, failed_context);
    let row = rows.first().expect("one resumed generation");
    let (Value::TxId(started), Value::TxId(attempt)) = (&row[1], &row[2]) else {
        panic!("resumed generation exposes typed timestamps: {row:?}")
    };
    let completed = resumed.completed_at.expect("resumed completion timestamp");
    assert_eq!(*started, resumed.started_at);
    assert!(*attempt > *started);
    let resumed_interleaved = resumed_interleaved.borrow();
    assert_eq!(resumed_interleaved.len(), 2);
    assert!(*attempt > resumed_interleaved[0]);
    assert!(completed > resumed_interleaved[1]);
    assert_eq!(consumer_search(&config, failed_context).last_failure, None);
}

#[test]
fn post_commit_readback_refuses_a_competing_owner_and_a_later_run_completes() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("post-commit-takeover.db"));
    let embedder = ControlledEmbedder::new();

    // New-generation attempt: a normal competing refusal commits after the insertion commit and
    // before this caller can read back its rewritten timestamp.
    let store = open(&config);
    let capture_context = create_context_named(&store, "capture post-commit takeover");
    store
        .append_event_batch(AppendEventBatch::live(
            capture_context,
            declaration(),
            "capture-post-commit",
            vec![draft(
                "capture-post-commit",
                1,
                EventRenderRole::HumanMessage,
                "capture competitor takes the marker",
            )],
        ))
        .expect("append capture takeover fixture");
    drop(store);
    let capture_owner = Rc::new(RefCell::new(None));
    let capture_owner_hook = Rc::clone(&capture_owner);
    let capture_config = config.clone();
    let capture_embedder = Arc::clone(&embedder);
    after_next_event_search_attempt_commit_before_readback_for_test(move |_| {
        EventSearchProjectionRunner::record_refusal(
            run_request(
                &capture_config,
                capture_context,
                &capture_embedder,
                profile(),
                8,
                LEASE_MS,
            ),
            "capture competitor owns the committed marker",
        )
        .expect("the competing capture attempt commits");
        *capture_owner_hook.borrow_mut() =
            Some(attempt_owner_row(&capture_config, capture_context));
    });
    let capture = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        capture_context,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert!(
        matches!(&capture.result, Err(CgError::Engine(reason)) if reason.contains("superseded before committed timestamp readback")),
        "the stale capture caller refuses its result: {:?}",
        capture.result
    );
    assert_eq!(
        capture.failure_disposition,
        EventSearchFailureDisposition::Stale
    );
    let capture_owner = capture_owner
        .borrow_mut()
        .take()
        .expect("capture competitor committed an identity");
    assert_eq!(attempt_owner_row(&config, capture_context), capture_owner);
    assert!(matches!(
        capture_owner.as_slice(),
        [Value::Uuid(_), Value::TxId(_), Value::TxId(_), Value::Uuid(_), Value::Null, Value::Text(reason)]
            if reason == "capture competitor owns the committed marker"
    ));
    run(&config, capture_context, &embedder, 8, LEASE_MS)
        .expect("a normal later run completes after the capture takeover");

    // Existing-generation resumption takes the other attempt path and receives the same refusal.
    let store = open(&config);
    let resume_context = create_context_named(&store, "resume post-commit takeover");
    store
        .append_event_batch(AppendEventBatch::live(
            resume_context,
            declaration(),
            "resume-post-commit",
            vec![draft(
                "resume-post-commit",
                1,
                EventRenderRole::HumanMessage,
                "resume competitor takes the marker",
            )],
        ))
        .expect("append resumption takeover fixture");
    drop(store);
    run(&config, resume_context, &embedder, 8, LEASE_MS)
        .expect("make the generation available for resumption");
    let resume_owner = Rc::new(RefCell::new(None));
    let resume_owner_hook = Rc::clone(&resume_owner);
    let resume_config = config.clone();
    let resume_embedder = Arc::clone(&embedder);
    after_next_event_search_attempt_commit_before_readback_for_test(move |_| {
        EventSearchProjectionRunner::record_refusal(
            run_request(
                &resume_config,
                resume_context,
                &resume_embedder,
                profile(),
                8,
                LEASE_MS,
            ),
            "resumption competitor owns the committed marker",
        )
        .expect("the competing resumption attempt commits");
        *resume_owner_hook.borrow_mut() = Some(attempt_owner_row(&resume_config, resume_context));
    });
    let resumed = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        resume_context,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert!(
        matches!(&resumed.result, Err(CgError::Engine(reason)) if reason.contains("superseded before committed timestamp readback")),
        "the stale resumption caller refuses its result: {:?}",
        resumed.result
    );
    assert_eq!(
        resumed.failure_disposition,
        EventSearchFailureDisposition::Stale
    );
    let resume_owner = resume_owner
        .borrow_mut()
        .take()
        .expect("resumption competitor committed an identity");
    assert_eq!(attempt_owner_row(&config, resume_context), resume_owner);
    assert!(matches!(
        resume_owner.as_slice(),
        [Value::Uuid(_), Value::TxId(_), Value::TxId(_), Value::Uuid(_), Value::TxId(_), Value::Text(reason)]
            if reason == "resumption competitor owns the committed marker"
    ));
    run(&config, resume_context, &embedder, 8, LEASE_MS)
        .expect("a normal later run completes after the resumption takeover");

    // Completion owns a live lease. The hook releases only that lease, then uses the ordinary
    // refusal path to create the competing attempt after completion has committed.
    let store = open(&config);
    let completion_context = create_context_named(&store, "completion post-commit takeover");
    store
        .append_event_batch(AppendEventBatch::live(
            completion_context,
            declaration(),
            "completion-post-commit",
            vec![draft(
                "completion-post-commit",
                1,
                EventRenderRole::HumanMessage,
                "completion competitor takes the marker",
            )],
        ))
        .expect("append completion takeover fixture");
    drop(store);
    let completion_owner = Rc::new(RefCell::new(None));
    let completion_owner_hook = Rc::clone(&completion_owner);
    let completion_config = config.clone();
    let completion_embedder = Arc::clone(&embedder);
    after_next_event_search_completion_commit_before_readback_for_test(move |store| {
        store
            .execute_sql_for_test(
                "DELETE FROM event_search_leases WHERE context_id = $context_id",
                &HashMap::from([(
                    "context_id".into(),
                    Value::Uuid(completion_context.as_uuid()),
                )]),
            )
            .expect("release the completed caller's lease for the competing normal attempt");
        EventSearchProjectionRunner::record_refusal(
            run_request(
                &completion_config,
                completion_context,
                &completion_embedder,
                profile(),
                8,
                LEASE_MS,
            ),
            "completion competitor owns the committed marker",
        )
        .expect("the competing completion attempt commits");
        *completion_owner_hook.borrow_mut() =
            Some(attempt_owner_row(&completion_config, completion_context));
    });
    let completion = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        completion_context,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert!(
        matches!(&completion.result, Err(CgError::Engine(reason)) if reason.contains("superseded before completion timestamp readback")),
        "the stale completion caller refuses its receipt: {:?}",
        completion.result
    );
    assert_eq!(
        completion.failure_disposition,
        EventSearchFailureDisposition::Stale
    );
    let completion_owner = completion_owner
        .borrow_mut()
        .take()
        .expect("completion competitor committed an identity");
    assert_eq!(
        attempt_owner_row(&config, completion_context),
        completion_owner
    );
    let [
        Value::Uuid(_),
        Value::TxId(_),
        Value::TxId(competitor_attempt),
        Value::Uuid(_),
        Value::TxId(completed),
        Value::Text(reason),
    ] = completion_owner.as_slice()
    else {
        panic!("completion competitor preserves typed timestamps and token: {completion_owner:?}");
    };
    assert_eq!(reason, "completion competitor owns the committed marker");
    assert!(
        competitor_attempt > completed,
        "beginning the competing attempt preserves the already committed completion timestamp"
    );
    let later = run(&config, completion_context, &embedder, 8, LEASE_MS)
        .expect("a normal later run completes after the completion-readback takeover");
    assert_eq!(later.completed_at, Some(*completed));
}

#[derive(Debug, PartialEq)]
struct CommittedProjection {
    passages: Vec<Vec<Value>>,
    sources: Vec<Vec<Value>>,
    coverage: (u64, u64, u64, context_graph::EventHistorySearchState),
}

fn committed_projection(store: &Store, context_id: ContextId) -> CommittedProjection {
    let passages = store
        .execute_sql_for_test(
            "SELECT passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension FROM event_search_passages WHERE context_id = $context_id AND is_current = $current ORDER BY passage_ordinal ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("current".into(), Value::Bool(true)),
            ]),
        )
        .expect("inspect committed passage order, content, and vectors")
        .rows;
    let status = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("inspect committed coverage")
        .contexts
        .remove(0);
    let passage_ids = store.execute_sql_for_test(
        "SELECT id, passage_ordinal FROM event_search_passages WHERE context_id = $context_id AND is_current = $current ORDER BY passage_ordinal ASC",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid())), ("current".into(), Value::Bool(true))]),
    ).unwrap();
    let mut sources = Vec::new();
    for passage in passage_ids.rows {
        let links = store.execute_sql_for_test(
            "SELECT event_id, source_order, event_start_byte, event_end_byte, passage_start_byte, passage_end_byte FROM event_search_passage_sources WHERE passage_id = $passage_id ORDER BY source_order ASC",
            &HashMap::from([("passage_id".into(), passage[0].clone())]),
        ).unwrap();
        for mut link in links.rows {
            let event = store
                .execute_sql_for_test(
                    "SELECT seq FROM events WHERE id = $event_id",
                    &HashMap::from([("event_id".into(), link[0].clone())]),
                )
                .unwrap();
            link[0] = event.rows[0][0].clone();
            link.insert(0, passage[1].clone());
            sources.push(link);
        }
    }
    CommittedProjection {
        passages,
        sources,
        coverage: (
            status.captured_events,
            status.search.covered_events,
            status.search.pending_events,
            status.search.state,
        ),
    }
}

fn open(config: &StoreConfig) -> Store {
    Store::open(config.clone()).expect("open Store")
}

fn create_context(store: &Store) -> ContextId {
    create_context_named(store, "event-search correctness")
}

fn create_context_named(store: &Store, name: &str) -> ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id
}

/// Every current passage in the Context as (passage id, rendered text), in passage order.
fn current_passages(store: &Store, context_id: ContextId) -> Vec<(uuid::Uuid, String)> {
    let result = store
        .execute_sql_for_test(
            "SELECT id, rendered_text, passage_ordinal FROM event_search_passages WHERE context_id = $context_id AND is_current = $current ORDER BY passage_ordinal ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("current".into(), Value::Bool(true)),
            ]),
        )
        .expect("inspect current passages");
    result
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Uuid(id), Value::Text(text)) => (*id, text.clone()),
            other => panic!("unexpected passage row {other:?}"),
        })
        .collect()
}

fn searchable_current_passages(store: &Store, context_id: ContextId) -> Vec<(uuid::Uuid, String)> {
    let result = store
        .execute_sql_for_test(
            "SELECT id, rendered_text, passage_ordinal FROM event_search_passages WHERE context_id = $context_id AND is_current = $current AND is_searchable = $searchable ORDER BY passage_ordinal ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("current".into(), Value::Bool(true)),
                ("searchable".into(), Value::Bool(true)),
            ]),
        )
        .expect("inspect searchable current passages");
    result
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Uuid(id), Value::Text(text)) => (*id, text.clone()),
            other => panic!("unexpected searchable passage row {other:?}"),
        })
        .collect()
}

fn all_passage_count(store: &Store, context_id: ContextId) -> usize {
    store
        .execute_sql_for_test(
            "SELECT id FROM event_search_passages WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect every passage row")
        .rows
        .len()
}

/// Force one superseded version after projection's own maintenance has completed, so the
/// following explicit cycle proves reclaim instead of racing the projection's cycle.
fn supersede_passage_version(store: &Store, passage_id: context_graph::EventSearchPassageId) {
    let params = HashMap::from([("id".into(), Value::Uuid(passage_id.as_uuid()))]);
    store
        .execute_sql_for_test(
            "UPDATE event_search_passages SET is_searchable = false WHERE id = $id",
            &params,
        )
        .expect("supersede the searchable passage version");
    store
        .execute_sql_for_test(
            "UPDATE event_search_passages SET is_searchable = true WHERE id = $id",
            &params,
        )
        .expect("restore the current searchable value");
}

#[test]
fn old_conversations_remain_searchable_after_superseded_passage_reclaim() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("old-conversations-reclaim.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 3);
    drop(store);
    let embedder = ControlledEmbedder::new();
    run(&config, context_id, &embedder, 1, OBSERVED_LEASE_MS)
        .expect("publish the earlier conversations");
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open searchable store");
    let find = |session: &str| {
        store
            .find_event_history(
                ContextSelector::one(context_id),
                "session marker",
                EventSearchFilter {
                    session_ref: Some(session.into()),
                    ..EventSearchFilter::default()
                },
            )
            .expect("find conversation")
            .hits
    };
    let sessions = ["scope-session-00", "scope-session-01", "scope-session-02"];
    let before = sessions.map(&find);
    assert!(before.iter().all(|hits| hits.len() == 1));
    append_timestamped_session(&store, context_id, 3);
    drop(store);
    run(&config, context_id, &embedder, 1, OBSERVED_LEASE_MS)
        .expect("publish another conversation in the same Context");
    let store = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen searchable store");
    for hit in before.iter().flat_map(|hits| hits.iter()) {
        supersede_passage_version(&store, hit.passage_id);
    }
    let report = store
        .sync_database()
        .run_maintenance_cycle()
        .expect("reclaim superseded passage versions");
    eprintln!("old conversations reclaim: {:?}", report.currency);
    assert!(report.currency.pruned_versions >= 3, "{report:?}");
    assert!(
        report
            .currency
            .compacted_tables
            .iter()
            .any(|table| table == "event_search_passages"),
        "the projection table supplied the reclaimed versions: {report:?}"
    );
    for (session, expected) in sessions.iter().zip(before) {
        let found = store
            .find_event_history(
                ContextSelector::one(context_id),
                "session marker",
                EventSearchFilter {
                    session_ref: Some((*session).into()),
                    ..EventSearchFilter::default()
                },
            )
            .expect("find earlier conversation after reclaim");
        assert_eq!(found.hits, expected, "{session} changed after reclaim");
    }
}

#[test]
fn passage_handles_and_sources_resolve_after_superseded_passage_reclaim() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("passage-handle-reclaim.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "tail-stream",
            vec![draft(
                "first",
                1,
                EventRenderRole::HumanMessage,
                "rollback deployment",
            )],
        ))
        .expect("append first event");
    drop(store);
    let embedder = ControlledEmbedder::new();
    run(&config, context_id, &embedder, 1, OBSERVED_LEASE_MS).expect("publish open tail");
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open searchable store");
    let hit = store
        .find_event_history(
            ContextSelector::one(context_id),
            "rollback deployment",
            EventSearchFilter::default(),
        )
        .expect("find open tail")
        .hits
        .remove(0);
    let before = store
        .get_event_search_passage(ContextSelector::one(context_id), hit.passage_id)
        .expect("resolve handle before retirement")
        .expect("passage exists");
    assert!(!before.sources.is_empty());
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "tail-stream",
            vec![draft(
                "second",
                2,
                EventRenderRole::HumanMessage,
                "deployment restored",
            )],
        ))
        .expect("append replacement event");
    drop(store);
    run(&config, context_id, &embedder, 1, OBSERVED_LEASE_MS).expect("retire the old open tail");
    let store = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen searchable store");
    let retired = store
        .get_event_search_passage(ContextSelector::one(context_id), hit.passage_id)
        .expect("resolve retired passage")
        .expect("retired passage still exists");
    assert!(!retired.is_current, "the old tail was retired");
    supersede_passage_version(&store, hit.passage_id);
    let report = store
        .sync_database()
        .run_maintenance_cycle()
        .expect("reclaim superseded retired passage versions");
    eprintln!("passage handle reclaim: {:?}", report.currency);
    assert!(report.currency.pruned_versions > 0, "{report:?}");
    assert!(
        report
            .currency
            .compacted_tables
            .iter()
            .any(|table| table == "event_search_passages"),
        "the retired passage supplied a reclaimed version: {report:?}"
    );
    let after = store
        .get_event_search_passage(ContextSelector::one(context_id), hit.passage_id)
        .expect("resolve handle after reclaim")
        .expect("current retired row survives");
    assert_eq!(after, retired);
    assert_eq!(after.embedding_id, before.embedding_id);
    assert_eq!(after.sources, before.sources);
}

/// The persisted projection cursors are the proof that a claimed source batch committed; status
/// fields are a separate public rendering of that state.
fn committed_cursor_sequences(
    store: &Store,
    context_id: ContextId,
    source_instance_ref: &str,
) -> Vec<u64> {
    let result = store
        .execute_sql_for_test(
            "SELECT committed_sequence FROM event_search_cursors WHERE context_id = $context_id AND source_instance_ref = $source ORDER BY committed_sequence ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("source".into(), Value::Text(source_instance_ref.into())),
            ]),
        )
        .expect("inspect committed event-search cursors");
    result
        .rows
        .iter()
        .map(|row| match row.as_slice() {
            [Value::Int64(sequence)] => {
                u64::try_from(*sequence).expect("committed cursor sequence is non-negative")
            }
            other => panic!("unexpected committed cursor row {other:?}"),
        })
        .collect()
}

fn generation_count(store: &Store, context_id: ContextId) -> usize {
    store
        .execute_sql_for_test(
            "SELECT id FROM event_search_generations WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect search generations")
        .rows
        .len()
}

fn work_spec(profile: EventSearchProfile) -> EventSearchWorkSpec {
    EventSearchWorkSpec {
        profile,
        embedding_space_id: SPACE.into(),
        encoding_profile_id: "event-search-correctness-v1".into(),
    }
}

fn one_work(
    source: &impl EventSearchWorkSource,
    context_id: ContextId,
    spec: EventSearchWorkSpec,
) -> EventSearchWorkAssessment {
    source
        .event_search_work(ContextSelector::one(context_id), &spec)
        .expect("requested projection assessment")
        .pop()
        .expect("one selected Context has one assessment")
}

trait EventSearchWorkSource {
    fn event_search_work(
        &self,
        selector: ContextSelector,
        spec: &EventSearchWorkSpec,
    ) -> Result<Vec<EventSearchWorkAssessment>>;
}

impl EventSearchWorkSource for Store {
    fn event_search_work(
        &self,
        selector: ContextSelector,
        spec: &EventSearchWorkSpec,
    ) -> Result<Vec<EventSearchWorkAssessment>> {
        Store::event_search_work(self, selector, spec)
    }
}

impl EventSearchWorkSource for ConsumerReader {
    fn event_search_work(
        &self,
        selector: ContextSelector,
        spec: &EventSearchWorkSpec,
    ) -> Result<Vec<EventSearchWorkAssessment>> {
        ConsumerReader::event_search_work(self, selector, spec)
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

/// Import and the detached worker must ask the same complete identity question as the projection
/// runner.  This probes the public read-only door directly, including a captured Context before
/// any generation exists, a matching head, and each caller-provided identity component.
#[test]
fn requested_projection_assessment_uses_full_identity_on_store_and_actual_consumer_reader_without_artifacts()
 {
    let now = Wallclock::now().0;
    let _clock = Wallclock::test_clock_guard(move || now);
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("requested-work.db"));
    let store = open(&config);
    let empty_context = create_context_named(&store, "requested work empty");
    let context_id = create_context_named(&store, "requested work captured");
    for (key, ordinal, text) in [
        ("one", 1, "first requested projection event"),
        ("two", 2, "second requested projection event"),
    ] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "requested-work-stream",
                vec![draft(key, ordinal, EventRenderRole::HumanMessage, text)],
            ))
            .expect("capture requested projection fixture");
    }

    let empty = one_work(&store, empty_context, work_spec(profile()));
    assert_eq!(empty.captured_events, 0);
    assert_eq!(empty.covered_events, 0);
    assert_eq!(empty.pending_events, 0);
    assert!(!empty.work_required, "an empty Context is already complete");
    assert_eq!(empty.generation_id, None);

    let captured = one_work(&store, context_id, work_spec(profile()));
    assert_eq!(captured.captured_events, 2);
    assert_eq!(captured.covered_events, 0);
    assert_eq!(captured.pending_events, 2);
    assert!(
        captured.work_required,
        "captured events require this projection"
    );
    assert_eq!(captured.generation_id, None);
    drop(store);

    let embedder = ControlledEmbedder::new();
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("build the matching head");
    let store = open(&config);
    let matching = one_work(&store, context_id, work_spec(profile()));
    assert_eq!(matching.captured_events, 2);
    assert_eq!(matching.covered_events, 2);
    assert_eq!(matching.pending_events, 0);
    assert!(
        !matching.work_required,
        "the matching active head is a no-op"
    );
    assert!(matching.generation_id.is_some());

    let changed_profile = EventSearchProfile {
        ordinary_overlap_events: NonZeroU32::new(2).expect("nonzero changed overlap"),
        ..profile()
    };
    let stale_specs = [
        ("renderer profile", work_spec(changed_profile)),
        (
            "embedding space",
            EventSearchWorkSpec {
                embedding_space_id: "other-event-search-space".into(),
                ..work_spec(profile())
            },
        ),
        (
            "encoding profile",
            EventSearchWorkSpec {
                encoding_profile_id: "other-event-search-encoding".into(),
                ..work_spec(profile())
            },
        ),
    ];
    for (name, spec) in &stale_specs {
        let stale = one_work(&store, context_id, spec.clone());
        assert_eq!(stale.captured_events, 2, "{name} preserves captured count");
        assert_eq!(
            stale.covered_events, 0,
            "{name} excludes wrong-identity coverage"
        );
        assert_eq!(stale.pending_events, 2, "{name} makes every event pending");
        assert!(
            stale.work_required,
            "{name} requires a replacement projection"
        );
    }
    drop(store);

    let before_consumer_read = directory_image(root.path());
    let reader = ConsumerReader::open(&config.db_path, event_history_reader_declaration())
        .expect("actual consumer reader opens the stopped store");
    let from_consumer = one_work(&reader, context_id, work_spec(profile()));
    assert_eq!(from_consumer.context_id, matching.context_id);
    assert_eq!(from_consumer.captured_events, matching.captured_events);
    assert_eq!(from_consumer.generation_id, matching.generation_id);
    assert_eq!(from_consumer.covered_events, matching.covered_events);
    assert_eq!(from_consumer.pending_events, matching.pending_events);
    assert_eq!(from_consumer.work_required, matching.work_required);
    assert_eq!(
        from_consumer.live_projection_lease,
        matching.live_projection_lease
    );
    assert_eq!(
        one_work(&reader, empty_context, work_spec(profile())),
        empty
    );
    for (name, spec) in &stale_specs {
        let stale = one_work(&reader, context_id, spec.clone());
        assert_eq!(
            (stale.covered_events, stale.pending_events),
            (0, 2),
            "consumer excludes stale {name}"
        );
        assert!(stale.work_required);
    }
    drop(reader);
    assert_eq!(
        directory_image(root.path()),
        before_consumer_read,
        "the actual ConsumerReader assessment creates no schema, model, or projection artifacts"
    );

    // Persisted historical renderer identities are library-owned inputs, absent from WorkSpec.
    // Change one old-head field at a time; the assessment and actual runner must both reject it.
    for column in ["renderer_id", "renderer_version"] {
        let store = open(&config);
        let old_head = one_work(&store, context_id, work_spec(profile()))
            .generation_id
            .unwrap();
        store
            .execute_sql_for_test(
                &format!("UPDATE event_search_generations SET {column} = $stale WHERE id = $id"),
                &HashMap::from([
                    ("id".into(), Value::Uuid(old_head.as_uuid())),
                    ("stale".into(), Value::Text("historical-renderer".into())),
                ]),
            )
            .unwrap();
        let stale = one_work(&store, context_id, work_spec(profile()));
        assert_eq!(
            (stale.covered_events, stale.pending_events),
            (0, 2),
            "stale persisted {column} cannot count as requested coverage"
        );
        assert!(stale.work_required);
        assert_eq!(stale.generation_id, None);
        drop(store);
        let before = directory_image(root.path());
        let reader =
            ConsumerReader::open(&config.db_path, event_history_reader_declaration()).unwrap();
        assert_eq!(one_work(&reader, context_id, work_spec(profile())), stale);
        drop(reader);
        assert_eq!(
            directory_image(root.path()),
            before,
            "stale {column} assessment is read-only"
        );
        let rebuilt = run(&config, context_id, &embedder, 8, LEASE_MS).unwrap();
        assert_ne!(
            rebuilt.generation_id, old_head,
            "runner replaces the identity rejected by assessment"
        );
    }

    // Adding a captured connector changes the current ordinary interpretation snapshot through
    // the public append API. Every caller-supplied identity component stays unchanged.
    let store = open(&config);
    let old_head = one_work(&store, context_id, work_spec(profile()))
        .generation_id
        .unwrap();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            EventAdapterDeclaration {
                connector_id: "second-captured-connector".into(),
                ..declaration()
            },
            "second-connector-stream",
            vec![draft(
                "new-connector",
                1,
                EventRenderRole::HumanMessage,
                "new ordinary interpretation binding",
            )],
        ))
        .unwrap();
    let changed_snapshot = one_work(&store, context_id, work_spec(profile()));
    assert_eq!(
        (
            changed_snapshot.covered_events,
            changed_snapshot.pending_events
        ),
        (0, 3)
    );
    assert!(changed_snapshot.work_required);
    assert_eq!(changed_snapshot.generation_id, None);
    drop(store);
    let before = directory_image(root.path());
    let reader = ConsumerReader::open(&config.db_path, event_history_reader_declaration()).unwrap();
    assert_eq!(
        one_work(&reader, context_id, work_spec(profile())),
        changed_snapshot
    );
    drop(reader);
    assert_eq!(directory_image(root.path()), before);
    let rebuilt = run(&config, context_id, &embedder, 8, LEASE_MS).unwrap();
    assert_ne!(rebuilt.generation_id, old_head);
    assert_eq!(rebuilt.source_watermark.event_count, 3);
    assert!(!one_work(&open(&config), context_id, work_spec(profile())).work_required);
}

/// A replacement that has covered every event is still work until its head activation commits.
/// The existing cfg-only finalization rendezvous makes this observable without inventing another
/// worker harness.
#[test]
fn requested_projection_assessment_keeps_a_finishing_unactivated_replacement_pending() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("unactivated-requested-work.db"));
    let embedder = ControlledEmbedder::new();
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open Store");
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "unactivated-requested-work-stream",
            vec![draft(
                "one",
                1,
                EventRenderRole::HumanMessage,
                "head activation is pending",
            )],
        ))
        .expect("capture fixture event");
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("build existing ready head");

    let replacement_profile = EventSearchProfile {
        ordinary_overlap_events: NonZeroU32::new(2).expect("nonzero replacement overlap"),
        ..profile()
    };
    let expected = work_spec(replacement_profile.clone());
    pause_next_event_search_finalization_for_test(move |paused: &Store| {
        let finishing = one_work(paused, context_id, expected);
        assert_eq!(finishing.captured_events, 1);
        assert_eq!(
            finishing.covered_events, 1,
            "the replacement finished its source work"
        );
        assert_eq!(
            finishing.pending_events, 0,
            "no source events remain in the replacement"
        );
        assert!(
            finishing.work_required,
            "zero pending in an unactivated replacement must not tell import that projection work is complete"
        );
    });
    store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: None,
            embedder: embedder.registration(),
            profile: replacement_profile,
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
            max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
            lease_duration_ms: NonZeroU64::new(OBSERVED_LEASE_MS).expect("nonzero lease"),
        })
        .expect("the replacement activates after the rendezvous");
}

/// The foreground receipt reports inference API calls, not covered events or returned passages.
/// A failed call is still an invocation: it crossed the embedder boundary and must remain counted.
#[test]
fn observed_run_counts_each_embedding_invocation_including_the_failed_call() {
    let now = Wallclock::now().0;
    let _clock = Wallclock::test_clock_guard(move || now);
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("observed-invocations.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    for (source, key, ordinal, text) in [
        ("a-first", "first", 1, "first embedding invocation α"),
        ("b-fails", "second", 2, "second embedding invocation β"),
    ] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![draft(key, ordinal, EventRenderRole::HumanMessage, text)],
            ))
            .expect("capture independently claimed fixture source");
    }
    drop(store);

    let embedder = ControlledEmbedder::new();
    let actual_calls = Arc::new(AtomicU64::new(0));
    let calls = actual_calls.clone();
    *embedder.embed_observer.lock().unwrap() = Some(Arc::new(move || {
        calls.fetch_add(1, Ordering::SeqCst);
    }));
    *embedder.fail_once_on.lock().expect("failure control") = Some('β');
    let observed = EventSearchProjectionRunner::catch_up_observed(EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: profile(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(1).expect("nonzero event batch"),
        max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
        lease_duration_ms: NonZeroU64::new(LEASE_MS).expect("nonzero lease"),
    });
    assert!(
        matches!(&observed.result, Err(CgError::Engine(message)) if message.contains("fixture worker killed")),
        "the second actual inference fails: {:?}",
        observed
            .result
            .as_ref()
            .map(|receipt| receipt.embedded_event_count)
    );
    assert_eq!(
        observed.failure_disposition,
        EventSearchFailureDisposition::Recorded,
        "the matching failed batch was durably recorded"
    );
    assert_eq!(
        observed.embedding_invocations, 2,
        "one successful and one failed call reached the inference API"
    );
    assert_eq!(
        actual_calls.load(Ordering::SeqCst),
        observed.embedding_invocations
    );
    let committed = embedder.embedded();
    assert_eq!(committed.len(), 1);
    let resumed = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        context_id,
        &embedder,
        profile(),
        1,
        LEASE_MS,
    ));
    assert_eq!(
        resumed.failure_disposition,
        EventSearchFailureDisposition::NoFailure
    );
    assert_eq!(resumed.result.unwrap().embedded_event_count, 2);
    assert_eq!(
        resumed.embedding_invocations, 1,
        "resume calls inference only for the uncommitted source"
    );
    assert_eq!(actual_calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        embedder
            .embedded()
            .iter()
            .filter(|text| text.contains('α'))
            .count(),
        1,
        "resume never repeats committed inference"
    );
}

#[test]
fn observed_invocations_distinguish_grouped_opaque_split_and_expired_claim_retry() {
    for case in ["grouped", "opaque", "split", "expired"] {
        let root = tempfile::tempdir().unwrap();
        let config = config(root.path().join(format!("{case}.db")));
        let clock = Arc::new(AtomicU64::new(Wallclock::now().0));
        let test_clock = clock.clone();
        let _clock = Wallclock::test_clock_guard(move || test_clock.load(Ordering::SeqCst));
        let store = open(&config);
        let context_id = create_context(&store);
        let events = match case {
            "grouped" => vec![
                draft("q", 1, EventRenderRole::HumanMessage, "α"),
                draft("a", 2, EventRenderRole::AgentMessage, "β"),
            ],
            "opaque" => vec![EventDraft {
                text: None,
                coverage: EventCoverage::Opaque,
                payload: json!({"retained": "opaque provider payload"}),
                ..draft("opaque", 1, EventRenderRole::Opaque, "")
            }],
            "split" => vec![draft(
                "split",
                1,
                EventRenderRole::ToolResult,
                &"α".repeat(400),
            )],
            _ => vec![draft("expired", 1, EventRenderRole::HumanMessage, "α")],
        };
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                case,
                events,
            ))
            .unwrap();
        drop(store);
        let embedder = ControlledEmbedder::new();
        let actual_calls = Arc::new(AtomicU64::new(0));
        let calls = actual_calls.clone();
        *embedder.embed_observer.lock().unwrap() = Some(Arc::new(move || {
            let call = calls.fetch_add(1, Ordering::SeqCst);
            if case == "expired" && call == 0 {
                // The first forward finishes after its lease expires. Its vector is discarded;
                // the unchanged cursor is reclaimed and inference must actually run again.
                clock.fetch_add(LEASE_MS + 1, Ordering::SeqCst);
            }
        }));
        let outcome = EventSearchProjectionRunner::catch_up_observed(run_request(
            &config,
            context_id,
            &embedder,
            profile(),
            8,
            LEASE_MS,
        ));
        let receipt = outcome
            .result
            .unwrap_or_else(|error| panic!("{case}: {error:?}"));
        assert_eq!(
            outcome.embedding_invocations,
            actual_calls.load(Ordering::SeqCst),
            "{case}: count actual API calls"
        );
        let store = open(&config);
        let passages = current_passages(&store, context_id);
        match case {
            "grouped" => {
                assert_eq!(receipt.embedded_event_count, 2);
                assert_eq!(passages.len(), 1);
                assert_eq!(outcome.embedding_invocations, 1);
            }
            "opaque" => {
                assert_eq!(receipt.source_watermark.event_count, 1);
                assert_eq!(receipt.excluded_event_count, 1);
                assert!(passages.is_empty());
                assert_eq!(outcome.embedding_invocations, 0);
            }
            "split" => {
                assert_eq!(receipt.embedded_event_count, 1);
                assert!(passages.len() > 1, "oversized fixture actually splits");
                assert_eq!(
                    outcome.embedding_invocations as usize,
                    passages.len(),
                    "the existing singleton forward bound counts each split call"
                );
            }
            _ => {
                assert_eq!(
                    receipt.retry_count, 1,
                    "one expired commit forces one retry"
                );
                assert_eq!(outcome.embedding_invocations, 2);
                assert_eq!(embedder.embedded().len(), 2);
                assert_eq!(embedder.embedded()[0], embedder.embedded()[1]);
                assert_eq!(
                    passages.len(),
                    1,
                    "the discarded vector never becomes duplicate coverage"
                );
                eprintln!(
                    "expiry reproduction: 2 identical successful forward calls, 1 stale-claim retry, 1 committed passage"
                );
            }
        }
        drop(store);
        let noop = EventSearchProjectionRunner::catch_up_observed(run_request(
            &config,
            context_id,
            &embedder,
            profile(),
            8,
            LEASE_MS,
        ));
        noop.result.expect("unchanged projection is complete");
        assert_eq!(
            noop.embedding_invocations, 0,
            "{case}: no-op makes no inference call"
        );
    }
}

#[test]
fn invalid_passage_profiles_are_refused_before_generation_writes() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("invalid-profile.db"));
    let store = open(&config);
    let valid_context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            valid_context_id,
            declaration(),
            "valid-profile-stream",
            vec![draft(
                "event-1",
                1,
                EventRenderRole::HumanMessage,
                "a hard passage limit at the declared prepared-token capability is valid",
            )],
        ))
        .expect("capture the valid-profile control event");
    drop(store);

    let embedder = ControlledEmbedder::new();
    let valid = EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(256).expect("nonzero target"),
        hard_passage_tokens: NonZeroU32::new(256).expect("nonzero hard limit"),
        ..profile()
    };
    run_with_profile(&config, valid_context_id, &embedder, valid, 8, LEASE_MS)
        .expect("a hard passage limit at the declared prepared-token capability runs");
    let embedded_after_valid_control = embedder.embedded().len();
    assert!(
        embedded_after_valid_control > 0,
        "the equal-limit control reached the embedder"
    );

    let forward_progress_boundary = EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(1).expect("nonzero target"),
        hard_passage_tokens: NonZeroU32::new(5).expect("nonzero hard limit"),
        oversized_event_overlap_tokens: NonZeroU32::new(4).expect("nonzero overlap"),
        ..profile()
    };
    forward_progress_boundary
        .validate()
        .expect("one token beyond the oversized-event overlap permits forward progress");

    let store = open(&config);
    let context_id = create_context_named(&store, "invalid-profile refusal");
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "invalid-profile-stream",
            vec![draft(
                "event-1",
                1,
                EventRenderRole::HumanMessage,
                "the captured event remains pending",
            )],
        ))
        .expect("capture one event before invalid-profile projection");
    drop(store);

    let invalid = EventSearchProfile {
        hard_passage_tokens: NonZeroU32::new(10_000).expect("nonzero hard limit"),
        ..profile()
    };
    let refusal = run_with_profile(&config, context_id, &embedder, invalid, 8, LEASE_MS)
        .expect_err("a hard passage limit above the declared prepared-token capability is refused");
    assert!(
        matches!(
            &refusal,
            CgError::EventSearchProfileExceedsCapability {
                requested_hard_passage_tokens: 10_000,
                maximum_prepared_tokens: 256,
            }
        ),
        "profile admission uses the typed refusal: {refusal:?}"
    );
    let refusal_text = refusal.to_string();
    assert!(
        refusal_text.contains("10000")
            && refusal_text.contains("256")
            && refusal_text.contains("exceeds")
            && refusal_text.contains("declared prepared-token maximum"),
        "the typed refusal names the requested limit, declared capability, and refusal relation: {refusal_text}"
    );

    let store = open(&config);
    assert_eq!(
        generation_count(&store, context_id),
        0,
        "profile admission must run before it creates a generation"
    );
    assert_eq!(
        all_passage_count(&store, context_id),
        0,
        "profile admission must run before it writes projected passages"
    );
    assert!(
        embedder.embedded().len() == embedded_after_valid_control,
        "profile admission must run before generation work reaches the embedder"
    );

    let overlap_invalid = EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(4).expect("nonzero target"),
        hard_passage_tokens: NonZeroU32::new(4).expect("nonzero hard limit"),
        oversized_event_overlap_tokens: NonZeroU32::new(4).expect("nonzero overlap"),
        ..profile()
    };
    let overlap_refusal =
        run_with_profile(&config, context_id, &embedder, overlap_invalid, 8, LEASE_MS)
            .expect_err("a hard limit equal to the oversized-event overlap is refused");
    assert!(
        matches!(
            overlap_refusal,
            CgError::EventSearchOverlapInvalid {
                hard_passage_tokens: 4,
                oversized_event_overlap_tokens: 4,
            }
        ),
        "profile admission names the forward-progress bounds: {overlap_refusal:?}"
    );
    let store = open(&config);
    assert_eq!(
        generation_count(&store, context_id),
        0,
        "overlap admission also runs before it creates a generation"
    );
}

#[test]
fn invalid_profile_preserves_an_in_progress_build() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("invalid-profile-in-progress.db"));
    let store = open(&config);
    let context_id = create_context_named(&store, "invalid-profile preserves building progress");
    for (source, key, text) in [
        ("a-committed", "first", "the first batch commits α"),
        ("b-failing", "second", "the second batch fails β"),
    ] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![draft(key, 1, EventRenderRole::HumanMessage, text)],
            ))
            .expect("capture the interrupted-build fixture event");
    }
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('β');
    run(&config, context_id, &embedder, 1, LEASE_MS)
        .expect_err("the second batch fails after the first one commits");

    let store = open(&config);
    let generation_rows = store
        .execute_sql_for_test(
            "SELECT id, state, sanitized_failure, eligible_event_count, rendered_event_count, embedded_event_count, passage_count FROM event_search_generations WHERE context_id = $context_id ORDER BY id ASC",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("snapshot building generation rows")
        .rows;
    let failure_reason = store
        .event_history_status(ContextSelector::one(context_id))
        .expect("status after deterministic failure")
        .contexts
        .remove(0)
        .search
        .last_failure;
    let cursor_rows = store
        .execute_sql_for_test(
            "SELECT generation_id, source_instance_ref, committed_sequence, open_tail_start_sequence, open_tail_passage_id, watermark_sha256 FROM event_search_cursors WHERE context_id = $context_id ORDER BY generation_id ASC, source_instance_ref ASC",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("snapshot committed cursor rows")
        .rows;
    let passages = current_passages(&store, context_id);
    let passage_count = all_passage_count(&store, context_id);
    assert_eq!(generation_rows.len(), 1, "one generation remains building");
    assert_eq!(
        cursor_rows.len(),
        2,
        "each sealed source keeps its cursor, including zero-progress work"
    );
    let cursor_progress = cursor_rows
        .iter()
        .map(|row| match row.as_slice() {
            [_, Value::Text(source), Value::Int64(sequence), ..] => (source.as_str(), *sequence),
            other => panic!("unexpected event-search cursor row {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        cursor_progress,
        [("a-committed", 1), ("b-failing", 0)],
        "the first source committed its batch while the failed source remains at zero progress"
    );
    assert_eq!(passage_count, 1, "the first batch committed one passage");
    assert!(
        failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("fixture worker killed")),
        "the interrupted build records its deterministic failure: {failure_reason:?}"
    );
    drop(store);

    let invalid = EventSearchProfile {
        hard_passage_tokens: NonZeroU32::new(10_000).expect("nonzero hard limit"),
        ..profile()
    };
    let invalid_outcome = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        context_id,
        &embedder,
        invalid.clone(),
        1,
        LEASE_MS,
    ));
    assert_eq!(
        invalid_outcome.failure_disposition,
        EventSearchFailureDisposition::Unrecorded,
        "pure admission fails before the run owns a generation attempt"
    );
    let refusal = invalid_outcome
        .result
        .expect_err("the public runner refuses an invalid profile without changing progress");
    assert!(
        matches!(
            &refusal,
            CgError::EventSearchProfileExceedsCapability {
                requested_hard_passage_tokens: 10_000,
                maximum_prepared_tokens: 256,
            }
        ),
        "the public runner returns the typed capability refusal: {refusal:?}"
    );

    let store = open(&config);
    let held_build = store
        .catch_up_event_search(EventSearchBuildRequest {
            context_id,
            embedder: embedder.registration(),
            profile: invalid.clone(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(1).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
            max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
            lease_duration_ms: NonZeroU64::new(LEASE_MS).expect("nonzero lease"),
        })
        .expect_err("held build refuses a profile above its embedder ceiling");
    assert!(matches!(
        held_build,
        CgError::EventSearchProfileExceedsCapability {
            requested_hard_passage_tokens: 10_000,
            maximum_prepared_tokens: 256,
        }
    ));
    let held_rebuild = store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: None,
            embedder: embedder.registration(),
            profile: invalid.clone(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(1).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
            max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
            lease_duration_ms: NonZeroU64::new(LEASE_MS).expect("nonzero lease"),
        })
        .expect_err("held rebuild refuses a profile above its embedder ceiling");
    assert!(matches!(
        held_rebuild,
        CgError::EventSearchProfileExceedsCapability {
            requested_hard_passage_tokens: 10_000,
            maximum_prepared_tokens: 256,
        }
    ));
    drop(store);
    let recorded = EventSearchProjectionRunner::record_refusal(
        run_request(&config, context_id, &embedder, invalid, 1, LEASE_MS),
        "this must not be recorded",
    )
    .expect_err("explicit refusal validates before it creates or changes a generation");
    assert!(matches!(
        recorded,
        CgError::EventSearchProfileExceedsCapability {
            requested_hard_passage_tokens: 10_000,
            maximum_prepared_tokens: 256,
        }
    ));

    let store = open(&config);
    assert_eq!(
        store
            .execute_sql_for_test(
                "SELECT id, state, sanitized_failure, eligible_event_count, rendered_event_count, embedded_event_count, passage_count FROM event_search_generations WHERE context_id = $context_id ORDER BY id ASC",
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
            )
            .expect("resnapshot building generation rows")
            .rows,
        generation_rows,
        "the invalid public run leaves every building-generation field unchanged"
    );
    assert_eq!(
        store
            .event_history_status(ContextSelector::one(context_id))
            .expect("status after invalid profile refusal")
            .contexts
            .remove(0)
            .search
            .last_failure,
        failure_reason,
        "the invalid public run preserves the recorded failure reason"
    );
    assert_eq!(
        store
            .execute_sql_for_test(
                "SELECT generation_id, source_instance_ref, committed_sequence, open_tail_start_sequence, open_tail_passage_id, watermark_sha256 FROM event_search_cursors WHERE context_id = $context_id ORDER BY generation_id ASC, source_instance_ref ASC",
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
            )
            .expect("resnapshot committed cursor rows")
            .rows,
        cursor_rows,
        "the invalid public run leaves committed cursors unchanged"
    );
    assert_eq!(
        current_passages(&store, context_id),
        passages,
        "the invalid public run leaves current passages unchanged"
    );
    assert_eq!(
        all_passage_count(&store, context_id),
        passage_count,
        "the invalid public run leaves the passage count unchanged"
    );
}

#[test]
fn pending_discovery_uses_stream_progress_instead_of_completed_events() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("stream-progress.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    let old_stream = (1..=12)
        .map(|ordinal| {
            draft(
                &format!("old-{ordinal}"),
                ordinal,
                EventRenderRole::HumanMessage,
                if ordinal == 9 {
                    "!"
                } else {
                    "old stream event"
                },
            )
        })
        .collect();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "resumed-stream",
            old_stream,
        ))
        .expect("capture the stream to be resumed");
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('!');
    run(&config, context_id, &embedder, 1, LEASE_MS)
        .expect_err("the first worker leaves a durable partially completed generation");

    clear_event_search_discovery_observations_for_test();
    let appended_after_first_claim = Arc::new(AtomicBool::new(false));
    let append_once = Arc::clone(&appended_after_first_claim);
    let append_config = config.clone();
    *embedder.observer.lock().expect("observer lock") = Some(Arc::new(move || {
        if append_once.swap(true, Ordering::SeqCst) {
            return;
        }
        // `prepare_document_hook` runs after the runner has claimed this source and released its
        // Store for rendering, so this is a deterministic source append after the first claim.
        let store = open(&append_config);
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "new-stream",
                vec![draft(
                    "new-1",
                    1,
                    EventRenderRole::AgentMessage,
                    "new stream event appended after the first claim",
                )],
            ))
            .expect("append a stream after the resumed runner's first claim");
    }));
    run(&config, context_id, &embedder, 1, LEASE_MS)
        .expect("the public runner resumes the generation and catches up the new stream");
    assert!(
        appended_after_first_claim.load(Ordering::SeqCst),
        "the deterministic embedder hook appended the source after the runner's first claim"
    );
    let discoveries = event_search_discovery_observations_for_test();
    assert!(
        !discoveries.is_empty(),
        "the resumed public runner made pending-stream discovery passes"
    );
    assert!(
        discoveries
            .iter()
            .all(|observation| observation.stream_rows_examined <= 2),
        "each discovery visits the two source streams, not every completed event: {discoveries:?}"
    );
    assert!(
        discoveries
            .iter()
            .all(|observation| observation.cursor_lookups <= 2),
        "each discovery reads progress once per source stream: {discoveries:?}"
    );

    let status = open(&config)
        .event_history_status(ContextSelector::one(context_id))
        .expect("status after resumed catch-up")
        .contexts
        .remove(0);
    assert_eq!(status.search.pending_events, 0);
    assert_eq!(status.source_streams.len(), 2);
    assert!(
        current_passages(&open(&config), context_id)
            .iter()
            .any(|(_, text)| text.contains("new stream event appended after the first claim")),
        "the late source is actually projected, rather than only reflected in ready status"
    );
}

#[test]
fn a_worker_killed_after_a_committed_batch_resumes_without_duplicates_or_re_embedding() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("resume.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    for (source, first, second) in [("a-stream", "α", "β"), ("b-stream", "γ", "δ")] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![
                    draft(
                        &format!("{source}-1"),
                        1,
                        EventRenderRole::HumanMessage,
                        first,
                    ),
                    draft(
                        &format!("{source}-2"),
                        2,
                        EventRenderRole::AgentMessage,
                        second,
                    ),
                ],
            ))
            .expect("append one stream");
    }
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('γ');
    let killed = run(&config, context_id, &embedder, 8, LEASE_MS)
        .expect_err("the worker dies while embedding the second stream's batch");
    assert!(
        matches!(&killed, CgError::Engine(message) if message.contains("fixture worker killed")),
        "the simulated kill surfaces unchanged: {killed:?}"
    );
    // A claim whose lease expires before its commit is re-rendered, so the first stream's batch may
    // be embedded more than once before it commits; the second stream's batch never completes.
    let first_run = embedder.embedded();
    assert!(
        !first_run.is_empty()
            && first_run
                .iter()
                .all(|text| text.contains('α') && text.contains('β') && !text.contains('γ')),
        "only the first stream's batch was embedded before the kill: {first_run:?}"
    );

    let inspect = open(&config);
    let committed = current_passages(&inspect, context_id);
    assert_eq!(
        committed.len(),
        1,
        "the first batch committed atomically before the kill: {committed:?}"
    );
    let committed_id = committed[0].0;
    drop(inspect);

    // The killed worker's lease is left behind exactly as a dead process would leave it. The rerun
    // reads the seamed clock past that lease's expiry instead of waiting it out.
    let past_expiry = Wallclock::now().0 + LEASE_MS * 2;
    let resumed = {
        let _clock = Wallclock::test_clock_guard(move || past_expiry);
        run(&config, context_id, &embedder, 8, LEASE_MS)
    }
    .expect("the rerun resumes after the lease expires");
    let resumed_inputs = embedder.embedded()[first_run.len()..].to_vec();
    assert!(
        !resumed_inputs.is_empty()
            && resumed_inputs
                .iter()
                .all(|text| !text.contains('α') && !text.contains('β')),
        "the completed passage is never re-embedded: {resumed_inputs:?}"
    );
    assert!(
        resumed_inputs
            .iter()
            .any(|text| text.contains('γ') && text.contains('δ')),
        "the interrupted range is embedded on resume: {resumed_inputs:?}"
    );
    assert_eq!(resumed.source_watermark.event_count, 4);
    assert_eq!(resumed.embedded_event_count, 4);

    let after = open(&config);
    let current = current_passages(&after, context_id);
    assert_eq!(
        current.len(),
        2,
        "one current passage per stream, no duplicate: {current:?}"
    );
    assert_eq!(
        current[0].0, committed_id,
        "the committed batch's passage is reused, not rebuilt"
    );
    assert_eq!(
        all_passage_count(&after, context_id),
        2,
        "the resume wrote no duplicate passage rows, current or retired"
    );
    for marker in ['α', 'β', 'γ', 'δ'] {
        assert_eq!(
            current
                .iter()
                .filter(|(_, text)| text.contains(marker))
                .count(),
            1,
            "event {marker} is covered by exactly one current passage"
        );
    }
    drop(after);

    let before_unchanged = embedder.embedded().len();
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("an unchanged rerun completes");
    assert_eq!(
        embedder.embedded().len(),
        before_unchanged,
        "a second unchanged run performs zero embedder calls"
    );
}

#[test]
fn reader_refusal_after_a_committed_first_build_batch_keeps_it_resumable() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("reader-refusal-resume.db"));
    let store = open(&config);
    let context_id = create_context_named(&store, "reader refusal resumes first build");
    for (source, first, second) in [("a-committed", "α", "β"), ("b-pending", "γ", "δ")] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![
                    draft(
                        &format!("{source}-1"),
                        1,
                        EventRenderRole::HumanMessage,
                        first,
                    ),
                    draft(
                        &format!("{source}-2"),
                        2,
                        EventRenderRole::AgentMessage,
                        second,
                    ),
                ],
            ))
            .expect("capture first-build fixture events");
    }
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('γ');
    run(&config, context_id, &embedder, 8, LEASE_MS)
        .expect_err("the first source commits before the second source stops the worker");
    let before_refusal = embedder.embedded();
    assert!(
        before_refusal
            .iter()
            .all(|text| text.contains('α') && text.contains('β')),
        "only the committed first batch reached embedding: {before_refusal:?}"
    );

    EventSearchProjectionRunner::record_refusal(
        EventSearchRunRequest {
            store: config.clone(),
            contexts: ContextSelector::one(context_id),
            embedder: embedder.registration(),
            profile: profile(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
            max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
            lease_duration_ms: NonZeroU64::new(LEASE_MS).expect("nonzero lease"),
        },
        "readers held the store past the worker allowance",
    )
    .expect("record reader refusal on the resumable first build");

    let failed = open(&config)
        .event_history_status(ContextSelector::one(context_id))
        .expect("status after reader refusal")
        .contexts
        .remove(0)
        .search;
    assert_eq!(failed.state, context_graph::EventHistorySearchState::Failed);
    assert_eq!(failed.covered_events, 0);
    assert_eq!(failed.pending_events, 4);
    assert!(
        failed.building_generation_id.is_some(),
        "the first build remains the resumable generation"
    );
    assert_eq!(failed.building_covered_events, 0);
    assert_eq!(failed.building_pending_events, 4);
    assert_eq!(
        failed.last_failure.as_deref(),
        Some("readers held the store past the worker allowance")
    );

    let building_id = failed
        .building_generation_id
        .expect("the resumable generation has an id");
    let store = open(&config);
    store
        .execute_sql_for_test(
            "UPDATE event_search_generations SET state = $state WHERE id = $id",
            &HashMap::from([
                ("state".into(), Value::Text("failed".into())),
                ("id".into(), Value::Uuid(building_id.as_uuid())),
            ]),
        )
        .expect("mark the historical failed-generation fixture");
    drop(store);
    let historical_failure = open(&config)
        .event_history_status(ContextSelector::one(context_id))
        .expect("status with a genuinely failed generation")
        .contexts
        .remove(0)
        .search;
    assert_eq!(
        historical_failure.building_generation_id, None,
        "a genuinely failed row is never shown as resumable building progress"
    );
    assert_eq!(
        historical_failure.state,
        context_graph::EventHistorySearchState::Failed,
        "a legacy failed generation remains failed after status separates it from building progress"
    );
    assert_eq!(
        historical_failure.last_failure.as_deref(),
        Some("readers held the store past the worker allowance"),
        "a legacy failed generation retains its recorded failure reason"
    );
    let store = open(&config);
    store
        .execute_sql_for_test(
            "UPDATE event_search_generations SET state = $state WHERE id = $id",
            &HashMap::from([
                ("state".into(), Value::Text("building".into())),
                ("id".into(), Value::Uuid(building_id.as_uuid())),
            ]),
        )
        .expect("restore the resumable first-build fixture");
    drop(store);

    let past_expiry = Wallclock::now().0 + LEASE_MS * 2;
    let resumed = {
        let _clock = Wallclock::test_clock_guard(move || past_expiry);
        run(&config, context_id, &embedder, 8, LEASE_MS)
    }
    .expect("a later run resumes the recorded reader refusal");
    let resumed_inputs = embedder.embedded()[before_refusal.len()..].to_vec();
    assert!(
        resumed_inputs
            .iter()
            .all(|text| !text.contains('α') && !text.contains('β')),
        "the committed first batch is never embedded again: {resumed_inputs:?}"
    );
    assert!(
        resumed_inputs
            .iter()
            .any(|text| text.contains('γ') && text.contains('δ')),
        "the later run embeds only the pending batch: {resumed_inputs:?}"
    );
    assert_eq!(resumed.embedded_event_count, 4);
    assert_eq!(
        generation_count(&open(&config), context_id),
        1,
        "resumption completes the original first-build generation"
    );
}

#[test]
fn readers_between_incremental_tail_claims_see_the_complete_old_or_new_tail_never_a_mix() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("tail-snapshot.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "tail-stream",
            vec![
                draft("closed-human", 1, EventRenderRole::HumanMessage, "α"),
                draft("closed-request", 2, EventRenderRole::ToolRequest, "β"),
                draft("tail-result", 3, EventRenderRole::ToolResult, "γ"),
                draft("tail-human", 4, EventRenderRole::HumanMessage, "δ"),
            ],
        ))
        .expect("append initial history");
    drop(store);

    let embedder = ControlledEmbedder::new();
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("initial projection");
    let before = open(&config);
    let initial = current_passages(&before, context_id);
    assert_eq!(
        initial.len(),
        2,
        "the fixture grouping has one closed passage and one open tail: {initial:?}"
    );
    let old_tail = initial
        .iter()
        .find(|(_, text)| text.contains('γ') && text.contains('δ'))
        .expect("the open tail holds γ and δ")
        .0;
    let initial_ids = initial.iter().map(|(id, _)| *id).collect::<BTreeSet<_>>();
    before
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "tail-stream",
            vec![
                draft("new-request", 5, EventRenderRole::ToolRequest, "ε"),
                draft("new-result", 6, EventRenderRole::ToolResult, "ζ"),
                draft("new-human", 7, EventRenderRole::HumanMessage, "η"),
                draft("new-agent", 8, EventRenderRole::AgentMessage, "θ"),
            ],
        ))
        .expect("append after the ready head");
    drop(before);

    let observations = Arc::new(Mutex::new(Vec::<Vec<(uuid::Uuid, String)>>::new()));
    let searchable_observations = Arc::new(Mutex::new(Vec::<Vec<(uuid::Uuid, String)>>::new()));
    let observer_config = config.clone();
    let observed = observations.clone();
    let observed_searchable = searchable_observations.clone();
    *embedder.observer.lock().expect("observer lock") = Some(Arc::new(move || {
        let reader = Store::open(observer_config.clone()).expect("reader opens between claims");
        observed
            .lock()
            .expect("observation log")
            .push(current_passages(&reader, context_id));
        observed_searchable
            .lock()
            .expect("searchable observation log")
            .push(searchable_current_passages(&reader, context_id));
    }));
    // One new event per claim, so the tail replacement spans several committed claims.
    run(&config, context_id, &embedder, 1, OBSERVED_LEASE_MS).expect("incremental catch-up");
    *embedder.observer.lock().expect("observer lock") = None;
    let after = open(&config);
    let mut observations = observations.lock().expect("observation log").clone();
    observations.push(current_passages(&after, context_id));
    let mut searchable_observations = searchable_observations
        .lock()
        .expect("searchable observation log")
        .clone();
    searchable_observations.push(searchable_current_passages(&after, context_id));
    assert!(
        observations.len() >= 3,
        "readers observed the projection between several claims: {}",
        observations.len()
    );

    for (index, current) in observations.iter().enumerate() {
        for marker in ['α', 'β', 'γ', 'δ'] {
            assert!(
                current.iter().any(|(_, text)| text.contains(marker)),
                "observation {index}: previously searchable event {marker} vanished: {current:?}"
            );
        }
        let old_tail_current = current.iter().any(|(id, _)| *id == old_tail);
        eprintln!(
            "observation {index}: old tail current = {old_tail_current}, current passages = {:?}",
            current.iter().map(|(_, text)| text).collect::<Vec<_>>()
        );
        let replacement_current = current.iter().any(|(id, text)| {
            !initial_ids.contains(id) && (text.contains('γ') || text.contains('δ'))
        });
        assert_ne!(
            old_tail_current, replacement_current,
            "observation {index}: exactly one of the old tail or its replacement is current: \
             {current:?}"
        );
    }
    for (index, searchable) in searchable_observations.iter().enumerate() {
        for marker in ['α', 'β', 'γ', 'δ'] {
            assert!(
                searchable.iter().any(|(_, text)| text.contains(marker)),
                "searchable observation {index}: previously published event {marker} vanished: {searchable:?}"
            );
        }
    }
    let last = observations.last().expect("final observation");
    assert!(
        !last.iter().any(|(id, _)| *id == old_tail),
        "the final projection retired the old tail"
    );
    for marker in ['ε', 'ζ', 'η', 'θ'] {
        assert!(
            last.iter().any(|(_, text)| text.contains(marker)),
            "the final projection exposes appended event {marker}: {last:?}"
        );
    }
}

#[test]
fn an_append_during_rebuild_finalization_is_caught_up_before_the_head_switches() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("finalization.db"));
    let embedder = ControlledEmbedder::new();
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open Store");
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "finalization-stream",
            vec![draft("before", 1, EventRenderRole::HumanMessage, "α")],
        ))
        .expect("append the event present when the rebuild begins");
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("an existing ready head");

    // The only way a second event exists is this append, made inside the rebuild's final
    // head-switch transaction after its revision guard and live count check.
    pause_next_event_search_finalization_for_test(move |paused: &Store| {
        paused
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "finalization-stream",
                vec![draft("during", 2, EventRenderRole::AgentMessage, "β")],
            ))
            .expect("append while the rebuild is finalizing");
    });
    let receipt = store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: None,
            embedder: embedder.registration(),
            profile: profile(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
            max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
            lease_duration_ms: NonZeroU64::new(OBSERVED_LEASE_MS).expect("nonzero lease"),
        })
        .expect("the rebuild completes");

    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    let head = store
        .execute_sql_for_test(
            "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
            &params,
        )
        .expect("inspect search head")
        .rows[0][0]
        .clone();
    let head_event_count = store
        .execute_sql_for_test(
            "SELECT event_count FROM event_search_generations WHERE id = $id",
            &HashMap::from([("id".into(), head)]),
        )
        .expect("inspect head generation")
        .rows[0][0]
        .clone();

    assert_eq!(
        receipt.source_watermark.event_count, 2,
        "the rebuild's receipt covers the event appended during its finalization"
    );
    assert_eq!(
        head_event_count,
        Value::Int64(2),
        "the head never switched to a generation that misses the appended event"
    );
}

#[test]
fn a_completed_config_runner_deletes_its_owned_lease_across_reopen_and_later_claims() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("completed-config-runner-lease.db"));
    let embedder = ControlledEmbedder::new();
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open fixture Store");
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "completed-config-runner-lease-stream",
            vec![draft(
                "completed-config-runner-lease",
                1,
                EventRenderRole::HumanMessage,
                "a completed config runner releases its projection lease",
            )],
        ))
        .expect("append rebuild fixture event");
    drop(store);

    EventSearchProjectionRunner::catch_up(run_request(
        &config,
        context_id,
        &embedder,
        profile(),
        8,
        OBSERVED_LEASE_MS,
    ))
    .expect("config runner completes its owned batch and exact lease deletion");

    let current = open(&config);
    assert_eq!(
        lease_rows(&current, context_id),
        0,
        "the completed config runner left a lease on its current Store handle"
    );
    drop(current);

    let reopened = open(&config);
    assert_eq!(
        lease_rows(&reopened, context_id),
        0,
        "a reopened Store re-read a lease after the completed config runner removed it"
    );
    reopened
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "completed-config-runner-lease-second-stream",
            vec![draft(
                "completed-config-runner-lease-second",
                2,
                EventRenderRole::AgentMessage,
                "a later config runner can claim and read back new work",
            )],
        ))
        .expect("append later claim fixture event");
    drop(reopened);
    EventSearchProjectionRunner::catch_up(run_request(
        &config,
        context_id,
        &embedder,
        profile(),
        8,
        OBSERVED_LEASE_MS,
    ))
    .expect("later config runner claims and reads back work after reopen");
}

/// Every projection lease row the Context holds, whatever its expiry.
fn lease_rows(store: &Store, context_id: ContextId) -> usize {
    store
        .execute_sql_for_test(
            "SELECT owner_token FROM event_search_leases WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect projection leases")
        .rows
        .len()
}

#[test]
fn a_run_that_gives_up_after_claiming_releases_its_claim_so_the_next_run_claims_at_once() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("release.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "a-stream",
            vec![
                draft("a-1", 1, EventRenderRole::HumanMessage, "α"),
                draft("a-2", 2, EventRenderRole::AgentMessage, "β"),
            ],
        ))
        .expect("append one stream");
    drop(store);

    // A ten-minute lease: if the run that gives up left its claim behind, nothing could claim this
    // Context again for ten minutes, and the rerun below would be refused rather than slow.
    let long_lease_ms = 600_000;
    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('α');
    let abandoned = run(&config, context_id, &embedder, 8, long_lease_ms)
        .expect_err("the run gives up while embedding the batch it claimed");
    assert!(
        matches!(&abandoned, CgError::Engine(message) if message.contains("fixture worker killed")),
        "the simulated give-up surfaces unchanged: {abandoned:?}"
    );

    let inspect = open(&config);
    assert_eq!(
        lease_rows(&inspect, context_id),
        0,
        "a run that gives up after claiming releases the claim it was holding"
    );
    drop(inspect);

    // No clock guard anywhere: the next run claims the released work at once.
    let resumed = run(&config, context_id, &embedder, 8, long_lease_ms)
        .expect("the next run claims the released work immediately");
    assert_eq!(resumed.source_watermark.event_count, 2);
    assert_eq!(resumed.embedded_event_count, 2);
    assert_eq!(
        current_passages(&open(&config), context_id).len(),
        1,
        "the retry covers the Context with one current passage"
    );
}

/// A run whose reopens meet direct readers still commits every batch.
///
/// The runner claims a batch, CLOSES the Store to render and embed, and reopens to commit. A reader
/// loading the committed image at one of those reopens refuses the writer — and a run that took
/// that refusal for an answer would return holding its claim, leaving the Context uncovered with
/// nothing running and the next attempt locked out by the orphan. Readers finish; the reopen must
/// simply be asked again.
///
/// One event per batch, so the run closes and reopens the Store for every event there is, with
/// three readers opening the store as fast as they can throughout. Whether any single reopen is
/// actually refused is up to the machine, which is why the scenario runs several times: the
/// assertion that must hold every time is that the run commits all of its work.
/// Waiting for direct readers never outlasts the allowance the caller gave it.
///
/// `wait_for_reader_release` has no deadline of its own: it wakes when the last reader lets go or
/// when its cancellation token is cancelled, and otherwise waits forever. A writer that hands it a
/// token nobody ever cancels therefore has no allowance at all, and one reader that never finishes
/// loading would stop the projection for good. The bounded form owns the token and cancels it the
/// moment the allowance is spent, so the wait always answers.
#[test]
fn waiting_for_readers_never_outlasts_the_allowance() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("bounded-wait.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "bounded-wait",
            vec![
                draft("w-1", 1, EventRenderRole::HumanMessage, "α"),
                draft("w-2", 2, EventRenderRole::AgentMessage, "β"),
            ],
        ))
        .expect("append one stream");
    drop(store);

    // A deadline already spent is answered at once, without waiting on anything at all.
    assert!(
        matches!(
            wait_for_reader_release_within(&config.db_path, Instant::now()),
            ReaderRelease::Stopped
        ),
        "a spent allowance is answered immediately"
    );

    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let config = &config;
            let stop = &stop;
            scope.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Ok(reader) =
                        ConsumerReader::open(&config.db_path, event_history_reader_declaration())
                    {
                        drop(reader);
                    }
                }
            });
        }
        for allowance in [Duration::from_millis(50), Duration::from_millis(400)] {
            // Either verdict is legitimate — the readers may let go inside the allowance, or the
            // allowance may run out first. What must never happen is not answering, and the test
            // runner's terminate-after owns that.
            let verdict =
                wait_for_reader_release_within(&config.db_path, Instant::now() + allowance);
            assert!(
                matches!(verdict, ReaderRelease::Released | ReaderRelease::Stopped),
                "a {allowance:?} allowance under churning readers answered {verdict:?}"
            );
        }
        stop.store(true, Ordering::Relaxed);
    });
}

/// Every lease row of the Context as (owner token, expires_at).
fn lease_expiries(store: &Store, context_id: ContextId) -> Vec<(String, i64)> {
    store
        .execute_sql_for_test(
            "SELECT owner_token, expires_at FROM event_search_leases WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect lease rows")
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Text(owner), Value::Timestamp(expires_at)) => (owner.clone(), *expires_at),
            other => panic!("unexpected lease row {other:?}"),
        })
        .collect()
}

fn append_many(store: &Store, context_id: ContextId, count: u64) {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "long-batch",
            (1..=count)
                .map(|ordinal| {
                    draft(
                        &format!("long-{ordinal}"),
                        ordinal,
                        EventRenderRole::ToolResult,
                        "α",
                    )
                })
                .collect(),
        ))
        .expect("append a long batch");
}

#[test]
fn a_live_worker_whose_batch_outlasts_the_lease_keeps_it_and_refuses_a_second_worker() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("renew.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_many(&store, context_id, 12);
    drop(store);

    // Injected time: every document preparation costs a tenth of the lease, so the one claimed
    // batch of twelve events runs for several whole leases.
    let clock = Arc::new(AtomicU64::new(Wallclock::now().0));
    let _clock = {
        let clock = clock.clone();
        Wallclock::test_clock_guard(move || clock.load(Ordering::SeqCst))
    };
    let embedder = ControlledEmbedder::new();
    let observations = Arc::new(Mutex::new(Vec::<(u64, Vec<(String, i64)>)>::new()));
    let second_claim = Arc::new(Mutex::new(None::<Result<EventSearchBuildReceipt>>));
    let tried_second = Arc::new(AtomicBool::new(false));
    let start = clock.load(Ordering::SeqCst);
    {
        let clock = clock.clone();
        let observations = observations.clone();
        let second_claim = second_claim.clone();
        let tried_second = tried_second.clone();
        let config = config.clone();
        *embedder.observer.lock().expect("observer lock") = Some(Arc::new(move || {
            let now = clock.fetch_add(LEASE_MS / 10, Ordering::SeqCst) + LEASE_MS / 10;
            // Opening a Store is costly, so the lease is inspected every half lease of time.
            if !(now - start).is_multiple_of(LEASE_MS / 2) {
                return;
            }
            let reader = open(&config);
            observations
                .lock()
                .expect("observations")
                .push((now, lease_expiries(&reader, context_id)));
            drop(reader);
            if now > start + LEASE_MS * 2 && !tried_second.swap(true, Ordering::SeqCst) {
                let second = ControlledEmbedder::new();
                *second_claim.lock().expect("second claim") =
                    Some(run(&config, context_id, &second, 16, LEASE_MS));
            }
        }));
    }

    let receipt = run(&config, context_id, &embedder, 16, LEASE_MS)
        .expect("the live worker commits its long batch");
    assert_eq!(receipt.embedded_event_count, 12);
    let observations = observations.lock().expect("observations").clone();
    assert!(
        observations
            .last()
            .is_some_and(|(now, _)| *now > start + LEASE_MS * 3),
        "the batch ran for several leases of injected time"
    );
    let owners = observations
        .iter()
        .flat_map(|(_, leases)| leases.iter().map(|(owner, _)| owner.clone()))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        owners.len(),
        1,
        "one worker held the claim throughout: {owners:?}"
    );
    for (now, leases) in &observations {
        assert!(
            leases
                .iter()
                .all(|(_, expires_at)| *expires_at > i64::try_from(*now).expect("time fits")),
            "the live worker's lease stays unexpired at {now}: {leases:?}"
        );
    }
    let second = second_claim
        .lock()
        .expect("second claim")
        .take()
        .expect("a second worker tried to claim mid-batch");
    assert!(
        matches!(&second, Err(CgError::Engine(message)) if message.contains("owns the unexpired claim")),
        "the second worker is refused while the first is alive: {:?}",
        second.as_ref().map(|receipt| receipt.embedded_event_count)
    );
}

#[test]
fn a_worker_that_stops_renewing_loses_its_lease_after_one_ttl() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("dead-worker.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_many(&store, context_id, 4);
    drop(store);

    let clock = Arc::new(AtomicU64::new(Wallclock::now().0));
    let _clock = {
        let clock = clock.clone();
        Wallclock::test_clock_guard(move || clock.load(Ordering::SeqCst))
    };
    // A worker dies mid-embedding without unwinding through its release path.
    let dying = ControlledEmbedder::new();
    *dying.observer.lock().expect("observer lock") =
        Some(Arc::new(|| panic!("fixture worker process killed")));
    let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(&config, context_id, &dying, 16, LEASE_MS)
    }));
    assert!(died.is_err(), "the fixture worker died mid-batch");
    let died_at = clock.load(Ordering::SeqCst);

    let inspect = open(&config);
    let left = lease_expiries(&inspect, context_id);
    drop(inspect);
    assert_eq!(left.len(), 1, "the dead worker's lease is left behind");
    let expires_at = u64::try_from(left[0].1).expect("expiry fits");
    assert!(
        expires_at <= died_at + LEASE_MS,
        "a worker that stops renewing holds its lease for at most one TTL"
    );

    let refused = run(
        &config,
        context_id,
        &ControlledEmbedder::new(),
        16,
        LEASE_MS,
    );
    assert!(
        matches!(&refused, Err(CgError::Engine(message)) if message.contains("owns the unexpired claim")),
        "within the TTL the dead worker's lease still refuses a claim"
    );
    clock.store(expires_at + 1, Ordering::SeqCst);
    let resumed = run(
        &config,
        context_id,
        &ControlledEmbedder::new(),
        16,
        LEASE_MS,
    )
    .expect("after the TTL a new worker claims the Context");
    assert_eq!(resumed.embedded_event_count, 4);
}

#[test]
fn serial_and_concurrent_runs_commit_identical_passages_vectors_order_and_coverage() {
    fn build(embedding_workers: NonZeroUsize) -> CommittedProjection {
        let root = tempfile::tempdir().expect("tempdir");
        let config = config(root.path().join("equivalent.db"));
        let store = open(&config);
        let context_id = create_context(&store);
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "equivalent-source",
                vec![
                    draft("one", 1, EventRenderRole::HumanMessage, &"α".repeat(80)),
                    draft("two", 2, EventRenderRole::AgentMessage, &"β".repeat(57)),
                    draft("three", 3, EventRenderRole::HumanMessage, &"γ".repeat(103)),
                    draft("four", 4, EventRenderRole::AgentMessage, &"δ".repeat(69)),
                ],
            ))
            .expect("capture equivalence fixture");
        drop(store);
        let embedder = ControlledEmbedder::new();
        EventSearchProjectionRunner::catch_up(run_request_with_workers(
            &config,
            context_id,
            &embedder,
            profile(),
            2,
            OBSERVED_LEASE_MS,
            embedding_workers,
        ))
        .expect("build complete fixture projection");
        committed_projection(&open(&config), context_id)
    }

    let serial = build(NonZeroUsize::MIN);
    let concurrent = build(NonZeroUsize::new(2).unwrap());
    assert!(serial.passages.len() >= 2);
    assert!(
        serial
            .passages
            .windows(2)
            .any(|pair| pair[0][5] != pair[1][5]),
        "fixture vectors must distinguish contents"
    );
    assert!(
        serial
            .passages
            .windows(2)
            .any(|pair| match (&pair[0][3], &pair[1][3]) {
                (Value::Text(a), Value::Text(b)) => a.len() != b.len(),
                _ => false,
            }),
        "fixture passage lengths must differ"
    );
    assert_eq!(
        concurrent, serial,
        "changing only the execution cap changed committed passage order, content, vectors, or coverage"
    );
}

#[test]
fn one_concurrent_failure_waits_for_its_held_sibling_commits_nothing_and_retries_cleanly() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("held-failure.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "held-failure-source",
            vec![
                {
                    let mut event = draft("held", 1, EventRenderRole::HumanMessage, "alpha held");
                    event.session_ref = "held-failure-session".into();
                    event.occurred_at = chrono::DateTime::from_timestamp(1_800_000_100, 0);
                    event
                },
                {
                    let mut event = draft("fails", 2, EventRenderRole::HumanMessage, "beta fails");
                    event.session_ref = "held-failure-session".into();
                    event.occurred_at = chrono::DateTime::from_timestamp(1_800_000_100, 0);
                    event
                },
                {
                    let mut event = draft("after", 3, EventRenderRole::HumanMessage, "gamma after");
                    event.session_ref = "held-failure-next-session".into();
                    event.occurred_at = chrono::DateTime::from_timestamp(1_800_000_000, 0);
                    event
                },
            ],
        ))
        .expect("capture one two-passage claim");
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control") = Some('b');
    let (entered_tx, entered_rx) = channel();
    let (release_tx, release_rx) = channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let call_index = Arc::new(AtomicUsize::new(0));
    let held_active = Arc::new(AtomicBool::new(false));
    {
        let release_rx = Arc::clone(&release_rx);
        let call_index = Arc::clone(&call_index);
        let held_active = Arc::clone(&held_active);
        *embedder.embed_observer.lock().expect("observer lock") = Some(Arc::new(move || {
            let ordinal = call_index.fetch_add(1, Ordering::SeqCst);
            if ordinal == 0 {
                held_active.store(true, Ordering::SeqCst);
            }
            let _ = entered_tx.send(ordinal);
            if ordinal == 0 {
                let _ = release_rx.lock().ok().and_then(|receiver| {
                    receiver
                        .recv_timeout(std::time::Duration::from_secs(30))
                        .ok()
                });
                held_active.store(false, Ordering::SeqCst);
            }
        }));
    }
    let failure_profile = EventSearchProfile {
        target_passage_tokens: NonZeroU32::MIN,
        hard_passage_tokens: NonZeroU32::new(64).unwrap(),
        ..profile()
    };
    let request = run_request_with_workers(
        &config,
        context_id,
        &embedder,
        failure_profile.clone(),
        2,
        OBSERVED_LEASE_MS,
        NonZeroUsize::new(2).unwrap(),
    );
    let (finished_tx, finished_rx) = channel();
    let active_at_return = Arc::clone(&held_active);
    let runner = std::thread::spawn(move || {
        let outcome = EventSearchProjectionRunner::catch_up_observed(request);
        assert!(
            !active_at_return.load(Ordering::SeqCst),
            "runner returned with an active sibling"
        );
        let _ = finished_tx.send(());
        outcome
    });

    let first = entered_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .ok();
    let second = entered_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .ok();
    let returned_while_held = finished_rx.try_recv().is_ok();
    let _ = release_tx.send(());
    let outcome = runner.join();

    assert_eq!(first, Some(0), "the held call never entered");
    assert_eq!(
        second,
        Some(1),
        "the failing sibling never overlapped the held call"
    );
    assert!(
        !returned_while_held,
        "the runner returned before its held call joined"
    );
    assert!(
        !held_active.load(Ordering::SeqCst),
        "a held call remained after return"
    );
    let outcome = outcome.expect("projection runner joins");
    assert!(outcome.result.is_err(), "the failing inference was hidden");

    let after_failure = open(&config);
    assert_eq!(
        lease_rows(&after_failure, context_id),
        0,
        "a failed concurrent run left a live lease behind"
    );
    assert_eq!(
        all_passage_count(&after_failure, context_id),
        0,
        "a failed claim partially committed passages"
    );
    assert_eq!(
        committed_cursor_sequences(&after_failure, context_id, "held-failure-source"),
        vec![0, 0],
        "a failed claim advanced either session cursor"
    );
    drop(after_failure);

    *embedder.embed_observer.lock().expect("observer lock") = None;
    let retry = EventSearchProjectionRunner::catch_up_observed(run_request_with_workers(
        &config,
        context_id,
        &embedder,
        failure_profile,
        2,
        OBSERVED_LEASE_MS,
        NonZeroUsize::new(2).unwrap(),
    ));
    assert!(
        retry.result.is_ok(),
        "retry did not complete: {:?}",
        retry.result
    );
    let committed = open(&config);
    assert!(!current_passages(&committed, context_id).is_empty());
    assert_eq!(
        committed_cursor_sequences(&committed, context_id, "held-failure-source"),
        vec![2, 3]
    );
}

#[test]
fn a_finished_build_with_only_a_worker_cap_change_makes_zero_embedding_calls() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("finished-cap-only.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_many(&store, context_id, 4);
    drop(store);

    let serial = ControlledEmbedder::new();
    EventSearchProjectionRunner::catch_up(run_request_with_workers(
        &config,
        context_id,
        &serial,
        profile(),
        4,
        OBSERVED_LEASE_MS,
        NonZeroUsize::MIN,
    ))
    .expect("finish serial build");

    let cap_changed = ControlledEmbedder::new();
    let rerun = EventSearchProjectionRunner::catch_up_observed(run_request_with_workers(
        &config,
        context_id,
        &cap_changed,
        profile(),
        4,
        OBSERVED_LEASE_MS,
        NonZeroUsize::new(2).unwrap(),
    ));
    assert!(
        rerun.result.is_ok(),
        "finished rerun failed: {:?}",
        rerun.result
    );
    assert_eq!(rerun.embedding_invocations, 0);
    assert!(cap_changed.embedded().is_empty());
}

#[test]
fn slow_forward_calls_within_one_claim_never_let_the_live_workers_lease_lapse() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("slow-embed.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    // One oversized event whose 3,800-scalar body slices into about forty passages: one claim,
    // embedded through several bounded batched forward calls.
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "slow-embed",
            vec![draft(
                "oversized",
                1,
                EventRenderRole::ToolResult,
                &"α".repeat(3_800),
            )],
        ))
        .expect("append one oversized event");
    drop(store);

    // Injected time: preparation is free, and every forward call costs three fifths of the
    // lease, so the claim's calls together outlast the lease even though no single call does.
    let clock = Arc::new(AtomicU64::new(Wallclock::now().0));
    let _clock = {
        let clock = clock.clone();
        Wallclock::test_clock_guard(move || clock.load(Ordering::SeqCst))
    };
    let start = clock.load(Ordering::SeqCst);
    let embedder = ControlledEmbedder::new();
    let observations = Arc::new(Mutex::new(Vec::<(u64, Vec<(String, i64)>)>::new()));
    let second_claim = Arc::new(Mutex::new(None::<Result<EventSearchBuildReceipt>>));
    let tried_second = Arc::new(AtomicBool::new(false));
    let forward_calls = Arc::new(AtomicUsize::new(0));
    let request = |embedder: &Arc<ControlledEmbedder>| EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: profile(),
        embedding_workers: NonZeroUsize::new(2).expect("concurrent lease proof"),
        max_events_per_batch: NonZeroU32::new(16).expect("nonzero batch"),
        max_passages_per_batch: NonZeroU32::new(64).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
        lease_duration_ms: NonZeroU64::new(LEASE_MS).expect("nonzero lease"),
    };
    {
        let clock = clock.clone();
        let observations = observations.clone();
        let second_claim = second_claim.clone();
        let tried_second = tried_second.clone();
        let forward_calls = forward_calls.clone();
        let config = config.clone();
        let second_request = request(&ControlledEmbedder::new());
        *embedder.embed_observer.lock().expect("observer lock") = Some(Arc::new(move || {
            let callback_clock = clock.clone();
            let _clock = Wallclock::test_clock_guard(move || callback_clock.load(Ordering::SeqCst));
            // The lease is inspected as the call begins, then the call spends its time.
            let now = clock.load(Ordering::SeqCst);
            let reader = open(&config);
            observations
                .lock()
                .expect("observations")
                .push((now, lease_expiries(&reader, context_id)));
            drop(reader);
            if now > start + LEASE_MS && !tried_second.swap(true, Ordering::SeqCst) {
                *second_claim.lock().expect("second claim") = Some(
                    EventSearchProjectionRunner::catch_up(second_request.clone()),
                );
            }
            // Two workers in one wave spend the same logical interval. Advancing once per wave
            // models elapsed time without adding concurrent call durations together.
            if forward_calls.fetch_add(1, Ordering::SeqCst) % 2 == 1 {
                clock.fetch_add(LEASE_MS * 3 / 5, Ordering::SeqCst);
            }
        }));
    }

    let receipt = EventSearchProjectionRunner::catch_up(request(&embedder))
        .expect("the live worker commits the oversized event");
    assert_eq!(receipt.embedded_event_count, 1);
    let observations = observations.lock().expect("observations").clone();
    assert!(
        observations.len() >= 3,
        "the claim's passages went through several bounded forward calls: {}",
        observations.len()
    );
    for (now, leases) in &observations {
        assert!(
            leases
                .iter()
                .all(|(_, expires_at)| *expires_at > i64::try_from(*now).expect("time fits")),
            "the live worker's lease is unexpired as each forward call begins at {now}: {leases:?}"
        );
    }
    let second = second_claim
        .lock()
        .expect("second claim")
        .take()
        .expect("a second worker tried to claim mid-claim");
    assert!(
        matches!(&second, Err(CgError::Engine(message)) if message.contains("owns the unexpired claim")),
        "the second worker is refused while the first is embedding: {:?}",
        second.as_ref().map(|receipt| receipt.embedded_event_count)
    );
}

#[test]
fn a_projection_run_that_fails_records_why_and_a_later_successful_run_clears_it() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("recorded-failure.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "failing-stream",
            vec![
                draft("one", 1, EventRenderRole::HumanMessage, "α"),
                draft("two", 2, EventRenderRole::AgentMessage, "β"),
            ],
        ))
        .expect("append");
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('α');
    let failed_run = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        context_id,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert!(
        matches!(&failed_run.result, Err(CgError::Engine(message)) if message.contains("fixture worker killed")),
        "the original specific failure reaches the caller: {:?}",
        failed_run.result
    );
    assert_eq!(
        failed_run.failure_disposition,
        EventSearchFailureDisposition::Recorded
    );

    let search = |config: &StoreConfig| {
        open(config)
            .event_history_status(ContextSelector::one(context_id))
            .expect("status")
            .contexts
            .remove(0)
            .search
    };
    let failed = search(&config);
    assert!(
        failed
            .last_failure
            .as_deref()
            .is_some_and(|failure| failure.contains("fixture worker killed")),
        "status names why the run stopped instead of reading as pending: {:?}",
        failed.last_failure
    );

    let preserved_reason = failed.last_failure.clone();
    let writer = HeldWriter::start(&config.db_path);
    context_graph::test_support::set_store_open_retry_allowance_for_test(Some(
        std::time::Duration::ZERO,
    ));
    let _ = take_event_search_membership_captures_for_test();
    let contention = EventSearchProjectionRunner::record_refusal(
        run_request(&config, context_id, &embedder, profile(), 8, LEASE_MS),
        "writer contention must not erase the existing reason",
    )
    .expect_err("a real writer denies the refusal write");
    context_graph::test_support::set_store_open_retry_allowance_for_test(None);
    assert!(matches!(contention, CgError::StoreLocked { .. }));
    assert_eq!(
        take_event_search_membership_captures_for_test(),
        0,
        "recording a refusal never rebuilds source membership"
    );
    writer.release();
    assert_eq!(
        search(&config).last_failure,
        preserved_reason,
        "a denied contention stamp leaves the prior specific failure untouched"
    );

    run(&config, context_id, &embedder, 8, LEASE_MS).expect("the rerun completes");
    let recovered = search(&config);
    assert_eq!(recovered.pending_events, 0);
    assert_eq!(
        recovered.last_failure, None,
        "a run that caught up clears the stale failure"
    );

    let store = open(&config);
    let preparation_context = create_context_named(&store, "preparation failure ownership");
    store
        .append_event_batch(AppendEventBatch::live(
            preparation_context,
            declaration(),
            "preparation-stream",
            vec![draft(
                "preparation-event",
                1,
                EventRenderRole::HumanMessage,
                "membership preparation fails once",
            )],
        ))
        .expect("append preparation fixture");
    drop(store);
    let _ = take_event_search_membership_captures_for_test();
    fail_next_event_search_membership_capture_for_test("specific membership preparation failure");
    let preparation = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        preparation_context,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert!(
        matches!(&preparation.result, Err(CgError::Engine(message)) if message == "specific membership preparation failure"),
        "preparation keeps its original error: {:?}",
        preparation.result
    );
    assert_eq!(
        preparation.failure_disposition,
        EventSearchFailureDisposition::Recorded
    );
    assert_eq!(
        take_event_search_membership_captures_for_test(),
        1,
        "failure recording must not restart preparation"
    );
    assert_eq!(
        consumer_search(&config, preparation_context)
            .last_failure
            .as_deref(),
        Some("specific membership preparation failure")
    );

    let preparation_recovered = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        preparation_context,
        &embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    assert_eq!(
        preparation_recovered.failure_disposition,
        EventSearchFailureDisposition::NoFailure
    );
    preparation_recovered
        .result
        .expect("the failed preparation remains resumable");

    let store = open(&config);
    let denied_context = create_context_named(&store, "denied failure record");
    store
        .append_event_batch(AppendEventBatch::live(
            denied_context,
            declaration(),
            "denied-record-stream",
            vec![draft(
                "denied-record-event",
                1,
                EventRenderRole::HumanMessage,
                "failure write denial γ",
            )],
        ))
        .expect("append denied-record fixture");
    drop(store);
    let denied_embedder = ControlledEmbedder::new();
    *denied_embedder
        .fail_once_on
        .lock()
        .expect("failure control lock") = Some('γ');
    let held_writer = Rc::new(RefCell::new(None));
    let hook_writer = Rc::clone(&held_writer);
    let writer_path = config.db_path.clone();
    before_next_event_search_failure_record_for_test(move || {
        *hook_writer.borrow_mut() = Some(HeldWriter::start(&writer_path));
    });
    context_graph::test_support::set_store_open_retry_allowance_for_test(Some(
        std::time::Duration::ZERO,
    ));
    let denied = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        denied_context,
        &denied_embedder,
        profile(),
        8,
        LEASE_MS,
    ));
    context_graph::test_support::set_store_open_retry_allowance_for_test(None);
    assert!(
        matches!(&denied.result, Err(CgError::Engine(message)) if message.contains("fixture worker killed")),
        "failure persistence never remaps the original error: {:?}",
        denied.result
    );
    assert_eq!(
        denied.failure_disposition,
        EventSearchFailureDisposition::Unrecorded,
        "a refused matching-attempt write cannot claim durable failure"
    );
    held_writer
        .borrow_mut()
        .take()
        .expect("failure-record hook opened its writer")
        .release();
    let pending = consumer_search(&config, denied_context);
    assert_eq!(
        pending.state,
        context_graph::EventHistorySearchState::Pending
    );
    assert_eq!(pending.last_failure, None);
    run(&config, denied_context, &denied_embedder, 8, LEASE_MS)
        .expect("a later run can finish an unrecorded failure");
}

fn attempt_rows(config: &StoreConfig, context_id: ContextId) -> Vec<Vec<Value>> {
    open(config).execute_sql_for_test(
        "SELECT id, started_at, last_attempt_at, sanitized_failure FROM event_search_generations WHERE context_id = $context_id ORDER BY started_at ASC",
        &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
    ).unwrap().rows
}

fn consumer_search(
    config: &StoreConfig,
    context_id: ContextId,
) -> context_graph::EventHistorySearchStatus {
    ConsumerReader::open(&config.db_path, event_history_reader_declaration())
        .unwrap()
        .event_history_status(ContextSelector::one(context_id))
        .unwrap()
        .contexts
        .remove(0)
        .search
}

#[test]
fn attempt_chronology_follows_resumption_and_activation_instead_of_generation_creation() {
    let now = Wallclock::now().0;
    let _clock = Wallclock::test_clock_guard(move || now);
    let root = tempfile::tempdir().unwrap();
    let config = config(root.path().join("attempt-chronology.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    for (source, marker) in [("a-committed", "α"), ("b-pending", "β")] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![draft(source, 1, EventRenderRole::HumanMessage, marker)],
            ))
            .unwrap();
    }
    drop(store);
    let old = ControlledEmbedder::new();
    *old.fail_once_on.lock().unwrap() = Some('β');
    run(&config, context_id, &old, 1, LEASE_MS)
        .expect_err("older partial fails after committing a source");
    let original = attempt_rows(&config, context_id);
    assert_eq!(original.len(), 1);
    let newer = ControlledEmbedder::new();
    let observed_new = Arc::new(AtomicBool::new(false));
    let saw = observed_new.clone();
    let inspect = config.clone();
    *newer.observer.lock().unwrap() = Some(Arc::new(move || {
        let search = consumer_search(&inspect, context_id);
        assert_eq!(
            search.last_failure, None,
            "a newer unfailed attempt suppresses the old error before any success"
        );
        saw.store(true, Ordering::SeqCst);
    }));
    *newer.fail_once_on.lock().unwrap() = Some('α');
    let changed = EventSearchProfile {
        ordinary_overlap_events: NonZeroU32::new(2).unwrap(),
        ..profile()
    };
    run_with_profile(&config, context_id, &newer, changed.clone(), 1, LEASE_MS)
        .expect_err("newer profile fails before committing");
    assert!(observed_new.load(Ordering::SeqCst));
    let two_failed = attempt_rows(&config, context_id);
    assert_eq!(two_failed.len(), 2);
    assert_eq!(
        two_failed[0], original[0],
        "a different profile's attempt never stamps or clears the old partial"
    );
    assert!(consumer_search(&config, context_id).last_failure.is_some());

    let observed_resume = Arc::new(AtomicBool::new(false));
    let saw = observed_resume.clone();
    let inspect = config.clone();
    let before = two_failed.clone();
    *old.observer.lock().unwrap() = Some(Arc::new(move || {
        let during = attempt_rows(&inspect, context_id);
        assert_eq!(
            &during[0][..2],
            &before[0][..2],
            "resumption preserves creation time and generation identity"
        );
        let (Value::TxId(resumed), Value::TxId(newer)) = (&during[0][2], &before[1][2]) else {
            panic!("actual attempt TXIDs")
        };
        assert!(
            resumed > newer,
            "the older generation now owns the newest attempt"
        );
        assert_eq!(
            during[0][3],
            Value::Null,
            "owned resumption atomically clears its own old error"
        );
        assert_eq!(
            during[1], before[1],
            "resumption never stamps or clears other generations"
        );
        let search = consumer_search(&inspect, context_id);
        assert_eq!(
            search.last_failure, None,
            "resumption suppresses the newer generation's older failure before activation"
        );
        assert_eq!(search.building_profile, Some(profile()));
        assert_eq!(search.building_covered_events, 0);
        assert_eq!(search.building_pending_events, 2);
        saw.store(true, Ordering::SeqCst);
    }));
    let receipt = run(&config, context_id, &old, 1, LEASE_MS)
        .expect("the older generation resumes and activates");
    *old.observer.lock().unwrap() = None;
    assert!(observed_resume.load(Ordering::SeqCst));
    assert_eq!(Value::Uuid(receipt.generation_id.as_uuid()), original[0][0]);
    let after = attempt_rows(&config, context_id);
    assert_eq!(
        after[1], two_failed[1],
        "activation leaves the competing failure as historical metadata"
    );
    let head = open(&config)
        .execute_sql_for_test(
            "SELECT activated_at FROM event_search_heads WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .unwrap()
        .rows;
    let (Value::TxId(activated), Value::TxId(attempt)) = (&head[0][0], &after[0][2]) else {
        panic!("activation and attempt TXIDs")
    };
    assert!(activated > attempt);
    assert_eq!(consumer_search(&config, context_id).last_failure, None);

    EventSearchProjectionRunner::record_refusal(
        run_request(&config, context_id, &old, profile(), 1, LEASE_MS),
        "active head maintenance refused",
    )
    .unwrap();
    let ready = consumer_search(&config, context_id);
    assert_eq!(
        attempt_rows(&config, context_id)[1],
        two_failed[1],
        "explicit refusal updates only the active generation it owns"
    );
    assert_eq!(
        ready.state,
        context_graph::EventHistorySearchState::Ready,
        "active coverage retains Ready precedence"
    );
    assert_eq!(
        ready.last_failure.as_deref(),
        Some("active head maintenance refused")
    );
    assert_eq!(ready.active_generation_id, Some(receipt.generation_id));
    assert_ne!(
        ready.building_generation_id, ready.active_generation_id,
        "failure never invents an active replacement"
    );
    run(&config, context_id, &old, 1, LEASE_MS).unwrap();
    assert_eq!(consumer_search(&config, context_id).last_failure, None);
}

#[test]
fn a_stale_attempt_failure_cannot_overwrite_a_newer_owned_refusal() {
    let root = tempfile::tempdir().unwrap();
    let config = config(root.path().join("stale-attempt.db"));
    let clock = Arc::new(AtomicU64::new(Wallclock::now().0));
    let test_clock = clock.clone();
    let _clock = Wallclock::test_clock_guard(move || test_clock.load(Ordering::SeqCst));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "stale-source",
            vec![draft("first", 1, EventRenderRole::HumanMessage, "α")],
        ))
        .unwrap();
    drop(store);
    let stale = ControlledEmbedder::new();
    *stale.fail_once_on.lock().unwrap() = Some('α');
    let inspect = config.clone();
    let saved = Arc::new(Mutex::new(None));
    let newer_marker = saved.clone();
    *stale.embed_observer.lock().unwrap() = Some(Arc::new(move || {
        let callback_clock = clock.clone();
        let _clock = Wallclock::test_clock_guard(move || callback_clock.load(Ordering::SeqCst));
        let before = attempt_rows(&inspect, context_id);
        let next = ControlledEmbedder::new();
        let request = run_request(&inspect, context_id, &next, profile(), 1, LEASE_MS);
        let refused = EventSearchProjectionRunner::record_refusal(
            request.clone(),
            "nonowner must not publish",
        )
        .expect_err("a live lease refuses another attempt");
        assert!(
            matches!(refused, CgError::Engine(message) if message.contains("owns the unexpired claim"))
        );
        let mut invalid = request.clone();
        invalid.profile.target_passage_tokens = NonZeroU32::new(300).unwrap();
        let invalid = EventSearchProjectionRunner::catch_up_observed(invalid);
        assert!(invalid.result.is_err());
        assert_eq!(invalid.embedding_invocations, 0);
        assert_eq!(
            invalid.failure_disposition,
            EventSearchFailureDisposition::Unrecorded
        );
        assert_eq!(
            attempt_rows(&inspect, context_id),
            before,
            "invalid and nonowner refusals change no attempt metadata"
        );
        clock.fetch_add(LEASE_MS + 1, Ordering::SeqCst);
        EventSearchProjectionRunner::record_refusal(request, "newer owned refusal")
            .expect("expired ownership allows a newer real attempt");
        let after = attempt_rows(&inspect, context_id);
        assert_eq!(&after[0][..2], &before[0][..2]);
        let (Value::TxId(old), Value::TxId(new)) = (&before[0][2], &after[0][2]) else {
            panic!("attempt TXIDs")
        };
        assert!(new > old);
        assert_eq!(after[0][3], Value::Text("newer owned refusal".into()));
        *newer_marker.lock().unwrap() = Some(after);
    }));
    let late = EventSearchProjectionRunner::catch_up_observed(run_request(
        &config,
        context_id,
        &stale,
        profile(),
        1,
        LEASE_MS,
    ));
    assert!(
        matches!(late.result, Err(CgError::Engine(message)) if message.contains("fixture worker killed")),
        "the original call really returns a late recordable failure"
    );
    assert_eq!(late.embedding_invocations, 1);
    assert_eq!(
        late.failure_disposition,
        EventSearchFailureDisposition::Stale,
        "a late result reports takeover instead of marking the generation failed"
    );
    assert_eq!(
        attempt_rows(&config, context_id),
        saved.lock().unwrap().take().expect("newer attempt ran"),
        "late failure cannot overwrite the new attempt's marker or reason"
    );
    assert_eq!(
        consumer_search(&config, context_id).last_failure.as_deref(),
        Some("newer owned refusal")
    );
}

#[test]
fn building_progress_is_visible_without_replacing_the_ready_head() {
    // Status progress is independent of lease expiry. The invocation proof above explicitly
    // exercises expired-claim retries; keep this fixture's claims live even under suite load.
    let now = Wallclock::now().0;
    let _clock = Wallclock::test_clock_guard(move || now);
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("building-progress-status.db"));
    let store = open(&config);

    let first_generation_context = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            first_generation_context,
            declaration(),
            "first-generation-progress",
            vec![
                draft(
                    "first",
                    1,
                    EventRenderRole::HumanMessage,
                    "committed first-generation progress α",
                ),
                draft(
                    "second",
                    2,
                    EventRenderRole::AgentMessage,
                    "first-generation failure β",
                ),
            ],
        ))
        .expect("append first-generation progress fixture");
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('β');
    let first_generation_failed = run(&config, first_generation_context, &embedder, 1, LEASE_MS)
        .expect_err("the first generation fails after its first batch commits");
    assert!(
        matches!(&first_generation_failed, CgError::Engine(message) if message.contains("fixture worker killed")),
        "the deterministic first-generation failure remains visible: {first_generation_failed:?}"
    );
    let store = open(&config);
    let context_id = create_context_named(&store, "building-progress ready head");
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "a-ready",
            vec![draft(
                "ready",
                1,
                EventRenderRole::HumanMessage,
                "ready head coverage",
            )],
        ))
        .expect("append the first generation's event");
    drop(store);

    run(&config, context_id, &embedder, 1, LEASE_MS).expect("the first generation is ready");
    let ready = open(&config)
        .event_history_status(ContextSelector::one(context_id))
        .expect("status after the first generation")
        .contexts
        .remove(0)
        .search;
    let ready_head = ready
        .active_generation_id
        .expect("the completed first generation is the ready head");
    assert_eq!(ready.covered_events, 1);
    assert_eq!(ready.pending_events, 0);

    let store = open(&config);
    for (source, key, ordinal, text) in [
        ("b-first", "first", 1, "committed building progress α"),
        ("z-failing", "second", 1, "building failure β"),
    ] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![draft(key, ordinal, EventRenderRole::HumanMessage, text)],
            ))
            .expect("append rebuilding event");
    }
    drop(store);

    let changed_profile = EventSearchProfile {
        ordinary_overlap_events: NonZeroU32::new(2).expect("nonzero changed overlap"),
        ..profile()
    };
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('β');
    let embedding_trace_before_rebuild = embedder.embedded().len();
    let failed = run_with_profile(
        &config,
        context_id,
        &embedder,
        changed_profile.clone(),
        1,
        LEASE_MS,
    )
    .expect_err("the changed-profile rebuild fails after two committed batches");
    assert!(
        matches!(&failed, CgError::Engine(message) if message.contains("fixture worker killed")),
        "the deterministic rebuild failure remains visible: {failed:?}"
    );

    let changed_profile_trace = &embedder.embedded()[embedding_trace_before_rebuild..];
    assert_eq!(
        changed_profile_trace.len(),
        2,
        "only the a-ready and b-first sources reached embedding before z-failing refused: {changed_profile_trace:?}"
    );
    assert!(
        changed_profile_trace[0].contains(&fixture_embedding_text("ready head coverage"))
            && changed_profile_trace[1]
                .contains(&fixture_embedding_text("committed building progress α")),
        "the ordered embedder trace proves a-ready then b-first completed before z-failing refused: {changed_profile_trace:?}"
    );
    let cursor_store = open(&config);
    assert_eq!(
        committed_cursor_sequences(&cursor_store, context_id, "a-ready"),
        vec![1, 1],
        "the ready source has a committed cursor in both the old and rebuilding generations"
    );
    assert_eq!(
        committed_cursor_sequences(&cursor_store, context_id, "b-first"),
        vec![1],
        "the second rebuilding source committed before the final failing source"
    );
    let requested_spec = work_spec(changed_profile);
    let requested = one_work(&cursor_store, context_id, requested_spec.clone());
    assert_eq!(
        (
            requested.captured_events,
            requested.covered_events,
            requested.pending_events
        ),
        (3, 0, 3),
        "an incomplete session's committed cursor work is not published searchable coverage"
    );
    assert!(requested.work_required);
    let wrong_identity = EventSearchWorkSpec {
        encoding_profile_id: "wrong-partial-encoding".into(),
        ..requested_spec.clone()
    };
    let excluded = one_work(&cursor_store, context_id, wrong_identity.clone());
    assert_eq!((excluded.covered_events, excluded.pending_events), (0, 3));
    assert_eq!(
        excluded.generation_id, None,
        "wrong-identity building progress must not be reused"
    );
    drop(cursor_store);

    let first_generation_search = serde_json::to_value(
        open(&config)
            .event_history_status(ContextSelector::one(first_generation_context))
            .expect("status after committed first-generation progress"),
    )
    .expect("serialize first-generation public status");
    let first_generation_search = &first_generation_search["contexts"][0]["search"];
    assert_eq!(
        first_generation_search["active_generation_id"],
        serde_json::Value::Null
    );
    assert_ne!(
        first_generation_search["building_generation_id"],
        serde_json::Value::Null,
        "the first generation remains observable while it is incomplete"
    );
    assert_eq!(first_generation_search["building_covered_events"], json!(0));
    assert_eq!(first_generation_search["building_pending_events"], json!(2));

    let reader = ConsumerReader::open(&config.db_path, event_history_reader_declaration())
        .expect("the event-history reader opens for committed building progress");
    assert_eq!(one_work(&reader, context_id, requested_spec), requested);
    assert_eq!(one_work(&reader, context_id, wrong_identity), excluded);
    let public = serde_json::to_value(
        reader
            .event_history_status(ContextSelector::one(context_id))
            .expect("reader status after committed building progress"),
    )
    .expect("serialize reader status");
    let search = &public["contexts"][0]["search"];
    assert_eq!(search["active_generation_id"], json!(ready_head));
    assert_eq!(search["covered_events"], json!(1));
    assert_eq!(search["pending_events"], json!(2));
    assert_ne!(
        search["building_generation_id"],
        serde_json::Value::Null,
        "a failed rebuild retains its independently observable building generation"
    );
    assert_ne!(
        search["building_generation_id"], search["active_generation_id"],
        "the changed profile keeps rebuilding progress separate from the ready head"
    );
    assert_eq!(
        search["building_profile"]["ordinary_overlap_events"],
        json!(2),
        "status reports the actual replacement profile instead of the ready head's profile"
    );
    assert_eq!(search["building_covered_events"], json!(0));
    assert_eq!(search["building_pending_events"], json!(3));
}

#[test]
fn open_tail_loads_only_text_members_across_a_long_metadata_suffix() {
    use context_graph::test_support::take_claim_loads_for_test;
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("bounded-tail.db"));
    let store = open(&cfg);
    let context = create_context(&store);
    let mut events = vec![draft(
        "text",
        1,
        EventRenderRole::HumanMessage,
        "unfinished alpha",
    )];
    for ordinal in 2..=257 {
        let mut event = draft(
            &format!("meta-{ordinal}"),
            ordinal,
            EventRenderRole::Metadata,
            "",
        );
        event.text = None;
        events.push(event);
    }
    events.push(draft("last", 258, EventRenderRole::AgentMessage, "omega"));
    store
        .append_event_batch(AppendEventBatch::live(
            context,
            declaration(),
            "bounded-tail",
            events,
        ))
        .unwrap();
    drop(store);
    take_claim_loads_for_test();
    run(
        &cfg,
        context,
        &ControlledEmbedder::new(),
        8,
        OBSERVED_LEASE_MS,
    )
    .unwrap();
    let loads = take_claim_loads_for_test();
    assert!(loads.len() >= 32, "long suffix must cross many claims");
    for (fresh, tail, actual_loaded_rows) in &loads {
        assert!(*fresh <= 8 && *tail <= 1, "fresh={fresh}, tail={tail}");
        assert_eq!(
            *actual_loaded_rows,
            fresh + tail,
            "counts rows actually returned by event payload reads"
        );
    }
    let store = open(&cfg);
    let passages = committed_projection(&store, context);
    assert_eq!(passages.passages.len(), 1);
    let text = &passages.passages[0][3];
    assert!(
        matches!(text, Value::Text(text) if text.contains("unfinished alpha") && text.contains("omega"))
    );
    let sources = store.execute_sql_for_test("SELECT source_order, event_start_byte, event_end_byte FROM event_search_passage_sources WHERE context_id = $context_id ORDER BY source_order ASC", &HashMap::from([("context_id".into(), Value::Uuid(context.as_uuid()))])).unwrap();
    assert_eq!(
        sources.rows.len(),
        2,
        "metadata never becomes a passage source"
    );
}

#[test]
fn missing_or_mismatched_open_tail_member_ids_are_refused_before_embedding() {
    for corruption in 0..3 {
        let root = tempfile::tempdir().unwrap();
        let cfg = config(root.path().join("bad-tail.db"));
        let store = open(&cfg);
        let context = create_context(&store);
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                "bad-tail",
                vec![
                    draft("text", 1, EventRenderRole::HumanMessage, "alpha"),
                    EventDraft {
                        text: None,
                        ..draft("metadata", 2, EventRenderRole::Metadata, "")
                    },
                ],
            ))
            .unwrap();
        drop(store);
        run(
            &cfg,
            context,
            &ControlledEmbedder::new(),
            8,
            OBSERVED_LEASE_MS,
        )
        .unwrap();
        let store = open(&cfg);
        let metadata = store.execute_sql_for_test("SELECT membership_id FROM event_search_source_events WHERE context_id = $context_id AND source_sequence = 2", &HashMap::from([("context_id".into(), Value::Uuid(context.as_uuid()))])).unwrap();
        let Value::Uuid(metadata_id) = metadata.rows[0][0] else {
            panic!("membership UUID");
        };
        let corrupt = match corruption {
            0 => json!([]),
            1 => json!([uuid::Uuid::nil()]),
            _ => json!([metadata_id]),
        };
        store.execute_sql_for_test("UPDATE event_search_cursors SET open_tail_members = $members WHERE context_id = $context_id", &HashMap::from([("members".into(), Value::Json(corrupt)), ("context_id".into(), Value::Uuid(context.as_uuid()))])).unwrap();
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                "bad-tail",
                vec![draft("next", 3, EventRenderRole::AgentMessage, "beta")],
            ))
            .unwrap();
        drop(store);
        let embedder = ControlledEmbedder::new();
        let result = run(&cfg, context, &embedder, 8, OBSERVED_LEASE_MS);
        assert!(
            result.is_err(),
            "corrupt continuation must not become a different passage"
        );
        assert!(embedder.embedded_documents.lock().unwrap().is_empty());
    }
}

#[test]
fn projection_writer_hold_child() {
    use std::io::Write;
    let Some(path) = std::env::var_os("CG_CORE_WRITER_HOLD_PATH") else {
        return;
    };
    let store = open(&config(PathBuf::from(path)));
    println!("CORE_WRITER_HELD");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    drop(store);
}

struct HeldWriter(std::process::Child);
impl HeldWriter {
    fn start(path: &Path) -> Self {
        use std::io::BufRead;
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "projection_writer_hold_child", "--nocapture"])
            .env("CG_CORE_WRITER_HOLD_PATH", path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (ready, observed) = channel();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout)
                .lines()
                .map_while(std::result::Result::ok)
            {
                if line.contains("CORE_WRITER_HELD") {
                    let _ = ready.send(());
                }
            }
        });
        let held = Self(child);
        observed
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("real writer owns Store");
        held
    }
    fn release(mut self) {
        use std::io::Write;
        writeln!(self.0.stdin.as_mut().unwrap(), "release").unwrap();
        assert!(self.0.wait().unwrap().success());
    }
}
impl Drop for HeldWriter {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn projection_retries_real_writer_at_claim_renewal_and_commit_without_reembedding() {
    for phase in ["claim", "renewal", "commit"] {
        let root = tempfile::tempdir().unwrap();
        let cfg = config(root.path().join("writer-contention.db"));
        let clock = Arc::new(AtomicU64::new(Wallclock::now().0));
        let clock_read = clock.clone();
        let _clock = Wallclock::test_clock_guard(move || clock_read.load(Ordering::SeqCst));
        let store = open(&cfg);
        let context = create_context(&store);
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                "writer-source",
                vec![draft("one", 1, EventRenderRole::HumanMessage, "alpha")],
            ))
            .unwrap();
        drop(store);
        let held = Arc::new(Mutex::new(None::<HeldWriter>));
        if phase == "claim" {
            *held.lock().unwrap() = Some(HeldWriter::start(&cfg.db_path));
        }
        let retries = Arc::new(AtomicUsize::new(0));
        let retry_count = retries.clone();
        let release_writer = held.clone();
        let (release_embed, embed_released) = channel();
        context_graph::test_support::observe_store_open_retries_for_test(move |refusal| {
            assert!(
                matches!(refusal, CgError::StoreLocked { .. }),
                "only genuine writer contention enters this allowance"
            );
            retry_count.fetch_add(1, Ordering::SeqCst);
            release_writer
                .lock()
                .unwrap()
                .take()
                .expect("writer still held at exact reopen")
                .release();
            if phase == "renewal" {
                release_embed.send(()).unwrap();
            }
        });
        let embedder = ControlledEmbedder::new();
        if phase != "claim" {
            let path = cfg.db_path.clone();
            let held = held.clone();
            let embed_released = Mutex::new(embed_released);
            *embedder.embed_observer.lock().unwrap() = Some(Arc::new(move || {
                *held.lock().unwrap() = Some(HeldWriter::start(&path));
                if phase == "renewal" {
                    clock.fetch_add(LEASE_MS / 2, Ordering::SeqCst);
                    embed_released.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10)).expect("lease renewal met and released the writer while inference remained held");
                }
            }));
        }
        let outcome = EventSearchProjectionRunner::catch_up_observed(run_request(
            &cfg,
            context,
            &embedder,
            profile(),
            8,
            LEASE_MS,
        ));
        assert!(outcome.result.is_ok(), "{phase}: {:?}", outcome.result);
        assert_eq!(
            outcome.embedding_invocations, 1,
            "{phase}: retain the same rendered batch and claim"
        );
        assert_eq!(
            retries.load(Ordering::SeqCst),
            1,
            "{phase}: exactly one real contention episode"
        );
        assert_eq!(committed_projection(&open(&cfg), context).passages.len(), 1);
        context_graph::test_support::observe_store_open_retries_for_test(|_| {});
    }
}

#[test]
fn shared_store_returns_real_writer_busy_immediately_and_projection_exhaustion_stays_typed() {
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("writer-busy.db"));
    let store = open(&cfg);
    let context = create_context(&store);
    drop(store);
    let held = HeldWriter::start(&cfg.db_path);
    context_graph::test_support::observe_store_open_retries_for_test(|_| {
        panic!("general Store admission must never wait for a writer")
    });
    assert!(matches!(
        Store::open(cfg.clone()),
        Err(CgError::StoreLocked { .. })
    ));
    let retries = Arc::new(AtomicUsize::new(0));
    let observed = retries.clone();
    context_graph::test_support::observe_store_open_retries_for_test(move |_| {
        observed.fetch_add(1, Ordering::SeqCst);
    });
    context_graph::test_support::set_store_open_retry_allowance_for_test(Some(
        std::time::Duration::ZERO,
    ));
    let outcome = EventSearchProjectionRunner::catch_up_observed(run_request(
        &cfg,
        context,
        &ControlledEmbedder::new(),
        profile(),
        8,
        LEASE_MS,
    ));
    context_graph::test_support::set_store_open_retry_allowance_for_test(None);
    context_graph::test_support::observe_store_open_retries_for_test(|_| {});
    assert!(matches!(outcome.result, Err(CgError::StoreLocked { .. })));
    assert_eq!(outcome.embedding_invocations, 0);
    assert_eq!(retries.load(Ordering::SeqCst), 1);
    held.release();
}

fn append_timestamped_sessions(store: &Store, context_id: ContextId, count: usize) {
    for index in 0..count {
        append_timestamped_session(store, context_id, index);
    }
}

fn append_timestamped_session(store: &Store, context_id: ContextId, index: usize) {
    let session = format!("scope-session-{index:02}");
    let mut event = draft(
        &format!("scope-event-{index:02}"),
        1,
        EventRenderRole::HumanMessage,
        &format!("session marker {index:02}"),
    );
    event.session_ref = session;
    event.occurred_at = Some(
        chrono::DateTime::from_timestamp(1_800_000_000 + index as i64, 0)
            .expect("fixture timestamp"),
    );
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            format!("scope-source-{index:02}"),
            vec![event],
        ))
        .expect("append timestamped session");
}

#[test]
fn recent_session_in_shared_source_survives_older_head_and_widening() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("shared-source-session-scope.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    let mut older = draft(
        "shared-older",
        1,
        EventRenderRole::HumanMessage,
        "older marker",
    );
    older.session_ref = "shared-older-session".into();
    older.occurred_at = chrono::DateTime::from_timestamp(1_800_000_000, 0);
    let mut recent = draft(
        "shared-recent",
        2,
        EventRenderRole::HumanMessage,
        "recent marker",
    );
    recent.session_ref = "shared-recent-session".into();
    recent.occurred_at = chrono::DateTime::from_timestamp(1_800_000_100, 0);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "shared-live-source",
            vec![older.clone(), recent.clone()],
        ))
        .expect("append two sessions in one live source");
    drop(store);

    let embedder = ControlledEmbedder::new();
    let recent_only = EventSearchWorkPolicy {
        order: EventSearchOrder::Newest,
        sessions: EventSearchSessionScope::Recent(NonZeroUsize::MIN),
    };
    let first = scoped_run(&config, context_id, &embedder, recent_only);
    assert_eq!(first.source_watermark.event_count, 1);
    assert_eq!(
        searchable_session_order(&open(&config), context_id),
        ["shared-recent-session"]
    );
    assert_eq!(
        committed_cursor_sequences(&open(&config), context_id, "shared-live-source"),
        [0, 2],
        "the unselected older session retains its own unadvanced sequence"
    );
    let calls = embedder.embedded().len();
    let repeated = scoped_run(&config, context_id, &embedder, recent_only);
    assert_eq!(repeated.generation_id, first.generation_id);
    assert_eq!(embedder.embedded().len(), calls);

    let widened = scoped_run(
        &config,
        context_id,
        &embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::All,
        },
    );
    assert_eq!(widened.generation_id, first.generation_id);
    assert_eq!(embedder.embedded().len(), calls + 1);
    assert_eq!(
        committed_cursor_sequences(&open(&config), context_id, "shared-live-source"),
        [1, 2],
        "widening advances only the older session cursor"
    );
    assert_eq!(
        searchable_session_order(&open(&config), context_id),
        ["shared-recent-session", "shared-older-session"]
    );

    // Restart after the older session's final passage was published while a newer source event
    // belonged to another session. Its next event must replace that bounded tail atomically.
    let before = open(&config);
    let old_older_passage = current_passages(&before, context_id)
        .into_iter()
        .find(|(_, text)| text.contains("older marker"))
        .expect("older session has a published tail")
        .0;
    let mut appended = draft(
        "shared-older-append",
        3,
        EventRenderRole::AgentMessage,
        "older appended marker",
    );
    appended.session_ref = "shared-older-session".into();
    appended.occurred_at = chrono::DateTime::from_timestamp(1_800_000_001, 0);
    before
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "shared-live-source",
            vec![appended.clone()],
        ))
        .expect("append to older session in shared source");
    drop(before);
    let calls_before_append = embedder.embedded().len();
    let updated = scoped_run(
        &config,
        context_id,
        &embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::All,
        },
    );
    assert_eq!(updated.generation_id, first.generation_id);
    assert_eq!(
        embedder.embedded().len(),
        calls_before_append + 1,
        "only the older session's bounded tail is re-embedded"
    );
    let after = open(&config);
    let current = current_passages(&after, context_id);
    assert_eq!(
        current.len(),
        2,
        "one current passage per session: {current:?}"
    );
    assert!(
        current.iter().all(|(id, _)| *id != old_older_passage),
        "the old tail is retired in the replacement transaction"
    );
    let replacement = current
        .iter()
        .find(|(_, text)| text.contains("older marker"))
        .expect("replacement includes older tail text");
    assert!(
        replacement.1.contains("older appended marker"),
        "replacement includes the appended event: {replacement:?}"
    );
    assert_eq!(
        current_source_sequences(&after, replacement.0),
        vec![1, 3],
        "the replacement has complete links to both older-session events"
    );
    drop(after);

    // Projecting the same final source in one run must yield the same passage boundaries.
    let together_config = crate::config(root.path().join("shared-source-together.db"));
    let together_store = open(&together_config);
    let together_context = create_context(&together_store);
    together_store
        .append_event_batch(AppendEventBatch::live(
            together_context,
            declaration(),
            "shared-live-source",
            vec![older, recent, appended],
        ))
        .expect("append the same final source together");
    drop(together_store);
    let together_embedder = ControlledEmbedder::new();
    scoped_run(
        &together_config,
        together_context,
        &together_embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::All,
        },
    );
    let together = open(&together_config);
    assert_eq!(
        current_passage_contents_and_sources(&open(&config), context_id),
        current_passage_contents_and_sources(&together, together_context),
        "incremental and one-run projection have the same boundaries and source links"
    );
}

fn current_source_sequences(store: &Store, passage_id: uuid::Uuid) -> Vec<i64> {
    store.execute_sql_for_test(
        "SELECT event_id FROM event_search_passage_sources WHERE passage_id = $passage_id ORDER BY source_order ASC",
        &HashMap::from([("passage_id".into(), Value::Uuid(passage_id))]),
    ).expect("read current passage source links").rows.into_iter().map(|row| {
        let event = store.execute_sql_for_test(
            "SELECT seq FROM events WHERE id = $event_id",
            &HashMap::from([("event_id".into(), row[0].clone())]),
        ).expect("read linked event sequence");
        match event.rows[0][0] {
            Value::Int64(sequence) => sequence,
            ref other => panic!("event sequence is not an integer: {other:?}"),
        }
    }).collect()
}

fn current_passage_contents_and_sources(
    store: &Store,
    context_id: ContextId,
) -> Vec<(String, Vec<i64>)> {
    let mut passages = current_passages(store, context_id)
        .into_iter()
        .map(|(id, text)| (text, current_source_sequences(store, id)))
        .collect::<Vec<_>>();
    passages.sort();
    passages
}

fn scoped_run(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<ControlledEmbedder>,
    policy: EventSearchWorkPolicy,
) -> EventSearchBuildReceipt {
    EventSearchProjectionRunner::catch_up_observed_with_policy(
        run_request(
            config,
            context_id,
            embedder,
            profile(),
            1,
            OBSERVED_LEASE_MS,
        ),
        policy,
    )
    .result
    .expect("scoped projection succeeds")
}

fn searchable_session_order(store: &Store, context_id: ContextId) -> Vec<String> {
    store
        .execute_sql_for_test(
            "SELECT session_ref, passage_ordinal FROM event_search_passages WHERE context_id = $context_id AND is_current = $current AND is_searchable = $searchable ORDER BY passage_ordinal ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("current".into(), Value::Bool(true)),
                ("searchable".into(), Value::Bool(true)),
            ]),
        )
        .expect("read searchable session order")
        .rows
        .into_iter()
        .map(|row| match &row[0] {
            Value::Text(value) => value.clone(),
            other => panic!("session_ref is not text: {other:?}"),
        })
        .collect()
}

fn embedded_session_markers(embedder: &ControlledEmbedder) -> Vec<String> {
    let prefix = fixture_embedding_text("session marker ");
    embedder
        .embedded()
        .into_iter()
        .filter_map(|text| {
            let marker = text.find(&prefix)? + prefix.len();
            Some(text[marker..].chars().step_by(2).take(2).collect())
        })
        .collect()
}

#[test]
fn recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("recent-session-scope.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 12);
    let before_build = store
        .event_search_work_with_policy(
            ContextSelector::one(context_id),
            &EventSearchWorkSpec {
                profile: profile(),
                embedding_space_id: SPACE.into(),
                encoding_profile_id: "event-search-correctness-v1".into(),
            },
            EventSearchWorkPolicy::default(),
        )
        .expect("assess default scope before a generation exists")
        .remove(0);
    assert_eq!(
        (
            before_build.captured_events,
            before_build.covered_events,
            before_build.pending_events,
            before_build.outside_scope_events,
        ),
        (12, 0, 10, 2)
    );
    drop(store);
    let embedder = ControlledEmbedder::new();
    let recent_ten = EventSearchWorkPolicy::default();
    let first = scoped_run(&config, context_id, &embedder, recent_ten);
    assert_eq!(first.source_watermark.event_count, 10);
    let bounded_status = consumer_search(&config, context_id);
    assert_eq!(
        (
            bounded_status.state,
            bounded_status.covered_events,
            bounded_status.pending_events,
            bounded_status.outside_scope_events,
            bounded_status.no_searchable_text_events,
        ),
        (context_graph::EventHistorySearchState::Ready, 10, 0, 2, 0,),
        "bounded completion is a readable ready head with honest policy coverage"
    );
    let store = open(&config);
    assert_eq!(searchable_session_order(&store, context_id).len(), 10);
    assert_eq!(
        embedded_session_markers(&embedder),
        ["11", "10", "09", "08", "07", "06", "05", "04", "03", "02"],
        "embedder call order, independent of stored passage ordinals, proves newest-first work"
    );
    let assessment = store
        .event_search_work_with_policy(
            ContextSelector::one(context_id),
            &EventSearchWorkSpec {
                profile: profile(),
                embedding_space_id: SPACE.into(),
                encoding_profile_id: "event-search-correctness-v1".into(),
            },
            recent_ten,
        )
        .expect("assess recent scope")
        .remove(0);
    assert_eq!(
        (
            assessment.captured_events,
            assessment.covered_events,
            assessment.pending_events,
            assessment.outside_scope_events,
            assessment.no_searchable_text_events,
        ),
        (12, 10, 0, 2, 0)
    );
    drop(store);

    let calls_after_ten = embedder.embedded().len();
    let widened = scoped_run(
        &config,
        context_id,
        &embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::All,
        },
    );
    assert_eq!(widened.generation_id, first.generation_id);
    assert_eq!(embedder.embedded().len() - calls_after_ten, 2);
    assert_eq!(
        searchable_session_order(&open(&config), context_id).len(),
        12
    );

    let calls_after_all = embedder.embedded().len();
    let narrowed = scoped_run(
        &config,
        context_id,
        &embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::Recent(NonZeroUsize::MIN),
        },
    );
    assert_eq!(narrowed.generation_id, first.generation_id);
    assert_eq!(embedder.embedded().len(), calls_after_all);
    assert_eq!(
        searchable_session_order(&open(&config), context_id).len(),
        12
    );

    let store = open(&config);
    let narrowed_coverage = store
        .event_search_work_with_policy(
            ContextSelector::one(context_id),
            &EventSearchWorkSpec {
                profile: profile(),
                embedding_space_id: SPACE.into(),
                encoding_profile_id: "event-search-correctness-v1".into(),
            },
            EventSearchWorkPolicy {
                order: EventSearchOrder::Newest,
                sessions: EventSearchSessionScope::Recent(NonZeroUsize::MIN),
            },
        )
        .expect("assess narrowed scope")
        .remove(0);
    assert_eq!(
        (
            narrowed_coverage.covered_events,
            narrowed_coverage.pending_events,
            narrowed_coverage.outside_scope_events,
            narrowed_coverage.outside_scope_indexed_events,
        ),
        (12, 0, 11, 11)
    );
    append_timestamped_session(&store, context_id, 12);
    let after_capture = store
        .event_search_work_with_policy(
            ContextSelector::one(context_id),
            &EventSearchWorkSpec {
                profile: profile(),
                embedding_space_id: SPACE.into(),
                encoding_profile_id: "event-search-correctness-v1".into(),
            },
            EventSearchWorkPolicy::default(),
        )
        .expect("assess a newly captured most-recent session")
        .remove(0);
    assert_eq!(
        (
            after_capture.captured_events,
            after_capture.covered_events,
            after_capture.pending_events,
            after_capture.outside_scope_events,
            after_capture.outside_scope_indexed_events,
        ),
        (13, 12, 1, 3, 3)
    );
    assert!(after_capture.work_required);
}

#[test]
fn oldest_order_builds_the_ten_oldest_sessions_oldest_first() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("oldest-session-order.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 12);
    drop(store);
    let embedder = ControlledEmbedder::new();
    scoped_run(
        &config,
        context_id,
        &embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Oldest,
            sessions: EventSearchSessionScope::Recent(
                NonZeroUsize::new(10).expect("positive scope"),
            ),
        },
    );
    assert_eq!(
        embedded_session_markers(&embedder),
        ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
        "embedder calls prove that oldest selects and builds the oldest window"
    );
    let visible = searchable_session_order(&open(&config), context_id);
    assert!(!visible.iter().any(|session| session == "scope-session-10"));
    assert!(!visible.iter().any(|session| session == "scope-session-11"));
}

#[test]
fn newest_complete_session_is_findable_before_any_older_session_is_embedded() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("newest-session-first-read.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 12);
    drop(store);
    let embedder = ControlledEmbedder::new();
    scoped_run(
        &config,
        context_id,
        &embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::Recent(NonZeroUsize::MIN),
        },
    );
    assert_eq!(
        embedder.embedded().len(),
        1,
        "no older session reached the embedder"
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open search reader with controlled embedder");
    let found = store
        .find_event_history(
            ContextSelector::one(context_id),
            "session marker 11",
            EventSearchFilter {
                limit: NonZeroU32::new(100).expect("positive result limit"),
                ..EventSearchFilter::default()
            },
        )
        .expect("read the published newest session");
    assert!(!found.hits.is_empty());
    assert!(
        found
            .hits
            .iter()
            .all(|hit| hit.session_ref == "scope-session-11")
    );
    assert_eq!(found.coverage[0].searchable_event_count, 1);
    assert_eq!(found.coverage[0].pending_event_count, 0);
    assert_eq!(found.coverage[0].outside_scope_event_count, 11);
    assert_eq!(found.coverage[0].no_searchable_text_event_count, 0);
    let status = consumer_search(&config, context_id);
    assert_eq!(status.covered_events, 1);
    assert_eq!(status.pending_events, 0);
    assert_eq!(status.outside_scope_events, 11);
}

#[test]
fn status_and_find_report_published_coverage_while_the_selected_scope_is_mid_build() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("mid-build-policy-coverage.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 12);
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control") = Some('0');
    let outcome = EventSearchProjectionRunner::catch_up_observed_with_policy(
        run_request(
            &config,
            context_id,
            &embedder,
            profile(),
            1,
            OBSERVED_LEASE_MS,
        ),
        EventSearchWorkPolicy::default(),
    );
    assert!(
        outcome.result.is_err(),
        "the second selected session stops the run"
    );

    let status = consumer_search(&config, context_id);
    assert_eq!(status.covered_events, 1);
    assert_eq!(status.pending_events, 9);
    assert_eq!(status.outside_scope_events, 2);
    assert_eq!(status.no_searchable_text_events, 0);

    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open search reader with controlled embedder");
    let found = store
        .find_event_history(
            ContextSelector::one(context_id),
            "session marker 11",
            EventSearchFilter::default(),
        )
        .expect("read the first published session while later selected work is pending");
    assert!(!found.hits.is_empty());
    assert_eq!(found.coverage[0].searchable_event_count, 1);
    assert_eq!(found.coverage[0].pending_event_count, 9);
    assert_eq!(found.coverage[0].outside_scope_event_count, 2);
}

#[test]
fn changed_profile_replacement_keeps_the_old_head_until_the_required_union_is_ready() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("bounded-profile-replacement.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 3);
    drop(store);

    let embedder = ControlledEmbedder::new();
    let all = EventSearchWorkPolicy {
        order: EventSearchOrder::Newest,
        sessions: EventSearchSessionScope::All,
    };
    let first = scoped_run(&config, context_id, &embedder, all);
    let recent_one = EventSearchWorkPolicy {
        order: EventSearchOrder::Newest,
        sessions: EventSearchSessionScope::Recent(NonZeroUsize::MIN),
    };
    let narrowed = scoped_run(&config, context_id, &embedder, recent_one);
    assert_eq!(narrowed.generation_id, first.generation_id);

    let changed_profile = EventSearchProfile {
        ordinary_overlap_events: NonZeroU32::new(2).expect("nonzero overlap"),
        ..profile()
    };
    let calls_before_replacement = embedder.embedded().len();
    *embedder.fail_once_on.lock().expect("failure control") = Some('1');
    let failed = EventSearchProjectionRunner::catch_up_observed_with_policy(
        run_request(
            &config,
            context_id,
            &embedder,
            changed_profile.clone(),
            1,
            OBSERVED_LEASE_MS,
        ),
        recent_one,
    )
    .result
    .expect_err("replacement pauses after publishing only part of its required union");
    assert!(
        matches!(failed, CgError::Engine(message) if message.contains("fixture worker killed"))
    );

    let interim = consumer_search(&config, context_id);
    assert_eq!(interim.active_generation_id, Some(first.generation_id));
    assert_ne!(interim.building_generation_id, Some(first.generation_id));
    let readable_store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open the previous head with its embedder");
    let readable = readable_store
        .find_event_history(
            ContextSelector::one(context_id),
            "session marker",
            EventSearchFilter {
                limit: NonZeroU32::new(100).expect("positive limit"),
                ..EventSearchFilter::default()
            },
        )
        .expect("the previous head remains readable during replacement");
    assert_eq!(readable.hits.len(), 3);

    let completed = EventSearchProjectionRunner::catch_up_observed_with_policy(
        run_request(
            &config,
            context_id,
            &embedder,
            changed_profile,
            1,
            OBSERVED_LEASE_MS,
        ),
        recent_one,
    )
    .result
    .expect("replacement resumes and atomically switches its complete required union");
    assert_ne!(completed.generation_id, first.generation_id);
    assert_eq!(completed.source_watermark.event_count, 3);
    assert_eq!(embedder.embedded().len() - calls_before_replacement, 3);
    let final_status = consumer_search(&config, context_id);
    assert_eq!(
        final_status.active_generation_id,
        Some(completed.generation_id)
    );
    assert_eq!(final_status.covered_events, 3);
    assert_eq!(final_status.pending_events, 0);
    assert_eq!(final_status.outside_scope_events, 2);
    assert_eq!(final_status.outside_scope_indexed_events, 2);
}

#[test]
fn selected_no_text_session_finishes_with_nonzero_excluded_coverage() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("selected-no-text.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_session(&store, context_id, 0);
    let mut no_text = draft("scope-event-01", 1, EventRenderRole::Metadata, "unused");
    no_text.session_ref = "scope-session-01".into();
    no_text.text = None;
    no_text.interaction_ref = None;
    no_text.occurred_at = chrono::DateTime::from_timestamp(1_800_000_001, 0);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "scope-source-01",
            vec![no_text],
        ))
        .expect("append selected no-text event");
    drop(store);

    let embedder = ControlledEmbedder::new();
    let receipt = scoped_run(
        &config,
        context_id,
        &embedder,
        EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::Recent(NonZeroUsize::MIN),
        },
    );
    assert_eq!(receipt.source_watermark.event_count, 1);
    assert_eq!(receipt.eligible_event_count, 0);
    assert_eq!(receipt.excluded_event_count, 1);
    assert!(embedder.embedded().is_empty());
    let status = consumer_search(&config, context_id);
    assert_eq!(status.covered_events, 0);
    assert_eq!(status.pending_events, 0);
    assert_eq!(status.no_searchable_text_events, 1);
    assert_eq!(status.outside_scope_events, 1);
    assert_eq!(status.outside_scope_indexed_events, 0);
}

#[test]
fn reopened_published_tail_updates_session_and_generation_coverage_atomically() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("reopened-tail-coverage.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    append_timestamped_session(&store, context_id, 0);
    drop(store);
    let embedder = ControlledEmbedder::new();
    let policy = EventSearchWorkPolicy {
        order: EventSearchOrder::Newest,
        sessions: EventSearchSessionScope::Recent(NonZeroUsize::MIN),
    };
    let first = scoped_run(&config, context_id, &embedder, policy);

    let store = open(&config);
    let mut appended = draft(
        "scope-event-00-tail",
        2,
        EventRenderRole::AgentMessage,
        "reopened tail marker",
    );
    appended.session_ref = "scope-session-00".into();
    appended.occurred_at = chrono::DateTime::from_timestamp(1_800_000_010, 0);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "scope-source-00",
            vec![appended],
        ))
        .expect("append to the published session tail");
    drop(store);

    let completed = scoped_run(&config, context_id, &embedder, policy);
    assert_eq!(completed.generation_id, first.generation_id);
    assert_eq!(completed.source_watermark.event_count, 2);
    let store = open(&config);
    let progress = store
        .execute_sql_for_test(
            "SELECT event_count, committed_event_count, searchable_event_count, no_searchable_text_event_count, is_published FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND session_ref = $session",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                (
                    "generation_id".into(),
                    Value::Uuid(completed.generation_id.as_uuid()),
                ),
                ("session".into(), Value::Text("scope-session-00".into())),
            ]),
        )
        .expect("read independent transactional session progress");
    assert_eq!(
        progress.rows,
        vec![vec![
            Value::Int64(2),
            Value::Int64(2),
            Value::Int64(2),
            Value::Int64(0),
            Value::Bool(true),
        ]]
    );
    let status = consumer_search(&config, context_id);
    assert_eq!((status.covered_events, status.pending_events), (2, 0));
}

#[test]
fn claim_batch_examines_only_pending_tail() {
    let mut roots = Vec::new();
    context_graph::test_support::assert_claim_batch_examines_only_pending_tail_for_test(|| {
        let root = tempfile::tempdir().expect("tempdir");
        let db = contextdb_engine::Database::open(root.path().join("claim-tail.db"))
            .expect("open fixture store");
        roots.push(root);
        db
    });
}

#[test]
fn selected_frontier_count_does_not_examine_completed_history() {
    // The live MAINTAIN COUNT route is an upstream engine gap. This strict
    // regression deliberately remains red while it discloses DeclaredNotServed.
    let mut roots = Vec::new();
    context_graph::test_support::assert_selected_frontier_count_is_bounded_for_test(|| {
        let root = tempfile::tempdir().unwrap();
        let db = contextdb_engine::Database::open(root.path().join("count-tail.db")).unwrap();
        roots.push(root);
        db
    });
}

fn bounded_read_rows(
    cost: &context_graph::test_support::EventHistoryReadCost,
    table: &str,
) -> (u64, u64) {
    let reads = cost
        .reads
        .iter()
        .filter(|read| read.sql.contains(table))
        .collect::<Vec<_>>();
    assert!(!reads.is_empty(), "the real cursor must read {table}");
    (
        reads
            .iter()
            .map(|read| {
                read.rows_examined
                    .expect("actual cursor examined-entry telemetry is required")
            })
            .sum(),
        reads.iter().map(|read| read.charged_bytes).max().unwrap(),
    )
}

#[test]
fn recent_session_selection_retains_only_the_selected_sessions() {
    use context_graph::test_support::{
        record_event_history_read_cost_for_test, take_event_history_read_cost_for_test,
    };
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("recent-selection.db"));
    let store = open(&cfg);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 10);
    drop(store);
    let embedder = ControlledEmbedder::new();
    scoped_run(
        &cfg,
        context_id,
        &embedder,
        EventSearchWorkPolicy::default(),
    );
    let read = || {
        let mut reader =
            ConsumerReader::open(&cfg.db_path, event_history_reader_declaration()).unwrap();
        reader
            .attach_event_search_embedder_registrations(vec![embedder.registration()])
            .unwrap();
        // Warm the maintained index before measuring its working set.
        reader
            .find_event_history(
                ContextSelector::one(context_id),
                "session marker",
                EventSearchFilter::default(),
            )
            .unwrap();
        record_event_history_read_cost_for_test();
        let selection = reader
            .select_event_search_work(context_id, EventSearchWorkPolicy::default())
            .unwrap();
        let selection_cost = take_event_history_read_cost_for_test();
        record_event_history_read_cost_for_test();
        let answer = reader
            .find_event_history(
                ContextSelector::one(context_id),
                "session marker",
                EventSearchFilter::default(),
            )
            .unwrap();
        let find_cost = take_event_history_read_cost_for_test();
        record_event_history_read_cost_for_test();
        let status = reader
            .event_history_status(ContextSelector::one(context_id))
            .unwrap();
        let status_cost = take_event_history_read_cost_for_test();
        (
            selection,
            selection_cost,
            answer,
            find_cost,
            status,
            status_cost,
        )
    };
    let before = read();
    assert!(
        before.0.streams.is_empty(),
        "completed streams are not selected as pending work"
    );
    let store = open(&cfg);
    for index in 0..1_000 {
        for dated in [true, false] {
            let session = format!(
                "older-{}-{index:04}",
                if dated { "dated" } else { "undated" }
            );
            let mut event = draft(&session, 1, EventRenderRole::HumanMessage, "older metadata");
            event.session_ref = session.clone();
            event.text = None;
            event.occurred_at =
                dated.then(|| chrono::DateTime::from_timestamp(1_700_000_000 + index, 0).unwrap());
            store
                .append_event_batch(AppendEventBatch::live(
                    context_id,
                    declaration(),
                    session,
                    vec![event],
                ))
                .unwrap();
        }
    }
    drop(store);
    let after = read();
    assert_eq!(
        after.0, before.0,
        "the completed recent scope has no pending streams before or after older captures"
    );
    assert_eq!(
        after.2.hits, before.2.hits,
        "recent find keeps every original passage and source link"
    );
    for (old, new, operation) in [
        (&before.1, &after.1, "selection"),
        (&before.3, &after.3, "find"),
        (&before.5, &after.5, "status"),
    ] {
        assert_eq!(
            bounded_read_rows(new, "FROM event_context_sessions"),
            bounded_read_rows(old, "FROM event_context_sessions"),
            "{operation}: 1,000 older dated and 1,000 undated sessions cannot increase recent-selection work or charges"
        );
    }
    for cost in [
        &before.1, &after.1, &before.3, &after.3, &before.5, &after.5,
    ] {
        let reads = cost
            .reads
            .iter()
            .filter(|read| read.sql.contains("FROM event_context_sessions"))
            .collect::<Vec<_>>();
        assert_eq!(
            reads.len(),
            1,
            "dated rows fill the scope, so undated history is never read"
        );
        assert!(reads[0].sql.contains("latest_occurred_at IS NOT NULL"));
        assert_eq!(
            reads[0].trace.as_ref().unwrap().index_used.as_deref(),
            Some("idx_event_context_sessions_recency"),
            "the actual grouped cursor uses the declared recency index"
        );
    }
    assert_eq!(after.2.coverage[0].pending_event_count, 0);
    assert_eq!(after.2.coverage[0].outside_scope_event_count, 2_000);
    assert_eq!(
        after.2.coverage[0].outside_scope_indexed_event_count,
        before.2.coverage[0].outside_scope_indexed_event_count
    );
    assert_eq!(
        after.4.contexts[0].search.pending_events,
        before.4.contexts[0].search.pending_events
    );
    assert_eq!(
        after.4.contexts[0].source_streams.len(),
        before.4.contexts[0].source_streams.len() + 2_000,
        "status still lists every captured stream"
    );
}

#[test]
fn select_work_examines_no_completed_events() {
    use context_graph::test_support::{
        record_event_history_read_cost_for_test, take_event_history_read_cost_for_test,
    };
    let root = tempfile::tempdir().unwrap();
    let mut event_growth = Vec::new();
    let mut session_growth = Vec::new();
    for (completed, completed_sessions) in [(8_u64, 2_usize), (80, 2), (8, 20)] {
        let cfg = config(root.path().join(format!(
            "completed-{completed}-sessions-{completed_sessions}.db"
        )));
        let store = open(&cfg);
        let context_id = create_context(&store);
        let sessions = std::iter::once("selected".to_owned())
            .chain((1..completed_sessions).map(|index| format!("older-{index:04}")));
        for session in sessions {
            let events = (1..=completed)
                .map(|ordinal| {
                    let mut event = draft(
                        &format!("{session}-{ordinal}"),
                        ordinal,
                        EventRenderRole::HumanMessage,
                        "completed metadata",
                    );
                    event.session_ref = session.clone();
                    event.text = None;
                    event.occurred_at = chrono::DateTime::from_timestamp(
                        if session == "selected" {
                            1_800_000_000
                        } else {
                            1_700_000_000
                        },
                        0,
                    );
                    event
                })
                .collect();
            store
                .append_event_batch(AppendEventBatch::live(
                    context_id,
                    declaration(),
                    format!("{session}-completed"),
                    events,
                ))
                .unwrap();
        }
        drop(store);
        let embedder = ControlledEmbedder::new();
        EventSearchProjectionRunner::catch_up_observed_with_policy(
            run_request(
                &cfg,
                context_id,
                &embedder,
                profile(),
                128,
                OBSERVED_LEASE_MS,
            ),
            EventSearchWorkPolicy {
                order: EventSearchOrder::Newest,
                sessions: EventSearchSessionScope::All,
            },
        )
        .result
        .expect("complete the fixture before measuring selection");
        let store = open(&cfg);
        let mut pending = draft(
            "pending",
            1,
            EventRenderRole::HumanMessage,
            "same pending tail",
        );
        pending.session_ref = "selected".into();
        pending.occurred_at = chrono::DateTime::from_timestamp(1_800_000_001, 0);
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "pending-source",
                vec![pending],
            ))
            .unwrap();
        let policy = EventSearchWorkPolicy {
            order: EventSearchOrder::Newest,
            sessions: EventSearchSessionScope::All,
        };
        record_event_history_read_cost_for_test();
        let selected = store.select_event_search_work(context_id, policy).unwrap();
        let cost = take_event_history_read_cost_for_test();
        assert!(
            cost.reads
                .iter()
                .filter(|read| read.sql.contains(" FROM events "))
                .all(|read| read.sql.contains("COUNT(*)")
                    || read.sql.contains("ORDER BY seq DESC LIMIT 1")),
            "selection reads only engine-maintained counts and indexed frontiers"
        );
        assert_eq!(
            selected.streams.len(),
            1,
            "selection reaches only the identical pending stream, never completed streams"
        );
        assert_eq!(
            selected
                .streams
                .iter()
                .find(|stream| stream.source_instance_ref == "pending-source")
                .unwrap()
                .frontier,
            1,
            "identical pending frontier"
        );
        let stream_cost = bounded_read_rows(&cost, "FROM events");
        let selection_cost = bounded_read_rows(&cost, "");
        drop(store);
        let request = run_request(&cfg, context_id, &embedder, profile(), 8, OBSERVED_LEASE_MS);
        let resumed =
            EventSearchProjectionRunner::catch_up_observed_with_selected_work_and_batch_limits(
                request,
                policy,
                selected,
                || NonZeroUsize::MIN,
                |_| {
                    Ok(context_graph::EventSearchBatchWriteLimit {
                        memory_limit_bytes: cfg.write_memory_limit_bytes,
                        through_sequence: None,
                    })
                },
            )
            .result
            .unwrap();
        assert_eq!(resumed.failed_event_count, 0);
        let reader =
            ConsumerReader::open(&cfg.db_path, event_history_reader_declaration()).unwrap();
        reader
            .event_history_status(ContextSelector::one(context_id))
            .unwrap();
        record_event_history_read_cost_for_test();
        let status = reader
            .event_history_status(ContextSelector::one(context_id))
            .unwrap();
        let status_reads = take_event_history_read_cost_for_test();
        let status_cost = bounded_read_rows(&status_reads, "FROM events");
        assert_eq!(
            status.contexts[0].search.pending_events, 0,
            "the selected pending tail is claimed and completed"
        );
        assert_eq!(
            status.contexts[0].captured_events,
            completed * completed_sessions as u64 + 1
        );
        assert_eq!(
            status.contexts[0].source_streams.len(),
            completed_sessions + 1
        );
        assert_eq!(
            status.contexts[0]
                .source_streams
                .iter()
                .map(|stream| stream.event_count)
                .sum::<u64>(),
            completed * completed_sessions as u64 + 1
        );
        if completed_sessions == 2 {
            event_growth.push((stream_cost, status_cost));
        }
        if completed == 8 {
            session_growth.push(selection_cost);
        }
    }
    assert_eq!(
        event_growth[0].0, event_growth[1].0,
        "ten times the completed history examines the same newest entries at the same read charge"
    );
    assert_eq!(
        session_growth[0], session_growth[1],
        "growing completed sessions under the all-session scope must examine the same entries and charge the same bytes for an identical pending tail"
    );
    assert_eq!(
        event_growth[0].1, event_growth[1].1,
        "ten times the completed history keeps ConsumerReader status grouped counts and newest-entry reads at the same examined entries and read charge"
    );
}

#[test]
fn pending_marker_and_cursor_publish_together_across_process_crashes() {
    const CHILD_PATH: &str = "CG_PENDING_MARKER_CRASH_PATH";
    const CHILD_CONTEXT: &str = "CG_PENDING_MARKER_CRASH_CONTEXT";
    const CHILD_CUT: &str = "CG_PENDING_MARKER_CRASH_CUT";
    if let Some(path) = std::env::var_os(CHILD_PATH) {
        let cfg = config(std::path::PathBuf::from(path));
        let context_id = context_graph::ContextId::from(
            uuid::Uuid::parse_str(&std::env::var(CHILD_CONTEXT).unwrap()).unwrap(),
        );
        let cut = std::env::var(CHILD_CUT).unwrap();
        if cut == "before" {
            context_graph::test_support::before_next_event_search_batch_commit_for_test(|_| {
                std::process::exit(72)
            });
        } else {
            context_graph::test_support::after_next_event_search_batch_commit_for_test(|_| {
                std::process::exit(73)
            });
        }
        let embedder = ControlledEmbedder::new();
        let _ = EventSearchProjectionRunner::catch_up_observed_with_policy(
            run_request(&cfg, context_id, &embedder, profile(), 8, OBSERVED_LEASE_MS),
            EventSearchWorkPolicy {
                order: EventSearchOrder::Newest,
                sessions: EventSearchSessionScope::All,
            },
        );
        panic!("the production batch commit must reach its crash cut");
    }
    let root = tempfile::tempdir().unwrap();
    for (cut, expected_code, expected_sequence, expected_pending) in
        [("before", 72, 1_i64, true), ("after", 73, 2_i64, false)]
    {
        let cfg = config(root.path().join(format!("marker-crash-{cut}.db")));
        let store = open(&cfg);
        let context_id = create_context(&store);
        let mut first = draft("initial", 1, EventRenderRole::HumanMessage, "metadata");
        first.session_ref = "crash-session".into();
        first.text = None;
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "crash-source",
                vec![first],
            ))
            .unwrap();
        drop(store);
        let embedder = ControlledEmbedder::new();
        EventSearchProjectionRunner::catch_up_observed_with_policy(
            run_request(&cfg, context_id, &embedder, profile(), 8, OBSERVED_LEASE_MS),
            EventSearchWorkPolicy {
                order: EventSearchOrder::Newest,
                sessions: EventSearchSessionScope::All,
            },
        )
        .result
        .unwrap();
        let store = open(&cfg);
        let mut second = draft("tail", 2, EventRenderRole::HumanMessage, "metadata");
        second.session_ref = "crash-session".into();
        second.text = None;
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "crash-source",
                vec![second],
            ))
            .unwrap();
        drop(store);
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "pending_marker_and_cursor_publish_together_across_process_crashes",
                "--nocapture",
            ])
            .env(CHILD_PATH, &cfg.db_path)
            .env(CHILD_CONTEXT, context_id.as_uuid().to_string())
            .env(CHILD_CUT, cut)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(expected_code),
            "actual production crash cut {cut}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let recovered = open(&cfg);
        let params = std::collections::HashMap::from([(
            "context_id".into(),
            contextdb_core::Value::Uuid(context_id.as_uuid()),
        )]);
        let answer = recovered.sync_database().execute(
            "SELECT event_search_cursors.committed_sequence, event_search_pending_streams.pending FROM event_search_heads INNER JOIN event_search_cursors ON event_search_heads.context_id = event_search_cursors.context_id AND event_search_heads.generation_id = event_search_cursors.generation_id INNER JOIN event_search_pending_streams ON event_search_cursors.context_id = event_search_pending_streams.context_id AND event_search_cursors.generation_id = event_search_pending_streams.generation_id AND event_search_cursors.connector_id = event_search_pending_streams.connector_id AND event_search_cursors.source_instance_ref = event_search_pending_streams.source_instance_ref AND event_search_cursors.session_ref = event_search_pending_streams.session_ref WHERE event_search_heads.context_id = $context_id", &params).unwrap();
        assert_eq!(
            answer.rows,
            vec![vec![
                contextdb_core::Value::Int64(expected_sequence),
                contextdb_core::Value::Bool(expected_pending)
            ]],
            "crash before commit keeps both old values; crash after commit keeps both new values"
        );
        let selected = recovered
            .select_event_search_work(
                context_id,
                EventSearchWorkPolicy {
                    order: EventSearchOrder::Newest,
                    sessions: EventSearchSessionScope::All,
                },
            )
            .unwrap();
        assert_eq!(selected.streams.len(), usize::from(expected_pending));
        drop(recovered);
        EventSearchProjectionRunner::catch_up_observed_with_policy(
            run_request(&cfg, context_id, &embedder, profile(), 8, OBSERVED_LEASE_MS),
            EventSearchWorkPolicy {
                order: EventSearchOrder::Newest,
                sessions: EventSearchSessionScope::All,
            },
        )
        .result
        .unwrap();
        let complete = open(&cfg);
        assert!(
            complete
                .select_event_search_work(
                    context_id,
                    EventSearchWorkPolicy {
                        order: EventSearchOrder::Newest,
                        sessions: EventSearchSessionScope::All
                    }
                )
                .unwrap()
                .streams
                .is_empty(),
            "recovered pending work completes and becomes unreachable from the pending index"
        );
    }
}

#[test]
fn recent_coverage_reads_only_selected_progress_keys() {
    use context_graph::test_support::{
        record_event_history_read_cost_for_test, take_event_history_read_cost_for_test,
    };
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("selected-progress.db"));
    let store = open(&cfg);
    let context_id = create_context(&store);
    append_timestamped_sessions(&store, context_id, 10);
    drop(store);
    let embedder = ControlledEmbedder::new();
    scoped_run(
        &cfg,
        context_id,
        &embedder,
        EventSearchWorkPolicy::default(),
    );
    let mut previous = 0;
    let mut measured = Vec::new();
    let mut original_hits = None;
    for older_sessions in [30, 150] {
        let store = open(&cfg);
        for index in previous..older_sessions {
            let session = format!("completed-older-{index:04}");
            let mut event = draft(&session, 1, EventRenderRole::HumanMessage, "metadata");
            event.session_ref = session.clone();
            event.text = None;
            event.occurred_at = chrono::DateTime::from_timestamp(1_700_000_000 + index, 0);
            store
                .append_event_batch(AppendEventBatch::live(
                    context_id,
                    declaration(),
                    session,
                    vec![event],
                ))
                .unwrap();
        }
        drop(store);
        let request = run_request(
            &cfg,
            context_id,
            &embedder,
            profile(),
            128,
            OBSERVED_LEASE_MS,
        );
        EventSearchProjectionRunner::catch_up_observed_with_policy(
            request,
            EventSearchWorkPolicy {
                order: EventSearchOrder::Newest,
                sessions: EventSearchSessionScope::All,
            },
        )
        .result
        .unwrap();
        let store = open(&cfg);
        store
            .persist_event_search_work_policy(context_id, EventSearchWorkPolicy::default())
            .unwrap();
        drop(store);
        let mut reader =
            ConsumerReader::open(&cfg.db_path, event_history_reader_declaration()).unwrap();
        reader
            .attach_event_search_embedder_registrations(vec![embedder.registration()])
            .unwrap();
        reader
            .find_event_history(
                ContextSelector::one(context_id),
                "session marker",
                EventSearchFilter::default(),
            )
            .unwrap();
        record_event_history_read_cost_for_test();
        let answer = reader
            .find_event_history(
                ContextSelector::one(context_id),
                "session marker",
                EventSearchFilter::default(),
            )
            .unwrap();
        let cost = take_event_history_read_cost_for_test();
        measured.push(bounded_read_rows(
            &cost,
            "FROM event_search_session_progress",
        ));
        assert_eq!(answer.coverage[0].pending_event_count, 0);
        assert_eq!(
            answer.coverage[0].outside_scope_event_count,
            older_sessions as u64
        );
        assert_eq!(
            answer.coverage[0].outside_scope_indexed_event_count, 0,
            "metadata-only completed history has no indexed passages"
        );
        if let Some(hits) = &original_hits {
            assert_eq!(
                &answer.hits, hits,
                "retained exact answers do not change when no-text history grows"
            );
        } else {
            original_hits = Some(answer.hits);
        }
        previous = older_sessions;
    }
    assert_eq!(
        measured[0], measured[1],
        "forty then one hundred sixty total sessions keep the same ten keyed progress probes and charged bytes"
    );
}

fn append_projection_fixture_stream(
    store: &Store,
    context_id: ContextId,
    source: &str,
    count: u64,
    text: &str,
) {
    let events = (1..=count)
        .map(|ordinal| {
            let mut event = draft(
                &format!("{source}-event-{ordinal}"),
                ordinal,
                EventRenderRole::HumanMessage,
                text,
            );
            event.session_ref = "one-large-session".into();
            event
        })
        .collect();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            source,
            events,
        ))
        .expect("append bounded-fold fixture stream");
}

fn append_large_metadata_stream(
    store: &Store,
    context_id: ContextId,
    source: &str,
    first_ordinal: u64,
) {
    let events = (first_ordinal..first_ordinal + 3)
        .map(|ordinal| {
            let mut event = draft(
                &format!("{source}-metadata-{ordinal}"),
                ordinal,
                EventRenderRole::Metadata,
                "metadata-only",
            );
            event.session_ref = "one-large-session".into();
            event.text = None;
            event.payload = json!({ "padding": "x".repeat(1_600_000) });
            event
        })
        .collect();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            source,
            events,
        ))
        .expect("append metadata large enough to split capture pages by bytes");
}

fn append_sparse_projection_stream(store: &Store, context_id: ContextId, source: &str, count: u64) {
    let events = (1..=count)
        .map(|ordinal| {
            let mut event = draft(
                &format!("{source}-event-{ordinal}"),
                ordinal,
                EventRenderRole::HumanMessage,
                "short searchable event",
            );
            event.session_ref = "one-large-session".into();
            if ordinal % 64 != 1 {
                event.text = None;
                event.render_role = EventRenderRole::Metadata;
            }
            event
        })
        .collect();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            source,
            events,
        ))
        .expect("append sparse large-session fixture events");
}

/// The engine's examined-entry meter, measured once per test process on the engine under test, so
/// the unit bound follows the meter rather than one engine's numbers. `seek` is the most any one
/// positioning statement the projection runs is charged when it returns at most one row: a seek
/// that finds a row, a seek that misses, a point lookup, one event's anchor read, a pending-stream,
/// cursor, membership, passage or session-progress read, and the commit transaction's generation,
/// capture-mark, session-progress and fold-row reads. `row` is the most each further returned
/// row adds. The probe store holds 200 events so each table's scan window is as full as in the rows.
#[derive(Clone, Copy, Debug)]
struct EngineMeter {
    seek: u64,
    row: u64,
}

fn engine_meter() -> EngineMeter {
    static METER: std::sync::OnceLock<EngineMeter> = std::sync::OnceLock::new();
    *METER.get_or_init(measure_engine_meter)
}

fn measure_engine_meter() -> EngineMeter {
    let embedder = ControlledEmbedder::new();
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("cg090-meter.db"));
    let store = open(&cfg);
    let context = create_context(&store);
    append_sparse_projection_stream(&store, context, "meter", 200);
    drop(store);
    let generation = run_until_projection_completes(&cfg, context, &embedder, 256).generation_id;
    let store = open(&cfg);
    let examined = |sql: &str, extra: &[(&str, Value)]| {
        let mut params = HashMap::from([
            ("context_id".to_string(), Value::Uuid(context.as_uuid())),
            (
                "generation_id".to_string(),
                Value::Uuid(generation.as_uuid()),
            ),
            ("connector".to_string(), Value::Text(CONNECTOR.into())),
            ("source".to_string(), Value::Text("meter".into())),
            (
                "session".to_string(),
                Value::Text("one-large-session".into()),
            ),
        ]);
        params.extend(
            extra
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone())),
        );
        let answer = store
            .execute_sql_for_test(sql, &params)
            .expect("meter probe statement");
        (answer.trace.rows_examined, answer.rows)
    };
    let stream =
        "context_id = $context_id AND connector_id = $connector AND source_instance_ref = $source";
    let (one, first) = examined(
        &format!("SELECT id FROM events WHERE {stream} AND seq > 0 ORDER BY seq ASC LIMIT 1"),
        &[],
    );
    let (many, _) = examined(
        &format!("SELECT id FROM events WHERE {stream} AND seq > 0 ORDER BY seq ASC LIMIT 101"),
        &[],
    );
    let event_id = first[0][0].clone();
    let membership = "context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session";
    let positioning = [
        one,
        examined(
            &format!("SELECT id FROM events WHERE {stream} AND seq > 10000 ORDER BY seq ASC LIMIT 1"),
            &[],
        )
        .0,
        examined(
            &format!("SELECT seq FROM events WHERE {stream} ORDER BY seq DESC LIMIT 1"),
            &[],
        )
        .0,
        examined(
            "SELECT id FROM events WHERE context_id = $context_id AND id = $event_id",
            &[("event_id", event_id.clone())],
        )
        .0,
        examined(
            "SELECT key FROM event_anchors WHERE event_id = $event_id AND context_id = $context_id ORDER BY declaration_version ASC, key ASC, identity_value ASC",
            &[("event_id", event_id.clone())],
        )
        .0,
        examined(
            "SELECT ordinal FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
            &[("event_id", event_id)],
        )
        .0,
        examined(
            &format!("SELECT ordinal FROM event_search_source_events WHERE {membership} AND source_sequence > 100 ORDER BY source_sequence ASC LIMIT 1"),
            &[],
        )
        .0,
        examined(
            &format!("SELECT committed_sequence FROM event_search_cursors WHERE {membership}"),
            &[],
        )
        .0,
        examined(
            "SELECT connector_id FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND pending = false ORDER BY connector_id ASC LIMIT 1",
            &[],
        )
        .0,
        examined(
            "SELECT event_count FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND session_ref = $session",
            &[],
        )
        .0,
        examined(
            "SELECT passage_ordinal FROM event_search_passages WHERE generation_id = $generation_id AND is_current = true ORDER BY passage_ordinal ASC LIMIT 1",
            &[],
        )
        .0,
        // The commit transaction's read shapes: the generation fence, the capture mark, the
        // session progress row and the fold row.
        examined(
            "SELECT interpretation_snapshot_sha256, last_attempt_token, passage_revision FROM event_search_generations WHERE context_id = $context_id AND id = $generation_id",
            &[],
        )
        .0,
        examined(
            "SELECT capture_open, capture_pending_tx FROM event_search_pending_streams WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND source_instance_ref = $source AND session_ref = $session",
            &[],
        )
        .0,
        examined(
            "SELECT event_count, is_required FROM event_search_session_progress WHERE context_id = $context_id AND generation_id = $generation_id AND connector_id = $connector AND session_ref = $session",
            &[],
        )
        .0,
        examined(
            "SELECT id, attempt_token, unit FROM event_search_fold_progress WHERE context_id = $context_id AND generation_id = $generation_id",
            &[],
        )
        .0,
    ];
    let meter = EngineMeter {
        seek: positioning.into_iter().max().unwrap().max(1),
        row: many.saturating_sub(one).div_ceil(100).max(1),
    };
    eprintln!("measured engine meter: {meter:?} (positioning {positioning:?}, rows {one}/{many})");
    meter
}

fn unit_allowance(fold: ProjectionFold, returned_rows: u64) -> u64 {
    let meter = engine_meter();
    // Every statement a unit runs, its commit transaction included, counted from the statement
    // shapes in `event_search_projection.rs`; the receipt charges each one's examined entries.
    let (statements, statements_per_row) = match fold {
        // The largest capture unit is a capture page. Reads: up to three manifest-key branches, the
        // stream's newest member and member count, and the event page. Commit: the generation fence,
        // the manifest-frontier fence, the member append's generation and head reads and its
        // generation update, the capture mark's read and close, the manifest row's delete and the
        // fold row's compare-and-set. Each captured event: its anchor read, the retained-member check,
        // the interpretation member and effective-value reads, the member insert, the session
        // progress read and write, the generation's required-count read and write, and the cursor
        // read and insert. The manifest unit (at most nine statements, one stream), the membership
        // page (three) and the manifest reset (three plus one delete per row) all fit inside it.
        ProjectionFold::Capture => (15, 11),
        // Up to three pending-stream branches; each stream reads its cursor and first pending member.
        // Discovery and session-scope pages commit nothing.
        ProjectionFold::Pending => (3, 2),
        // Two passage-key branches, the generation fence and the fold row's compare-and-set.
        ProjectionFold::Passages => (4, 0),
    };
    statements * meter.seek + returned_rows * (meter.row + statements_per_row * meter.seek)
}

fn assert_projection_receipt_bounds(
    receipts: &[context_graph::test_support::ProjectionUnitReceipt],
) {
    for receipt in receipts {
        let returned_ceiling = match receipt.fold {
            ProjectionFold::Capture | ProjectionFold::Pending => 256,
            ProjectionFold::Passages => 32,
        };
        assert!(
            receipt.returned_rows <= returned_ceiling,
            "returned rows stay inside the fold's fixed batch allowance: {receipt:?}"
        );
        assert!(
            receipt.examined_entries <= unit_allowance(receipt.fold, receipt.returned_rows),
            "examined entries stay within the unit's fixed statement allowance: {receipt:?}"
        );
    }
}

fn hash_membership_rows(rows: &[Vec<Value>]) -> (String, String) {
    use sha2::{Digest, Sha256};
    fn part(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
    }
    let mut ordered = format!(
        "{:x}",
        Sha256::digest(b"context-graph/event-search-ordered-source-set/v1")
    );
    let mut content = format!(
        "{:x}",
        Sha256::digest(b"context-graph/event-search-source-content/v1")
    );
    for row in rows {
        let [
            Value::Int64(ordinal),
            Value::Uuid(event_id),
            Value::Text(connector),
            Value::Text(source),
            Value::Text(session),
            Value::Int64(sequence),
            Value::Text(captured_hash),
            Value::Text(effective_hash),
        ] = row.as_slice()
        else {
            panic!("membership row has the expected typed columns: {row:?}");
        };
        let mut next_ordered = Sha256::new();
        next_ordered.update(b"context-graph/event-search-ordered-source-set/v1");
        part(&mut next_ordered, ordered.as_bytes());
        part(&mut next_ordered, &ordinal.to_be_bytes());
        part(&mut next_ordered, event_id.as_bytes());
        part(&mut next_ordered, connector.as_bytes());
        part(&mut next_ordered, source.as_bytes());
        part(&mut next_ordered, session.as_bytes());
        part(&mut next_ordered, &sequence.to_be_bytes());
        part(&mut next_ordered, captured_hash.as_bytes());
        let mut next_content = Sha256::new();
        next_content.update(b"context-graph/event-search-source-content/v1");
        part(&mut next_content, content.as_bytes());
        part(&mut next_content, effective_hash.as_bytes());
        ordered = format!("{:x}", next_ordered.finalize());
        content = format!("{:x}", next_content.finalize());
    }
    (ordered, content)
}

/// CG-090: a run stopped after two durable units resumes at the saved frontier under a new
/// attempt, never redoes a committed unit, reuses an unchanged completed passage fold and refolds
/// it after the passage population changes.
#[test]
fn projection_cut_resumes_at_the_saved_frontier_under_a_new_attempt() {
    // Measure the engine meter alongside the row; the first bound check waits for it.
    std::thread::spawn(engine_meter);
    // A single 260-event session cannot be one unit. The run stops after two committed units,
    // then resumes under a new attempt and adopts the durable frontier.
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("cg090-large-session.db"));
    let store = open(&cfg);
    let context = create_context(&store);
    append_sparse_projection_stream(&store, context, "large", 260);
    drop(store);

    let embedder = ControlledEmbedder::new();
    set_projection_unit_cut(Some(2));
    let partial = run(&cfg, context, &embedder, 256, OBSERVED_LEASE_MS)
        .expect("a deterministic cut returns the last committed partial result");
    assert!(
        partial.completed_at.is_none(),
        "two units did not finish the whole projection"
    );
    let first = take_projection_unit_receipts();
    assert_eq!(
        first.iter().filter(|row| row.durable).count(),
        2,
        "the cut follows exactly two durable units"
    );
    assert_projection_receipt_bounds(&first);
    let prior_attempt = attempt_owner_row(&cfg, context);

    set_projection_unit_cut(None);
    let resumed = run_until_projection_completes(&cfg, context, &embedder, 256);
    let resumed_attempt = attempt_owner_row(&cfg, context);
    assert_ne!(
        prior_attempt[3], resumed_attempt[3],
        "the saved frontier is adopted under a new attempt token"
    );
    // Every 64th of the 260 events carries text: 1, 65, 129, 193 and 257.
    assert_eq!(resumed.embedded_event_count, 5);
    assert!(resumed.completed_at.is_some());
    assert_eq!(
        open(&cfg)
            .event_history_status(ContextSelector::one(context))
            .unwrap()
            .contexts[0]
            .captured_events,
        260,
        "all source events are captured across the bounded units"
    );
    let continuation = take_projection_unit_receipts();
    assert_projection_receipt_bounds(&continuation);
    let capture_units = first
        .iter()
        .chain(&continuation)
        .filter(|row| row.durable && row.fold == ProjectionFold::Capture)
        .count();
    assert!(
        capture_units >= 2,
        "260 source events require multiple fixed-size capture units: {capture_units}"
    );
    for receipt in &continuation {
        assert!(
            !receipt.durable
                || !first.iter().any(|prior| {
                    prior.durable
                        && prior.fold == receipt.fold
                        && prior.frontier_after == receipt.frontier_after
                }),
            "resume must not redo an already committed frontier: {receipt:?}"
        );
    }
    run(&cfg, context, &embedder, 256, OBSERVED_LEASE_MS)
        .expect("an unchanged completed generation remains readable");
    let unchanged = take_projection_unit_receipts();
    assert!(
        unchanged
            .iter()
            .all(|row| !row.durable && row.fold != ProjectionFold::Passages),
        "an unchanged completed result commits no unit and does not refold passages: {unchanged:?}"
    );

    // Grow the passage population after completion. The old complete fold cannot answer for the
    // changed population; new durable passage work is required.
    let store = open(&cfg);
    let mut later = draft(
        "large-event-261",
        261,
        EventRenderRole::HumanMessage,
        "new population member extends the open passage tail",
    );
    later.session_ref = "one-large-session".into();
    store
        .append_event_batch(AppendEventBatch::live(
            context,
            declaration(),
            "large",
            vec![later],
        ))
        .expect("append to the completed generation's existing source stream");
    drop(store);
    let mut changed_receipts = Vec::new();
    for _ in 0..4 {
        set_projection_unit_cut(Some(1));
        match run(&cfg, context, &embedder, 256, OBSERVED_LEASE_MS) {
            Ok(_) | Err(CgError::ReadActiveTimeExceeded { .. }) => {}
            Err(error) => panic!("changed-passage fold failed: {error:?}"),
        }
        changed_receipts.extend(take_projection_unit_receipts());
        if changed_receipts
            .iter()
            .any(|row| row.durable && row.fold == ProjectionFold::Passages)
        {
            break;
        }
    }
    set_projection_unit_cut(None);
    assert!(
        changed_receipts
            .iter()
            .any(|row| row.durable && row.fold == ProjectionFold::Passages),
        "passage revision change forces a fresh passage fold"
    );
    assert_projection_receipt_bounds(&changed_receipts);
}

/// CG-090: appends to a partly captured stream, to a later stream and a new stream landing while the
/// capture pass is paused inside the first stream stay outside the pass's frozen manifest. The
/// finished pass holds exactly the 520 frozen events, both chains recomputed from its rows match the
/// certified ones, and the next pass captures exactly the three appends.
#[test]
fn capture_reads_only_inside_the_frozen_stream_frontiers() {
    // Measure the engine meter alongside the row; the first bound check waits for it.
    std::thread::spawn(engine_meter);
    let embedder = ControlledEmbedder::new();
    // Stream a and stream z each hold 260 events, so stream a cannot be captured in one unit. The
    // run stops once stream a is partly captured, with both frontiers frozen in the manifest. Then
    // sequence 261 lands on stream a, sequence 261 on stream z and a new stream m (ordered between
    // a and z) gets its first event. The frozen pass must finish with exactly the 520 events at or
    // below its frontiers; the next pass picks the three appends up.
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("cg090-multistream.db"));
    let store = open(&cfg);
    let context = create_context(&store);
    append_sparse_projection_stream(&store, context, "stream-a", 260);
    append_sparse_projection_stream(&store, context, "stream-z", 260);
    drop(store);
    let member_query = "SELECT ordinal, event_id, connector_id, source_instance_ref, session_ref, source_sequence, captured_source_sha256, effective_source_sha256 FROM event_search_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal";
    let generation_params = |generation_id: context_graph::EventSearchGenerationId| {
        HashMap::from([
            ("context_id".to_string(), Value::Uuid(context.as_uuid())),
            (
                "generation_id".to_string(),
                Value::Uuid(generation_id.as_uuid()),
            ),
        ])
    };
    let source_and_sequence = |row: &Vec<Value>| match row.as_slice() {
        [_, _, _, Value::Text(source), _, Value::Int64(sequence), ..] => {
            (source.clone(), *sequence)
        }
        other => panic!("captured membership exposes its source and sequence: {other:?}"),
    };
    let mut paused: Option<(context_graph::EventSearchGenerationId, usize)> = None;
    let mut manifest = HashMap::<String, i64>::new();
    for _ in 0..8 {
        set_projection_unit_cut(Some(1));
        let partial = run(&cfg, context, &embedder, 256, OBSERVED_LEASE_MS)
            .expect("single-unit cut leaves a resumable capture");
        assert_projection_receipt_bounds(&take_projection_unit_receipts());
        let store = open(&cfg);
        let params = generation_params(partial.generation_id);
        let captured_a = store
            .execute_sql_for_test(member_query, &params)
            .expect("read the captured membership")
            .rows
            .iter()
            .map(&source_and_sequence)
            .filter(|(source, _)| source == "stream-a")
            .count();
        if (1..260).contains(&captured_a) {
            manifest = store
                .execute_sql_for_test(
                    "SELECT source_instance_ref, upper_seq FROM event_search_capture_manifest WHERE context_id = $context_id AND generation_id = $generation_id",
                    &params,
                )
                .expect("read the committed stream frontiers")
                .rows
                .into_iter()
                .map(|row| match row.as_slice() {
                    [Value::Text(source), Value::Int64(upper)] => (source.clone(), *upper),
                    other => panic!("manifest frontier has typed source and sequence: {other:?}"),
                })
                .collect();
            paused = Some((partial.generation_id, captured_a));
            break;
        }
    }
    let (generation_id, captured_before) =
        paused.expect("capture pauses inside stream a within bounded units");
    assert_eq!(
        manifest,
        HashMap::from([("stream-a".to_string(), 260), ("stream-z".to_string(), 260)]),
        "both streams' frontiers are frozen before the appends"
    );

    let store = open(&cfg);
    let mut excluded = Vec::new();
    for (source, ordinal) in [("stream-a", 261), ("stream-z", 261), ("stream-m", 1)] {
        let mut appended = draft(
            &format!("{source}-after-frontier"),
            ordinal,
            EventRenderRole::HumanMessage,
            "post-frontier append",
        );
        appended.session_ref = "one-large-session".into();
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                source,
                vec![appended],
            ))
            .expect("append while capture is paused inside stream a");
        excluded.push((source.to_string(), ordinal as i64));
    }
    drop(store);

    // Finish the frozen pass and nothing more: one durable unit at a time until the pass certifies
    // (its interpretation stamp is published). A run left uncut would go straight on into the next
    // capture pass, which picks the appends up.
    let mut certified = false;
    for _ in 0..16 {
        set_projection_unit_cut(Some(1));
        match run(&cfg, context, &embedder, 256, OBSERVED_LEASE_MS) {
            Ok(receipt) => assert_eq!(receipt.generation_id, generation_id),
            Err(CgError::ReadActiveTimeExceeded { .. }) => {}
            Err(error) => panic!("the frozen pass resumes: {error:?}"),
        }
        assert_projection_receipt_bounds(&take_projection_unit_receipts());
        let stamp = open(&cfg)
            .execute_sql_for_test(
                "SELECT interpretation_verified_stamp FROM event_search_generations WHERE id = $generation_id",
                &generation_params(generation_id),
            )
            .expect("read the generation's certification")
            .rows;
        if stamp.first().is_some_and(|row| row[0] != Value::Null) {
            certified = true;
            break;
        }
    }
    set_projection_unit_cut(None);
    assert!(certified, "the frozen pass certifies within bounded units");
    let store = open(&cfg);
    let members = store
        .execute_sql_for_test(member_query, &generation_params(generation_id))
        .expect("read the frozen pass's membership")
        .rows;
    let oracle = ["stream-a", "stream-z"]
        .into_iter()
        .flat_map(|source| (1..=260).map(move |sequence| (source.to_string(), sequence)))
        .collect::<Vec<_>>();
    assert_eq!(
        members.iter().map(&source_and_sequence).collect::<Vec<_>>(),
        oracle,
        "the frozen pass holds exactly the 520 events at or below its frontiers, in key order \
         (stream a was paused after {captured_before} members)"
    );
    for (index, row) in members.iter().enumerate() {
        assert_eq!(
            row[0],
            Value::Int64(index as i64),
            "ordinals are contiguous"
        );
    }
    let (ordered_hash, content_hash) = hash_membership_rows(&members);
    assert_eq!(
        store
            .execute_sql_for_test(
                "SELECT capture_ordinal, capture_ordered_sha256, capture_content_sha256 FROM event_search_generations WHERE id = $generation_id",
                &generation_params(generation_id),
            )
            .expect("read the generation's certified capture")
            .rows,
        vec![vec![
            Value::Int64(520),
            Value::Text(ordered_hash),
            Value::Text(content_hash)
        ]],
        "both chains, recomputed from the 520 member rows, equal the certified ones"
    );
    drop(store);

    // The excluded appends were deferred, not lost: the next pass captures exactly them, after the
    // frozen 520, in key order (a, m, z).
    run_until_projection_completes(&cfg, context, &embedder, 256);
    take_projection_unit_receipts();
    let later = open(&cfg)
        .execute_sql_for_test(member_query, &generation_params(generation_id))
        .expect("read the next pass's membership")
        .rows;
    assert_eq!(later.len(), 523);
    assert_eq!(
        later[520..]
            .iter()
            .map(&source_and_sequence)
            .collect::<Vec<_>>(),
        vec![
            excluded[0].clone(),
            excluded[2].clone(),
            excluded[1].clone()
        ],
    );
}

fn flatness_population(
    prefix_events: u64,
) -> Vec<context_graph::test_support::ProjectionUnitReceipt> {
    let embedder = ControlledEmbedder::new();
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join(format!("cg090-flat-{prefix_events}.db")));
    let store = open(&cfg);
    let context = create_context(&store);
    // A completed prefix that grows N/2N/4N, then an identical tail stream and an identical
    // metadata stream whose 1.6 MB events force byte-split capture pages.
    for (source, count, every) in [("a-prefix", prefix_events, 8_u64), ("z-tail", 64, 2)] {
        let events = (1..=count)
            .map(|ordinal| {
                let mut event = draft(
                    &format!("{source}-event-{ordinal}"),
                    ordinal,
                    EventRenderRole::HumanMessage,
                    "short searchable event",
                );
                event.session_ref = "one-large-session".into();
                if ordinal % every != 1 {
                    event.text = None;
                    event.render_role = EventRenderRole::Metadata;
                }
                event
            })
            .collect();
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                source,
                events,
            ))
            .expect("append flatness fixture stream");
    }
    append_large_metadata_stream(&store, context, "zz-meta", 1);
    drop(store);
    let _ = take_projection_unit_receipts();
    run_until_projection_completes(&cfg, context, &embedder, 256);
    take_projection_unit_receipts()
}

/// CG-090 proof row 2: the work of a unit over the identical tail, including byte-split capture
/// pages, the last discovery unit and the last full passage unit, stays flat while the completed
/// population grows N/2N/4N. A unit that scanned the completed prefix would add at least the
/// prefix's growth to these counts. The three populations run on their own threads.
#[test]
fn unit_work_stays_flat_as_the_population_grows_n_2n_4n() {
    // Same-shape units measured 0-1 entries apart across populations on the pinned engine; a read
    // proportional to the prefix adds at least 3N/8 = 24 entries at 4N even over the sparse
    // prefix's passages, and at least one examined row per prefix row on a rows-only meter.
    const FLAT_TOLERANCE: u64 = 16;
    let sizes = [64_u64, 128, 256];
    let meter = std::thread::spawn(engine_meter);
    let receipts = sizes
        .map(|size| std::thread::spawn(move || flatness_population(size)))
        .map(|handle| handle.join().expect("flatness population thread"));
    meter.join().expect("engine meter thread");
    for population in &receipts {
        assert_projection_receipt_bounds(population);
    }
    let tail_capture = |population: &[context_graph::test_support::ProjectionUnitReceipt]| {
        population
            .iter()
            .filter(|row| {
                row.durable
                    && row.fold == ProjectionFold::Capture
                    && (row.frontier_after.contains("\"z-tail\"")
                        || row.frontier_after.contains("\"zz-meta\""))
            })
            .map(|row| (row.returned_rows, row.examined_entries))
            .collect::<Vec<_>>()
    };
    let byte_split_pages = receipts[0]
        .iter()
        .filter(|row| {
            row.durable
                && row.fold == ProjectionFold::Capture
                && row.frontier_after.contains("\"zz-meta\", ")
        })
        .count();
    assert!(
        byte_split_pages >= 2,
        "the 4.8 MB metadata stream is captured in more than one byte-bounded page: {:?}",
        receipts[0]
    );
    let last_discovery = |population: &[context_graph::test_support::ProjectionUnitReceipt]| {
        population
            .iter()
            .filter(|row| row.fold == ProjectionFold::Pending && row.returned_rows > 0)
            .map(|row| (row.returned_rows, row.examined_entries))
            .next_back()
            .expect("a discovery unit found pending work")
    };
    // The first passage unit seeks with no saved key; later units continue after one, so the
    // comparison takes the last full continuing unit.
    let last_full_passages =
        |population: &[context_graph::test_support::ProjectionUnitReceipt]| {
            let full = population
                .iter()
                .filter(|row| row.fold == ProjectionFold::Passages)
                .map(|row| row.returned_rows)
                .max()
                .expect("a passage unit ran");
            population
                .iter()
                .filter(|row| {
                    row.fold == ProjectionFold::Passages
                        && row.returned_rows == full
                        && row.frontier_before != "None"
                })
                .map(|row| (row.returned_rows, row.examined_entries))
                .next_back()
                .expect("a full passage unit continued after a saved key")
        };
    let within = |label: &str, values: Vec<(u64, u64)>| {
        assert!(
            values.windows(2).all(|pair| pair[0].0 == pair[1].0),
            "{label}: the compared units have the same shape at N/2N/4N: {values:?}"
        );
        let low = values.iter().map(|value| value.1).min().unwrap();
        let high = values.iter().map(|value| value.1).max().unwrap();
        assert!(
            high - low <= FLAT_TOLERANCE,
            "{label}: examined entries stay flat at N/2N/4N: {values:?}"
        );
    };
    let tails = receipts
        .iter()
        .map(|population| tail_capture(population))
        .collect::<Vec<_>>();
    assert!(
        tails.windows(2).all(|pair| pair[0].len() == pair[1].len()),
        "the identical tail is captured in the same units at N/2N/4N: {tails:?}"
    );
    for position in 0..tails[0].len() {
        within(
            &format!("tail capture unit {position}"),
            tails.iter().map(|units| units[position]).collect(),
        );
    }
    within(
        "last discovery unit",
        receipts
            .iter()
            .map(|population| last_discovery(population))
            .collect(),
    );
    within(
        "last full passage unit",
        receipts
            .iter()
            .map(|population| last_full_passages(population))
            .collect(),
    );
}

/// CG-090: a duplicate current passage ordinal is refused on sight by one non-durable passage unit.
#[test]
fn a_duplicate_current_passage_ordinal_is_refused_inside_one_unit() {
    // Measure the engine meter alongside the row; the first bound check waits for it.
    std::thread::spawn(engine_meter);
    let embedder = ControlledEmbedder::new();
    // Passage ordinals are generation-wide, so create one duplicate directly in the current
    // passage index and advance the publication revision to force the completed fold to recheck.
    let duplicate_root = tempfile::tempdir().unwrap();
    let duplicate_cfg = config(duplicate_root.path().join("cg090-duplicate.db"));
    let duplicate_store = open(&duplicate_cfg);
    let duplicate_context = create_context(&duplicate_store);
    append_projection_fixture_stream(
        &duplicate_store,
        duplicate_context,
        "duplicate-source",
        80,
        "duplicate fanout text",
    );
    drop(duplicate_store);
    let duplicate_generation =
        run_until_projection_completes(&duplicate_cfg, duplicate_context, &embedder, 256)
            .generation_id;
    let duplicate_store = open(&duplicate_cfg);
    let source_passage = duplicate_store
        .execute_sql_for_test(
            "SELECT context_id, generation_id, connector_id, session_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_searchable, created_at FROM event_search_passages WHERE generation_id = $generation_id AND is_current = true ORDER BY passage_ordinal LIMIT 1",
            &HashMap::from([(
                "generation_id".into(),
                Value::Uuid(duplicate_generation.as_uuid()),
            )]),
        )
        .expect("read the first current passage")
        .rows
        .remove(0);
    let mut duplicate_values = [
        "context_id",
        "generation_id",
        "connector_id",
        "session_ref",
        "passage_ordinal",
        "first",
        "last",
        "text",
        "sha",
        "vector",
        "dimension",
        "searchable",
        "created_at",
    ]
    .into_iter()
    .zip(source_passage)
    .map(|(name, value)| (name.to_string(), value))
    .collect::<HashMap<_, _>>();
    duplicate_values.insert("new_id".into(), Value::Uuid(Uuid::new_v4()));
    duplicate_store
        .execute_sql_for_test(
            "INSERT INTO event_search_passages (id, context_id, generation_id, connector_id, session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_current, is_searchable, created_at) VALUES ($new_id, $context_id, $generation_id, $connector_id, $session_ref, 'duplicate-injected', $passage_ordinal, $first, $last, $text, $sha, $vector, $dimension, true, $searchable, $created_at)",
            &duplicate_values,
        )
        .expect("inject one current passage with a duplicate generation ordinal");
    duplicate_store
        .execute_sql_for_test(
            "UPDATE event_search_generations SET passage_revision = passage_revision + 1 WHERE id = $generation_id",
            &HashMap::from([(
                "generation_id".into(),
                Value::Uuid(duplicate_generation.as_uuid()),
            )]),
        )
        .expect("advance the passage publication revision with the injected duplicate");
    drop(duplicate_store);
    let _setup_receipts = take_projection_unit_receipts();
    assert!(
        run(
            &duplicate_cfg,
            duplicate_context,
            &embedder,
            256,
            OBSERVED_LEASE_MS
        )
        .is_err(),
        "equal ordinals across sources are refused at their second occurrence"
    );
    let duplicate_receipts = take_projection_unit_receipts();
    let duplicate = duplicate_receipts
        .iter()
        .find(|row| row.fold == ProjectionFold::Passages)
        .expect("the refusing passage unit reports its examined entries");
    assert!(
        !duplicate.durable,
        "the duplicate passage unit is refused before commit"
    );
    assert!(
        duplicate.returned_rows <= 32,
        "the refusing unit reads at most one passage page: {duplicate:?}"
    );
    assert!(
        duplicate.examined_entries
            <= unit_allowance(ProjectionFold::Passages, duplicate.returned_rows),
        "duplicate refusal stays within one passage unit's statement allowance: {duplicate:?}"
    );
}

/// CG-090: forget purges the generation's fold progress and frozen manifest, and a unit in flight
/// across the forget is refused at its commit with the stale claim.
#[test]
fn forget_purges_fold_progress_and_fences_an_in_flight_unit() {
    // Measure the engine meter alongside the row; the first bound check waits for it.
    std::thread::spawn(engine_meter);
    let embedder = ControlledEmbedder::new();
    // Forget the selected session after a durable unit. It purges the partial generation and both
    // persisted fold inputs, so a later call cannot adopt stale progress.
    let forget_root = tempfile::tempdir().unwrap();
    let forget_cfg = config(forget_root.path().join("cg090-forget.db"));
    let forget_store = open(&forget_cfg);
    let forget_context = create_context(&forget_store);
    append_sparse_projection_stream(&forget_store, forget_context, "forget", 300);
    drop(forget_store);
    set_projection_unit_cut(Some(1));
    let partial = run(
        &forget_cfg,
        forget_context,
        &embedder,
        256,
        OBSERVED_LEASE_MS,
    )
    .unwrap();
    let before_forget = take_projection_unit_receipts();
    assert_eq!(before_forget.iter().filter(|row| row.durable).count(), 1);
    let forget_store = open(&forget_cfg);
    forget_store
        .forget_event_history(ForgetSelection::Session {
            context_id: forget_context,
            connector_id: CONNECTOR.into(),
            session_ref: "one-large-session".into(),
        })
        .unwrap();
    let remaining_progress = forget_store
        .execute_sql_for_test(
            "SELECT id FROM event_search_fold_progress WHERE context_id = $context_id AND generation_id = $generation_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(forget_context.as_uuid())),
                (
                    "generation_id".into(),
                    Value::Uuid(partial.generation_id.as_uuid()),
                ),
            ]),
        )
        .unwrap()
        .rows;
    let remaining_manifest = forget_store
        .execute_sql_for_test(
            "SELECT connector_id FROM event_search_capture_manifest WHERE context_id = $context_id AND generation_id = $generation_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(forget_context.as_uuid())),
                (
                    "generation_id".into(),
                    Value::Uuid(partial.generation_id.as_uuid()),
                ),
            ]),
        )
        .unwrap()
        .rows;
    assert!(remaining_progress.is_empty());
    assert!(remaining_manifest.is_empty());
    drop(forget_store);
    set_projection_unit_cut(None);

    // Forget while a unit is in flight. The pause seam stops the run inside the named unit after its
    // reads, the forget runs there, and the unit then continues into its own production commit with
    // the generation, attempt, fold row and manifest frontier it read under. Its fences must refuse
    // it; were they gone, the unit would commit into the forgotten generation. Unit 1 is the
    // manifest unit, whose only fence is the generation re-read (it has no prior fold row and no
    // frozen frontier yet); unit 2 is the first capture page, fenced by the generation, its fold
    // row's compare-and-set and its manifest frontier.
    for paused_unit in [1, 2] {
        let fence_root = tempfile::tempdir().unwrap();
        let fence_cfg = config(fence_root.path().join("cg090-in-flight-forget.db"));
        let fence_store = open(&fence_cfg);
        let fence_context = create_context(&fence_store);
        append_sparse_projection_stream(&fence_store, fence_context, "in-flight", 300);
        drop(fence_store);
        let forgotten = Rc::new(RefCell::new(None));
        let forgotten_in_flight = forgotten.clone();
        pause_projection_unit_before_commit(paused_unit, move |store| {
            let receipt = store
                .forget_event_history(ForgetSelection::Session {
                    context_id: fence_context,
                    connector_id: CONNECTOR.into(),
                    session_ref: "one-large-session".into(),
                })
                .expect("forget the session while a unit is in flight");
            *forgotten_in_flight.borrow_mut() = Some(receipt.events);
        });
        match run(&fence_cfg, fence_context, &embedder, 256, OBSERVED_LEASE_MS) {
            Err(CgError::Engine(reason)) => assert!(
                reason.starts_with("event-search claim refused:"),
                "unit {paused_unit} is refused with the stale claim: {reason}"
            ),
            other => panic!("unit {paused_unit} must be fenced after forget: {other:?}"),
        }
        assert_eq!(
            *forgotten.borrow(),
            Some(300),
            "the forget ran inside unit {paused_unit}"
        );
        let fenced = take_projection_unit_receipts();
        assert_eq!(
            fenced.iter().filter(|row| row.durable).count(),
            paused_unit as usize - 1,
            "only the units before the paused one committed: {fenced:?}"
        );
        assert!(
            fenced
                .last()
                .is_some_and(|row| !row.durable && row.frontier_after == "refused"),
            "the paused unit's commit was refused: {fenced:?}"
        );
        let fence_store = open(&fence_cfg);
        let params = HashMap::from([("context_id".into(), Value::Uuid(fence_context.as_uuid()))]);
        for table in [
            "event_search_fold_progress",
            "event_search_capture_manifest",
            "event_search_source_events",
            "event_search_generations",
        ] {
            assert!(
                fence_store
                    .execute_sql_for_test(
                        &format!("SELECT context_id FROM {table} WHERE context_id = $context_id"),
                        &params,
                    )
                    .unwrap()
                    .rows
                    .is_empty(),
                "the refused unit {paused_unit} left nothing in {table}"
            );
        }
        drop(fence_store);
    }
}

const CG090_INTERPRETATION: &str = "cg090-released-v1";

fn cg090_released() -> context_graph::ReleasedInterpretationDeclaration {
    context_graph::ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: CG090_INTERPRETATION.into(),
        target_adapter_version: "correctness-v1".into(),
        target_anchor_declaration_version: "correctness-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "correctness-v1".into(),
            anchor_declaration_version: "correctness-v1".into(),
        }],
        mapping_contract_id: "cg090-interpretation-contract".into(),
        mapping_contract_sha256: "cg090-interpretation-contract-v1".into(),
    }
}

/// Keeps every captured field and gives each event one interpreted anchor, so every captured event
/// has an interpretation member and an effective value the capture must read.
struct Cg090Mapper(context_graph::ReleasedInterpretationDeclaration);

impl context_graph::ReleasedConnectorInterpretation for Cg090Mapper {
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

struct Cg090Contract(context_graph::ReleasedInterpretationDeclaration);

impl context_graph::InstalledInterpretationContract for Cg090Contract {
    fn declaration(&self) -> &context_graph::ReleasedInterpretationDeclaration {
        &self.0
    }

    fn validate(
        &self,
        _captured: &context_graph::CapturedEventAndAnchors,
        _candidate: &context_graph::EffectiveEventFields,
    ) -> Result<()> {
        Ok(())
    }
}

/// CG-090: a capture unit's receipt charges every statement the unit runs, including the two
/// interpretation reads per interpreted event and the commit transaction's fence reads and member
/// writes, and still fits the fixed per-unit allowance. The same 120 events are captured twice, in 64-event units,
/// once plainly and once through a released interpretation; the interpreted capture pages must be
/// charged at least the two extra reads per event on top of the plain ones.
#[test]
fn interpreted_capture_units_charge_every_statement_within_the_allowance() {
    std::thread::spawn(engine_meter);
    let embedder = ControlledEmbedder::new();
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("cg090-interpreted.db"));
    let store = open(&cfg);
    let context = create_context(&store);
    let source_streams = [("interpreted", 120)];
    let captured_events = source_streams.iter().map(|(_, events)| *events as u64).sum::<u64>();
    for (session, events) in source_streams {
        append_sparse_projection_stream(&store, context, session, events);
    }
    drop(store);
    take_projection_unit_receipts();
    run_until_projection_completes(&cfg, context, &embedder, 64);
    let plain = take_projection_unit_receipts();
    assert_projection_receipt_bounds(&plain);

    let mapper: Arc<dyn context_graph::ReleasedConnectorInterpretation> =
        Arc::new(Cg090Mapper(cg090_released()));
    let contract: Arc<dyn context_graph::InstalledInterpretationContract> =
        Arc::new(Cg090Contract(cg090_released()));
    let interpretation = context_graph::ConnectorInterpretationRunner::build(
        context_graph::InterpretationBuildRequest {
            store: cfg.clone(),
            context_id: context,
            connector_id: CONNECTOR.into(),
            target_interpretation_version: CG090_INTERPRETATION.into(),
            mapper,
            contract,
            max_events_per_batch: NonZeroU32::new(64).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
            lease_duration_ms: NonZeroU64::new(OBSERVED_LEASE_MS).unwrap(),
        },
    )
    .expect("build the released interpretation over all 120 events");
    take_projection_unit_receipts();
    let store = open(&cfg);
    store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id: context,
            interpretation_generation_id: Some(interpretation.generation_id),
            embedder: embedder.registration(),
            profile: profile(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(64).unwrap(),
            max_passages_per_batch: NonZeroU32::new(16).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(32_768).unwrap(),
            lease_duration_ms: NonZeroU64::new(OBSERVED_LEASE_MS).unwrap(),
        })
        .expect("stage the interpreted search generation");
    drop(store);
    let interpreted = take_projection_unit_receipts();
    assert_projection_receipt_bounds(&interpreted);

    // The first capture page also reads the fixture's stream manifest. Charge
    // interpretation reads to the events only; membership pages return no events.
    let pages = |receipts: &[context_graph::test_support::ProjectionUnitReceipt]| {
        receipts
            .iter()
            .filter(|row| {
                row.durable && row.fold == ProjectionFold::Capture && row.returned_rows > 1
            })
            .enumerate()
            .map(|(page, row)| {
                let manifest_rows = if page == 0 { source_streams.len() as u64 } else { 0 };
                (row.returned_rows - manifest_rows, row.examined_entries)
            })
            .collect::<Vec<_>>()
    };
    let plain_pages = pages(&plain);
    let interpreted_pages = pages(&interpreted);
    assert!(
        !interpreted_pages.is_empty() && interpreted_pages.len() == plain_pages.len(),
        "both captures read the same pages: {plain_pages:?} vs {interpreted_pages:?}"
    );
    assert_eq!(
        plain_pages.iter().map(|(events, _)| events).sum::<u64>(),
        captured_events,
        "capture pages account for every event in the fixture, excluding manifests"
    );
    for ((plain_rows, plain_examined), (rows, examined)) in
        plain_pages.iter().zip(&interpreted_pages)
    {
        assert_eq!(plain_rows, rows, "the same events per page");
        assert_eq!(
            examined - plain_examined,
            2 * rows,
            "an interpreted page is charged its two interpretation reads per event: \
             {examined} vs plain {plain_examined} over {rows} events"
        );
    }
}

#[test]
fn appending_a_session_folds_only_new_passages_at_n_and_four_n_history() {
    use sha2::{Digest, Sha256};
    fn population(history: u64) -> (u64, u64) {
        let root = tempfile::tempdir().unwrap();
        let cfg = config(root.path().join("append-fold.db"));
        let store = open(&cfg);
        let context = create_context(&store);
        let body = "short searchable event";
        let events = (1..=history)
            .map(|ordinal| {
                draft(
                    &format!("prefix-{ordinal}"),
                    ordinal,
                    EventRenderRole::HumanMessage,
                    body,
                )
            })
            .collect();
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                "prefix",
                events,
            ))
            .unwrap();
        drop(store);
        let embedder = ControlledEmbedder::new();
        let before = run_until_projection_completes(&cfg, context, &embedder, 256);
        take_projection_unit_receipts();
        // A different session appends generation ordinals without retiring the saved prefix.
        let store = open(&cfg);
        let events = (1..=8)
            .map(|ordinal| {
                let mut event = draft(
                    &format!("new-{ordinal}"),
                    ordinal,
                    EventRenderRole::HumanMessage,
                    body,
                );
                event.session_ref = "new-session".into();
                event
            })
            .collect();
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                "new-source",
                events,
            ))
            .unwrap();
        drop(store);
        let after = run_until_projection_completes(&cfg, context, &embedder, 256);
        assert_eq!(before.generation_id, after.generation_id);
        let receipts = take_projection_unit_receipts();
        let examined = receipts
            .iter()
            .filter(|row| row.fold == ProjectionFold::Passages)
            .map(|row| row.examined_entries)
            .sum();
        let returned = receipts
            .iter()
            .filter(|row| row.fold == ProjectionFold::Passages)
            .map(|row| row.returned_rows)
            .sum();
        // Reopen and independently hash the entire current set: prefix reuse must preserve
        // the established digest, not merely skip its verification.
        let store = open(&cfg);
        let rows = store.execute_sql_for_test(
            "SELECT passage_ordinal, content_sha256 FROM event_search_passages WHERE generation_id = $generation_id AND is_current = true ORDER BY passage_ordinal ASC, id ASC",
            &HashMap::from([("generation_id".into(), Value::Uuid(after.generation_id.as_uuid()))]),
        ).unwrap().rows;
        let mut digest = Sha256::new();
        digest.update(b"context-graph/event-search-generation/v1");
        for row in &rows {
            let [Value::Int64(ordinal), Value::Text(content)] = row.as_slice() else {
                panic!("typed passage: {row:?}");
            };
            let ordinal = u64::try_from(*ordinal).unwrap().to_be_bytes();
            for bytes in [ordinal.as_slice(), content.as_bytes()] {
                digest.update(u64::try_from(bytes.len()).unwrap().to_be_bytes());
                digest.update(bytes);
            }
        }
        assert_eq!(
            after.generation_content_sha256,
            format!("{:x}", digest.finalize())
        );
        assert_eq!(after.passage_count, rows.len() as u64);
        eprintln!(
            "append_fold history={history} prefix_passages={} examined={examined} returned={returned}",
            before.passage_count
        );
        (examined, returned)
    }
    let receipts = [64, 256].map(|history| {
        std::thread::spawn(move || population(history))
            .join()
            .unwrap()
    });
    assert_eq!(
        receipts[0].1, receipts[1].1,
        "a fixed new session folds the same suffix at N and 4N: {receipts:?}"
    );
    assert!(
        receipts[0].0.abs_diff(receipts[1].0) <= 32,
        "append fold examined work is bounded by the suffix and index height: {receipts:?}"
    );
}

#[test]
fn passage_completion_keeps_durable_search_while_replacement_is_pending() {
    let _clock = Wallclock::test_clock_guard(|| 1_800_000_000_000);
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path().join("completion-pending.db"));
    cfg.write_memory_limit_bytes = NonZeroUsize::new(64 * 1024 * 1024).unwrap();
    let store = open(&cfg);
    let db = store.sync_database();
    db.set_maintenance_policy(contextdb_engine::MaintenancePolicy::CallerDriven);
    // The completion coordinator handles the same engine routes regardless
    // of dimension; use a small synthetic route to isolate its lifecycle.
    db.execute("CREATE TABLE completion_points (id INT PRIMARY KEY, bucket INT NOT NULL, embedding VECTOR(3) PARTITION_KEY (bucket) MAX_PARTITIONS 2 SEARCH_MODE INDEXED AUTO_INDEX_AT 1 HNSW (M = 8, EF_CONSTRUCTION = 32, EF_SEARCH = 512))", &HashMap::new()).unwrap();
    let append = |id: i64| {
        let vector = if id == 10000 {
            vec![1.0, 0.0, 0.0]
        } else {
            vec![0.0, 1.0, id as f32 + 1.0]
        };
        db.execute(
            "INSERT INTO completion_points VALUES ($id, 1, $vector)",
            &HashMap::from([
                ("id".into(), Value::Int64(id)),
                ("vector".into(), Value::Vector(vector)),
            ]),
        )
        .unwrap();
    };
    for id in 0..16 {
        append(id);
    }
    store.maintain_event_search_vectors_for_test().unwrap();
    for id in 16..2064 {
        append(id);
    }
    append(10000);
    store.maintain_event_search_vectors_for_test().unwrap();
    let status = db
        .execute(
            "SHOW VECTOR_PARTITIONS FOR completion_points.embedding",
            &HashMap::new(),
        )
        .unwrap();
    let status: HashMap<_, _> = status
        .columns
        .into_iter()
        .zip(status.rows.into_iter().next().unwrap())
        .collect();
    assert_eq!(status["query_state"], Value::Text("ready".into()));
    assert_eq!(
        status["maintenance_state"],
        Value::Text("compacting".into())
    );
    assert!(matches!(status["maintenance_vectors_remaining"], Value::Int64(n) if n > 0));
    drop(store);
    let store = open(&cfg);
    let db = store.sync_database();
    db.set_maintenance_policy(contextdb_engine::MaintenancePolicy::CallerDriven);
    let params = HashMap::from([("q".into(), Value::Vector(vec![1.0, 0.0, 0.0]))]);
    let query = "SELECT id FROM completion_points WHERE bucket = 1 ORDER BY embedding <=> $q USE VECTOR INDEXED LIMIT 1";
    assert_eq!(
        db.execute(query, &params).unwrap().rows,
        vec![vec![Value::Int64(10000)]]
    );
    let mut landed = false;
    for _ in 0..128 {
        let report = db.run_maintenance_cycle().unwrap().vector;
        assert!(report.first_failure.is_none(), "{:?}", report.first_failure);
        let status = db
            .execute(
                "SHOW VECTOR_PARTITIONS FOR completion_points.embedding",
                &HashMap::new(),
            )
            .unwrap();
        let status: HashMap<_, _> = status
            .columns
            .into_iter()
            .zip(status.rows.into_iter().next().unwrap())
            .collect();
        if status["maintenance_state"] == Value::Text("idle".into()) {
            landed = true;
            break;
        }
    }
    assert!(landed);
    assert_eq!(
        db.execute(query, &params).unwrap().rows,
        vec![vec![Value::Int64(10000)]]
    );
}
