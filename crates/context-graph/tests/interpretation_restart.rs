//! regression regressions for durable interpretation-build restart behavior.
//!
//! The fixtures capture real events through the public Store API.  Mapper callbacks only inspect
//! the generation and sealed membership that the public runner has already created.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use context_graph::{
    AppendEventBatch, CapturedEventAndAnchors, CgError, ConnectorInterpretationRunner,
    ContextSelector, ConversationEvent, CreateContext, EffectiveEventFields,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole,
    InstalledInterpretationContract, InterpretationBuildRequest, InterpretationGenerationId,
    InterpretationGenerationState, ReleasedConnectorInterpretation,
    ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "interpretation-restart-red";
const TARGET: &str = "released-restart-v1";
const SENSITIVE_SENTINEL: &str = "UNIQUE-SENSITIVE-INTERPRETATION-FAILURE-7f93d4";

fn config(path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn capture_declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "capture-restart-v1".into(),
        anchor_declaration_version: "anchors-restart-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn release() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        target_adapter_version: "capture-restart-v1".into(),
        target_anchor_declaration_version: "anchors-restart-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "capture-restart-v1".into(),
            anchor_declaration_version: "anchors-restart-v1".into(),
        }],
        mapping_contract_id: "restart-red-contract".into(),
        mapping_contract_sha256: "restart-red-contract-sha256".into(),
    }
}

fn effective() -> EffectiveEventFields {
    EffectiveEventFields {
        occurred_at: None,
        author: "released-provider".into(),
        kind: "released-message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("released interpretation".into()),
        payload: json!({"released": true}),
        anchors: json!([{
            "key": "session",
            "canonical_value": "restart-session",
            "original_value": "restart-session"
        }]),
        coverage: EventCoverage::Complete,
    }
}

type MapperControl = Box<dyn Fn() + Send + Sync>;

struct FixtureMapper {
    declaration: ReleasedInterpretationDeclaration,
    control: Mutex<Option<MapperControl>>,
    control_call: Option<usize>,
    failure: Option<(usize, String)>,
    calls: AtomicUsize,
    called_events: Mutex<Vec<context_graph::EventId>>,
}

impl FixtureMapper {
    fn succeeds() -> Self {
        Self {
            declaration: release(),
            control: Mutex::new(None),
            control_call: None,
            failure: None,
            calls: AtomicUsize::new(0),
            called_events: Mutex::new(Vec::new()),
        }
    }

    fn fails_on_call(
        failure_call: usize,
        control: impl FnOnce() + Send + 'static,
        failure: &str,
    ) -> Self {
        let control = Mutex::new(Some(control));
        Self {
            declaration: release(),
            control: Mutex::new(Some(Box::new(move || {
                if let Some(control) = control.lock().expect("one-shot control lock").take() {
                    control();
                }
            }))),
            control_call: Some(failure_call),
            failure: Some((failure_call, failure.into())),
            calls: AtomicUsize::new(0),
            called_events: Mutex::new(Vec::new()),
        }
    }

    fn succeeds_after(control: impl FnOnce() + Send + 'static) -> Self {
        let control = Mutex::new(Some(control));
        Self {
            declaration: release(),
            control: Mutex::new(Some(Box::new(move || {
                if let Some(control) = control.lock().expect("one-shot control lock").take() {
                    control();
                }
            }))),
            control_call: Some(1),
            failure: None,
            calls: AtomicUsize::new(0),
            called_events: Mutex::new(Vec::new()),
        }
    }

    fn configured_failure(&self) -> Option<&str> {
        self.failure.as_ref().map(|(_, failure)| failure.as_str())
    }

    fn called_events(&self) -> Vec<context_graph::EventId> {
        self.called_events
            .lock()
            .expect("mapper event list lock")
            .clone()
    }
}

impl ReleasedConnectorInterpretation for FixtureMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn interpret_event(&self, event: &ConversationEvent) -> Result<EffectiveEventFields> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        self.called_events
            .lock()
            .expect("mapper event list lock")
            .push(event.id);
        if self.control_call == Some(call) {
            let control = self.control.lock().expect("mapper control lock").take();
            if let Some(control) = control {
                control();
            }
        }
        if let Some((failure_call, failure)) = &self.failure
            && *failure_call == call
        {
            return Err(CgError::Engine(failure.clone()));
        }
        Ok(effective())
    }
}

struct FixtureContract;

impl InstalledInterpretationContract for FixtureContract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        static DECLARATION: std::sync::OnceLock<ReleasedInterpretationDeclaration> =
            std::sync::OnceLock::new();
        DECLARATION.get_or_init(release)
    }

    fn validate(
        &self,
        captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        assert_eq!(captured.event.connector_id, CONNECTOR);
        assert_eq!(candidate, &effective());
        Ok(())
    }
}

fn request(
    path: PathBuf,
    context_id: context_graph::ContextId,
    mapper: Arc<FixtureMapper>,
) -> InterpretationBuildRequest {
    InterpretationBuildRequest {
        store: config(path),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        mapper,
        contract: Arc::new(FixtureContract),
        max_events_per_batch: NonZeroU32::new(1).expect("nonzero batch size"),
        max_bytes_per_batch: NonZeroUsize::new(4096).expect("nonzero byte size"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease duration"),
    }
}

fn draft(ordinal: u64) -> EventDraft {
    EventDraft {
        session_ref: "restart-session".into(),
        source_event_key: format!("restart-event-{ordinal}"),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "captured-provider".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("captured event {ordinal}")),
        payload: json!({"ordinal": ordinal}),
        anchors: json!({"session": "restart-session"}),
        source_remainder: json!({"ordinal": ordinal}),
        coverage: EventCoverage::Complete,
    }
}

fn captured_store(
    path: PathBuf,
    event_count: u64,
) -> (context_graph::ContextId, Vec<context_graph::EventId>) {
    let store = Store::open(config(path)).expect("open capture store");
    let context_id = store
        .create_context(CreateContext {
            name: "interpretation restart context".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    let events = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            capture_declaration(),
            "restart-source",
            (0..event_count).map(draft).collect(),
        ))
        .expect("capture real source rows")
        .events
        .into_iter()
        .map(|outcome| outcome.event.id)
        .collect();
    (context_id, events)
}

fn rows(store: &Store, sql: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    store
        .execute_sql_for_test(sql, &params)
        .expect("inspect runner-created durable rows")
        .rows
}

#[derive(Clone, Debug, PartialEq)]
struct ClaimSnapshot {
    id: InterpretationGenerationId,
    build_ordinal: i64,
    lease_epoch: i64,
    sealed_membership: Vec<Vec<Value>>,
}

fn actual_claim(path: &Path, context_id: context_graph::ContextId) -> ClaimSnapshot {
    let store = Store::open(config(path.to_path_buf())).expect("open mapper inspection Store");
    let generation = rows(
        &store,
        "SELECT id, build_ordinal, lease_token, lease_epoch, lease_expires_at, state FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id ORDER BY started_at DESC LIMIT 1",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(CONNECTOR.into())),
        ]),
    );
    assert_eq!(
        generation.len(),
        1,
        "mapper sees one runner-created generation"
    );
    let row = &generation[0];
    let id = match row[0] {
        Value::Uuid(id) => InterpretationGenerationId::from(id),
        ref other => panic!("runner generation id: {other:?}"),
    };
    let build_ordinal = match row[1] {
        Value::Int64(value) => value,
        ref other => panic!("runner build ordinal: {other:?}"),
    };
    assert!(
        matches!(row[2], Value::Text(_)),
        "mapper sees the real lease token"
    );
    let lease_epoch = match row[3] {
        Value::Int64(value) => value,
        ref other => panic!("runner lease epoch: {other:?}"),
    };
    assert!(
        matches!(row[4], Value::Timestamp(_)),
        "mapper sees the real lease expiry"
    );
    assert!(matches!(&row[5], Value::Text(state) if state == "building"));
    let sealed_membership = rows(
        &store,
        "SELECT ordinal, event_id, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(id.as_uuid())),
        ]),
    );
    assert!(
        !sealed_membership.is_empty(),
        "mapper sees real sealed source rows"
    );
    ClaimSnapshot {
        id,
        build_ordinal,
        lease_epoch,
        sealed_membership,
    }
}

fn durable_progress(
    store: &Store,
    context_id: context_graph::ContextId,
    id: InterpretationGenerationId,
) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(id.as_uuid())),
    ]);
    let cursor = rows(
        store,
        "SELECT build_ordinal, interpreted_event_count, state FROM event_interpretation_generations WHERE context_id = $context_id AND id = $generation_id",
        params.clone(),
    );
    let values = rows(
        store,
        "SELECT event_id, effective_sha256 FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY event_id",
        params,
    );
    (cursor, values)
}

#[derive(Clone, Debug, PartialEq)]
struct DurableRowsSnapshot {
    generation: Vec<Vec<Value>>,
    values: Vec<Vec<Value>>,
    anchors: Vec<Vec<Value>>,
    membership: Vec<Vec<Value>>,
}

fn durable_rows_snapshot(
    store: &Store,
    context_id: context_graph::ContextId,
    generation_id: InterpretationGenerationId,
) -> DurableRowsSnapshot {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
    ]);
    DurableRowsSnapshot {
        generation: rows(
            store,
            "SELECT * FROM event_interpretation_generations WHERE context_id = $context_id AND id = $generation_id",
            params.clone(),
        ),
        values: rows(
            store,
            "SELECT * FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY event_id",
            params.clone(),
        ),
        anchors: rows(
            store,
            "SELECT * FROM event_interpretation_anchors WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY event_id, declaration_version, key, identity_value",
            params.clone(),
        ),
        membership: rows(
            store,
            "SELECT * FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal",
            params,
        ),
    }
}

fn interpreted_value_rows(
    store: &Store,
    context_id: context_graph::ContextId,
    generation_id: InterpretationGenerationId,
    event_id: context_graph::EventId,
) -> Vec<Vec<Value>> {
    rows(
        store,
        "SELECT * FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
            ("event_id".into(), Value::Uuid(event_id.as_uuid())),
        ]),
    )
}

#[test]
fn repeated_ready_build_reuses_the_generation_without_remapping_or_new_values() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("repeated-ready.db");
    let (context_id, events) = captured_store(path.clone(), 2);

    let first_mapper = Arc::new(FixtureMapper::succeeds());
    let first = ConnectorInterpretationRunner::build(request(
        path.clone(),
        context_id,
        first_mapper.clone(),
    ))
    .expect("first public build succeeds");
    assert_eq!(first.state, InterpretationGenerationState::Ready);
    assert_eq!(first.interpreted_event_count, events.len() as u64);
    assert_eq!(first_mapper.calls.load(Ordering::SeqCst), events.len());
    let inspected = Store::open(config(path.clone())).expect("inspect completed generation");
    let before = durable_rows_snapshot(&inspected, context_id, first.generation_id);
    drop(inspected);

    let retry_mapper = Arc::new(FixtureMapper::succeeds());
    let retry = ConnectorInterpretationRunner::build(request(
        path.clone(),
        context_id,
        retry_mapper.clone(),
    ))
    .expect("repeated public build succeeds");
    assert_eq!(retry.state, InterpretationGenerationState::Ready);
    assert_eq!(
        retry.generation_id, first.generation_id,
        "same watermark must return the ready generation"
    );
    assert_eq!(retry.source_watermark, first.source_watermark);
    assert_eq!(
        retry_mapper.calls.load(Ordering::SeqCst),
        0,
        "ready generation does not invoke the mapper again"
    );
    let inspected = Store::open(config(path)).expect("inspect repeated build");
    let after = durable_rows_snapshot(&inspected, context_id, retry.generation_id);
    assert_eq!(after.generation.len(), before.generation.len());
    assert_eq!(after.values.len(), before.values.len());
    assert_eq!(after.anchors.len(), before.anchors.len());
    assert_eq!(after.membership.len(), before.membership.len());
    assert_eq!(
        after, before,
        "ready repeat keeps every persisted generation, value, anchor, and membership row unchanged"
    );
}

#[test]
fn mapper_interruption_restarts_the_same_sealed_generation_at_its_claimed_cursor() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("interrupted-restart.db");
    let (context_id, events) = captured_store(path.clone(), 3);
    let interrupted_claim = Arc::new(Mutex::new(None));
    let claim_for_failure = Arc::clone(&interrupted_claim);
    let failure_path = path.clone();
    let failed_mapper = Arc::new(FixtureMapper::fails_on_call(
        2,
        move || {
            *claim_for_failure.lock().expect("record interrupted claim") =
                Some(actual_claim(&failure_path, context_id));
        },
        "simulated mapper interruption",
    ));
    assert!(
        ConnectorInterpretationRunner::build(request(
            path.clone(),
            context_id,
            failed_mapper.clone(),
        ))
        .is_err(),
        "mapper interruption reaches the public runner"
    );
    let interrupted = interrupted_claim
        .lock()
        .expect("read interrupted claim")
        .clone()
        .expect("mapper observed a public-runner claim");
    assert_eq!(
        interrupted.build_ordinal, 1,
        "the first one-event batch committed before the second mapper call interrupted"
    );
    assert_eq!(
        failed_mapper.called_events(),
        events[..2].to_vec(),
        "the interruption is injected by the second actual mapper call"
    );
    let inspected = Store::open(config(path.clone())).expect("inspect interrupted generation");
    let committed_progress = durable_progress(&inspected, context_id, interrupted.id);
    assert_eq!(committed_progress.0.len(), 1, "one durable cursor row");
    assert_eq!(committed_progress.0[0][0], Value::Int64(1));
    assert_eq!(committed_progress.0[0][1], Value::Int64(1));
    assert_eq!(committed_progress.0[0][2], Value::Text("failed".into()));
    assert_eq!(
        committed_progress.1.len(),
        1,
        "only the first batch has values"
    );
    assert_eq!(
        committed_progress.1[0][0],
        Value::Uuid(events[0].as_uuid()),
        "the committed value belongs to the first real appended event"
    );
    let committed_value = interpreted_value_rows(&inspected, context_id, interrupted.id, events[0]);
    assert_eq!(
        committed_value.len(),
        1,
        "one complete earlier value is durable"
    );
    drop(inspected);

    let resumed_claim = Arc::new(Mutex::new(None));
    let resumed_progress = Arc::new(Mutex::new(None));
    let claim_for_resume = Arc::clone(&resumed_claim);
    let progress_for_resume = Arc::clone(&resumed_progress);
    let resume_path = path.clone();
    let resumed_mapper = Arc::new(FixtureMapper::succeeds_after(move || {
        *claim_for_resume.lock().expect("record resumed claim") =
            Some(actual_claim(&resume_path, context_id));
        let store = Store::open(config(resume_path.clone())).expect("inspect resumed generation");
        *progress_for_resume.lock().expect("record resumed progress") =
            Some(durable_progress(&store, context_id, interrupted.id));
    }));
    let resumed = ConnectorInterpretationRunner::build(request(
        path.clone(),
        context_id,
        resumed_mapper.clone(),
    ))
    .expect("restart completes the interrupted build");
    let observed_resume = resumed_claim
        .lock()
        .expect("read resumed claim")
        .clone()
        .expect("restart mapper observed a runner claim");
    let observed_progress = resumed_progress
        .lock()
        .expect("read resumed progress")
        .clone()
        .expect("restart mapper inspected durable progress");
    assert_eq!(
        resumed.generation_id, interrupted.id,
        "restart resumes the interrupted generation"
    );
    assert_eq!(
        observed_resume.id, interrupted.id,
        "mapper is claimed from the interrupted generation"
    );
    assert_eq!(
        observed_resume.build_ordinal, interrupted.build_ordinal,
        "restart keeps the claimed cursor"
    );
    assert_ne!(
        observed_resume.build_ordinal, 0,
        "restart claims from the committed nonzero cursor"
    );
    assert_eq!(
        observed_resume.sealed_membership, interrupted.sealed_membership,
        "restart keeps the exact sealed source membership"
    );
    assert_eq!(
        observed_progress.0[0][0], committed_progress.0[0][0],
        "restart leaves the committed cursor unchanged before mapping"
    );
    assert_eq!(
        observed_progress.0[0][1], committed_progress.0[0][1],
        "restart leaves the committed interpreted count unchanged before mapping"
    );
    assert_eq!(
        observed_progress.1, committed_progress.1,
        "restart leaves earlier persisted values unchanged before mapping"
    );
    assert_eq!(
        resumed_mapper.called_events(),
        events[1..].to_vec(),
        "restart maps only the uncommitted real events"
    );
    assert_eq!(resumed.state, InterpretationGenerationState::Ready);
    assert_eq!(resumed.interpreted_event_count, events.len() as u64);
    let inspected = Store::open(config(path)).expect("reopen completed restarted build");
    assert_eq!(
        interpreted_value_rows(&inspected, context_id, resumed.generation_id, events[0]),
        committed_value,
        "restart completion preserves the complete earlier committed value"
    );
}

#[test]
fn mapper_failure_does_not_persist_sensitive_display_text_in_generation_or_status() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("sensitive-failure.db");
    let (context_id, _) = captured_store(path.clone(), 1);
    let mapper = Arc::new(FixtureMapper::fails_on_call(1, || {}, SENSITIVE_SENTINEL));
    assert_eq!(
        mapper.configured_failure(),
        Some(SENSITIVE_SENTINEL),
        "the fixture itself injects the unique sentinel"
    );
    ConnectorInterpretationRunner::build(request(path.clone(), context_id, mapper.clone()))
        .expect_err("mapper failure reaches the caller without restricting its error text");
    assert_eq!(mapper.calls.load(Ordering::SeqCst), 1);

    let store = Store::open(config(path)).expect("inspect failed generation");
    let persisted = rows(
        &store,
        "SELECT sanitized_failure FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(CONNECTOR.into())),
        ]),
    );
    assert_eq!(persisted.len(), 1, "one failed generation is persisted");
    assert_eq!(
        persisted[0][0],
        Value::Text("interpretation_mapper_failed".into()),
        "failed generation persists the adopted mapper failure class"
    );
    let status = store
        .interpretation_status(ContextSelector::one(context_id), CONNECTOR.into())
        .expect("public status reads failed generation")
        .contexts
        .into_iter()
        .next()
        .expect("one selected Context status");
    assert_eq!(
        status.sanitized_failure.as_deref(),
        Some("interpretation_mapper_failed"),
        "public status exposes the adopted mapper failure class"
    );
}
