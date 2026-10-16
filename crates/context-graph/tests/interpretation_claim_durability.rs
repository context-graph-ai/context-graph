//! Durable interpretation claims persist across store reopen.
//!
//! Every source row is captured through `append_event_batch`.  The controls only inspect rows
//! which the public runner created; they never manufacture a generation, membership, ID, or hash.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::{
    AppendEventBatch, CapturedEventAndAnchors, CgError, ConnectorInterpretationRunner,
    ConversationEvent, CreateContext, EffectiveEventFields, EventAdapterDeclaration, EventCoverage,
    EventDraft, EventRenderRole, InstalledInterpretationContract, InterpretationBuildRequest,
    ReleasedConnectorInterpretation, ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use contextdb_core::{Value, Wallclock};
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
use serde_json::json;

const CONNECTOR: &str = "interpretation-claim-durability-red";
const TARGET: &str = "claim-durability-v1";
const T0: u64 = 1_700_000_000_000;
const PRELUDE_EVENTS: u64 = 96;
const UNCLAIMED_TAIL_EVENTS: u64 = 10_000;
const APPEND_BATCH_EVENTS: u64 = 128;

fn config(path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "claim-durability-capture-v1".into(),
        anchor_declaration_version: "claim-durability-anchors-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn release() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        target_adapter_version: "claim-durability-capture-v1".into(),
        target_anchor_declaration_version: "claim-durability-anchors-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "claim-durability-capture-v1".into(),
            anchor_declaration_version: "claim-durability-anchors-v1".into(),
        }],
        mapping_contract_id: "claim-durability-contract".into(),
        mapping_contract_sha256: "claim-durability-contract-sha256".into(),
    }
}

fn effective() -> EffectiveEventFields {
    EffectiveEventFields {
        occurred_at: None,
        author: "released-provider".into(),
        kind: "released-message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("durable released interpretation".into()),
        payload: json!({"released": true}),
        anchors: json!([{
            "key": "session",
            "canonical_value": "claim-durability-session",
            "original_value": "claim-durability-session"
        }]),
        coverage: EventCoverage::Complete,
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

#[derive(Default)]
struct QueryRecorder {
    enabled: AtomicBool,
    queries: Mutex<Vec<String>>,
}

impl QueryRecorder {
    fn enable(&self) {
        self.queries.lock().expect("query recorder lock").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    fn queries(&self) -> Vec<String> {
        self.queries.lock().expect("query recorder lock").clone()
    }
}

impl DatabasePlugin for QueryRecorder {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.queries
                .lock()
                .expect("query recorder lock")
                .push(sql.into());
        }
        Ok(())
    }

    fn post_query(&self, sql: &str, _elapsed: Duration, _outcome: &QueryOutcome) {
        // The first cursor update is the end of the resumed transactional commit.  Leaving the
        // recorder enabled would include the next claim's ordinary bounded cursor.
        if sql.contains("UPDATE event_interpretation_generations SET build_ordinal") {
            self.disable();
        }
    }
}

/// Advances only after a mapper has returned a real result.  The retained Store that owns this
/// plugin is intentional: plain runner reopens share its database plugin; it is not evidence
/// that mapping retains or releases a Store handle.
struct ExpiryOnReopenedValidationQuery {
    clock: Arc<AtomicU64>,
    expiry: AtomicU64,
    armed: AtomicBool,
    advanced: AtomicBool,
    fired_sql: Mutex<Option<String>>,
}

impl ExpiryOnReopenedValidationQuery {
    fn new(clock: Arc<AtomicU64>) -> Self {
        Self {
            clock,
            expiry: AtomicU64::new(0),
            armed: AtomicBool::new(false),
            advanced: AtomicBool::new(false),
            fired_sql: Mutex::new(None),
        }
    }

    fn arm_after_mapping(&self, expiry: u64) {
        self.expiry.store(expiry, Ordering::SeqCst);
        assert!(
            !self.armed.swap(true, Ordering::SeqCst),
            "the expiry callback is armed once, after the first mapped result"
        );
    }

    fn fired_sql(&self) -> String {
        self.fired_sql
            .lock()
            .expect("expiry callback SQL lock")
            .clone()
            .expect("the reopened commit validation queried through the plugin")
    }
}

impl DatabasePlugin for ExpiryOnReopenedValidationQuery {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.clock.store(
                self.expiry
                    .load(Ordering::SeqCst)
                    .checked_add(1)
                    .expect("lease expiry has a successor"),
                Ordering::SeqCst,
            );
            *self.fired_sql.lock().expect("expiry callback SQL lock") = Some(sql.into());
            self.advanced.store(true, Ordering::SeqCst);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ClaimSnapshot {
    generation_id: uuid::Uuid,
    start: i64,
    end: i64,
    digest: String,
    membership: Vec<(uuid::Uuid, String)>,
    lease_epoch: i64,
    lease_token: String,
    lease_expires_at: u64,
}

fn assert_same_pending_claim(actual: &ClaimSnapshot, expected: &ClaimSnapshot, message: &str) {
    assert_eq!(
        actual.generation_id, expected.generation_id,
        "{message}: generation ID"
    );
    assert_eq!(actual.start, expected.start, "{message}: start ordinal");
    assert_eq!(actual.end, expected.end, "{message}: end ordinal");
    assert_eq!(
        actual.digest, expected.digest,
        "{message}: membership digest"
    );
    assert_eq!(
        actual.membership, expected.membership,
        "{message}: ordered membership"
    );
}

fn rows(store: &Store, sql: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    store
        .execute_sql_for_test(sql, &params)
        .expect("inspect public-runner durable rows")
        .rows
}

fn claim_snapshot(store: &Store, context_id: context_graph::ContextId) -> ClaimSnapshot {
    let generation = rows(
        store,
        "SELECT id, build_ordinal, claim_end_ordinal, claim_membership_sha256, lease_epoch, lease_token, lease_expires_at FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id AND claim_end_ordinal IS NOT NULL",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(CONNECTOR.into())),
        ]),
    );
    assert_eq!(
        generation.len(),
        1,
        "the runner created one durable live claim"
    );
    let generation_id = match generation[0][0] {
        Value::Uuid(value) => value,
        ref other => panic!("generation ID: {other:?}"),
    };
    let start = match generation[0][1] {
        Value::Int64(value) => value,
        ref other => panic!("claim start ordinal: {other:?}"),
    };
    let end = match generation[0][2] {
        Value::Int64(value) => value,
        ref other => panic!("claim end ordinal: {other:?}"),
    };
    let digest = match &generation[0][3] {
        Value::Text(value) => value.clone(),
        other => panic!("claim membership digest: {other:?}"),
    };
    let lease_epoch = match generation[0][4] {
        Value::Int64(value) => value,
        ref other => panic!("claim lease epoch: {other:?}"),
    };
    let lease_token = match &generation[0][5] {
        Value::Text(value) => value.clone(),
        other => panic!("claim lease token: {other:?}"),
    };
    let lease_expires_at = match generation[0][6] {
        Value::Timestamp(value) => u64::try_from(value).expect("positive lease expiry"),
        ref other => panic!("claim lease expiry: {other:?}"),
    };
    assert!(
        start < end,
        "durable claim is a nonempty half-open interval"
    );
    let membership = rows(
        store,
        "SELECT event_id, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal >= $start AND ordinal < $end ORDER BY ordinal",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id)),
            ("start".into(), Value::Int64(start)),
            ("end".into(), Value::Int64(end)),
        ]),
    )
    .into_iter()
    .map(|row| match (&row[0], &row[1]) {
        (Value::Uuid(id), Value::Text(hash)) => (*id, hash.clone()),
        other => panic!("real membership ID/hash: {other:?}"),
    })
    .collect::<Vec<_>>();
    assert_eq!(
        i64::try_from(membership.len()).expect("membership fits i64"),
        end - start
    );
    ClaimSnapshot {
        generation_id,
        start,
        end,
        digest,
        membership,
        lease_epoch,
        lease_token,
        lease_expires_at,
    }
}

fn generation_build_ordinal(
    store: &Store,
    context_id: context_graph::ContextId,
    generation_id: uuid::Uuid,
) -> i64 {
    let generation = rows(
        store,
        "SELECT build_ordinal FROM event_interpretation_generations WHERE context_id = $context_id AND id = $generation_id",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id)),
        ]),
    );
    match generation.as_slice() {
        [row] => match row.as_slice() {
            [Value::Int64(ordinal)] => *ordinal,
            other => panic!("one generation build cursor: {other:?}"),
        },
        other => panic!("one generation build-cursor row: {other:?}"),
    }
}

fn claim_value_ids(
    store: &Store,
    context_id: context_graph::ContextId,
    generation_id: uuid::Uuid,
) -> Vec<uuid::Uuid> {
    let mut ids = rows(
        store,
        "SELECT event_id FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id)),
        ]),
    )
    .into_iter()
    .map(|row| match row.as_slice() {
        [Value::Uuid(id)] => *id,
        other => panic!("interpreted value event ID: {other:?}"),
    })
    .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

fn assert_claim_uncommitted(
    store: &Store,
    context_id: context_graph::ContextId,
    pending: &ClaimSnapshot,
) {
    assert_eq!(
        generation_build_ordinal(store, context_id, pending.generation_id),
        pending.start,
        "the expired first mapped result did not advance the durable cursor"
    );
    assert!(
        claim_value_ids(store, context_id, pending.generation_id).is_empty(),
        "the expired first mapped result did not persist values"
    );
}

fn assert_claim_values_once(
    store: &Store,
    context_id: context_graph::ContextId,
    pending: &ClaimSnapshot,
) {
    let mut expected = pending
        .membership
        .iter()
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(
        claim_value_ids(store, context_id, pending.generation_id),
        expected,
        "the resumed pending claim persisted exactly one value for every claimed source ID"
    );
}

fn draft(ordinal: u64, text: String, fixture: &str) -> EventDraft {
    EventDraft {
        session_ref: "claim-durability-session".into(),
        source_event_key: format!("{fixture}-{ordinal}"),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "provider".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(text),
        payload: json!({"fixture": fixture, "ordinal": ordinal}),
        anchors: json!({"session": "claim-durability-session"}),
        source_remainder: json!({"ordinal": ordinal}),
        coverage: EventCoverage::Complete,
    }
}

fn append_bounded_drafts(
    store: &Store,
    context_id: context_graph::ContextId,
    drafts: impl Iterator<Item = EventDraft>,
) {
    let mut batch = Vec::with_capacity(usize::try_from(APPEND_BATCH_EVENTS).expect("batch fits"));
    for draft in drafts {
        batch.push(draft);
        if batch.len() == usize::try_from(APPEND_BATCH_EVENTS).expect("batch fits") {
            store
                .append_event_batch(AppendEventBatch::live(
                    context_id,
                    declaration(),
                    "claim-durability-source",
                    std::mem::take(&mut batch),
                ))
                .expect("public bounded append creates immutable source IDs and hashes");
        }
    }
    if !batch.is_empty() {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "claim-durability-source",
                batch,
            ))
            .expect("public bounded append creates immutable source IDs and hashes");
    }
}

fn capture(store: &Store, unclaimed_tail_events: u64) -> context_graph::ContextId {
    let context_id = store
        .create_context(CreateContext {
            name: "claim durability context".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    append_bounded_drafts(
        store,
        context_id,
        std::iter::once(draft(0, "x".repeat(12_288), "claimed-large-event")).chain(
            (1..=PRELUDE_EVENTS)
                .map(|ordinal| draft(ordinal, format!("bounded prelude {ordinal}"), "prelude")),
        ),
    );
    append_bounded_drafts(
        store,
        context_id,
        ((PRELUDE_EVENTS + 1)..=(PRELUDE_EVENTS + unclaimed_tail_events)).map(|ordinal| {
            draft(
                ordinal,
                format!("unclaimed tail {ordinal}"),
                "unclaimed-tail",
            )
        }),
    );
    context_id
}

fn request(
    path: PathBuf,
    context_id: context_graph::ContextId,
    mapper: Arc<dyn ReleasedConnectorInterpretation>,
) -> InterpretationBuildRequest {
    InterpretationBuildRequest {
        store: config(path),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        mapper,
        contract: Arc::new(FixtureContract),
        max_events_per_batch: NonZeroU32::new(97).expect("nonzero event bound"),
        max_bytes_per_batch: NonZeroUsize::new(16_384).expect("nonzero byte bound"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease duration"),
    }
}

struct InterruptPendingClaim {
    store: Store,
    context_id: context_graph::ContextId,
    pending: Mutex<Option<ClaimSnapshot>>,
}

impl ReleasedConnectorInterpretation for InterruptPendingClaim {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        static DECLARATION: std::sync::OnceLock<ReleasedInterpretationDeclaration> =
            std::sync::OnceLock::new();
        DECLARATION.get_or_init(release)
    }

    fn interpret_event(&self, event: &ConversationEvent) -> Result<EffectiveEventFields> {
        let snapshot = claim_snapshot(&self.store, self.context_id);
        let mut pending = self.pending.lock().expect("pending claim lock");
        assert!(
            pending.is_none(),
            "the interruption stops inside the first actual pending claim"
        );
        assert!(
            snapshot
                .membership
                .iter()
                .any(|(id, _)| *id == event.id.as_uuid()),
            "mapper receives an actual claimed source ID"
        );
        *pending = Some(snapshot);
        Err(CgError::Engine(
            "interrupt inside the real pending durable claim".into(),
        ))
    }
}

struct ResumePendingClaim {
    store: Store,
    context_id: context_graph::ContextId,
    pending: ClaimSnapshot,
    calls: Mutex<Vec<uuid::Uuid>>,
    recorder: Option<Arc<QueryRecorder>>,
}

impl ReleasedConnectorInterpretation for ResumePendingClaim {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        static DECLARATION: std::sync::OnceLock<ReleasedInterpretationDeclaration> =
            std::sync::OnceLock::new();
        DECLARATION.get_or_init(release)
    }

    fn interpret_event(&self, event: &ConversationEvent) -> Result<EffectiveEventFields> {
        let snapshot = claim_snapshot(&self.store, self.context_id);
        if snapshot.start >= self.pending.end {
            return Err(CgError::Engine(
                "stop after committing the resumed pending claim".into(),
            ));
        }
        assert_same_pending_claim(
            &snapshot,
            &self.pending,
            "restart maps the same pending bounds, digest, ordered IDs, and source hashes",
        );
        let mut calls = self.calls.lock().expect("mapper calls lock");
        calls.push(event.id.as_uuid());
        if snapshot
            .membership
            .last()
            .is_some_and(|(id, _)| *id == event.id.as_uuid())
            && let Some(recorder) = &self.recorder
        {
            // All mapper-owned snapshot queries precede this boundary.  The next query is
            // the runner's commit work, including one primary-key membership read per row.
            recorder.enable();
        }
        Ok(effective())
    }
}

struct ReclaimAfterReopenedExpiry {
    store: Store,
    context_id: context_graph::ContextId,
    pending: ClaimSnapshot,
    expiry: Arc<ExpiryOnReopenedValidationQuery>,
    calls: Mutex<Vec<uuid::Uuid>>,
    attempts: Mutex<Vec<(i64, String)>>,
    checked_reclaim: AtomicBool,
}

impl ReleasedConnectorInterpretation for ReclaimAfterReopenedExpiry {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        static DECLARATION: std::sync::OnceLock<ReleasedInterpretationDeclaration> =
            std::sync::OnceLock::new();
        DECLARATION.get_or_init(release)
    }

    fn interpret_event(&self, event: &ConversationEvent) -> Result<EffectiveEventFields> {
        let snapshot = claim_snapshot(&self.store, self.context_id);
        if snapshot.start >= self.pending.end {
            return Err(CgError::Engine(
                "stop after committing the reclaimed pending claim".into(),
            ));
        }
        assert_same_pending_claim(
            &snapshot,
            &self.pending,
            "the expired result reclaims the same persisted pending claim",
        );
        let mut attempts = self.attempts.lock().expect("mapping attempts lock");
        let lease = (snapshot.lease_epoch, snapshot.lease_token.clone());
        if attempts.last() != Some(&lease) {
            if let Some((previous_epoch, previous_token)) = attempts.last() {
                assert!(
                    snapshot.lease_epoch > *previous_epoch
                        && snapshot.lease_token != *previous_token,
                    "the subsequent mapping attempt has a renewed lease epoch and token"
                );
                assert!(
                    self.expiry.advanced.load(Ordering::SeqCst)
                        && !self.checked_reclaim.swap(true, Ordering::SeqCst),
                    "only the subsequent mapping attempt checks the expired first attempt"
                );
                assert_claim_uncommitted(&self.store, self.context_id, &self.pending);
            }
            attempts.push(lease);
        }
        let first_attempt = attempts.len() == 1;
        drop(attempts);
        self.calls
            .lock()
            .expect("mapper calls lock")
            .push(event.id.as_uuid());
        if first_attempt
            && snapshot
                .membership
                .last()
                .is_some_and(|(id, _)| *id == event.id.as_uuid())
        {
            // Every first-attempt member has inspected its persisted claim before this callback
            // can advance the clock; a later member cannot masquerade as a reclaimed attempt.
            self.expiry.arm_after_mapping(snapshot.lease_expires_at);
        }
        Ok(effective())
    }
}

fn pending_claim_fixture(
    unclaimed_tail_events: u64,
) -> (
    tempfile::TempDir,
    PathBuf,
    Store,
    context_graph::ContextId,
    Arc<InterruptPendingClaim>,
) {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("claims.db");
    let store = Store::open(config(path.clone())).expect("open capture Store");
    let context_id = capture(&store, unclaimed_tail_events);
    let mapper = Arc::new(InterruptPendingClaim {
        store: Store::open(config(path.clone())).expect("open mapper inspection Store"),
        context_id,
        pending: Mutex::new(None),
    });
    (root, path, store, context_id, mapper)
}

#[test]
fn interrupted_pending_byte_claim_reopens_with_identical_membership_and_values_once() {
    let (_root, path, retained_store, context_id, interrupted) = pending_claim_fixture(0);
    let first = ConnectorInterpretationRunner::build(request(
        path.clone(),
        context_id,
        interrupted.clone(),
    ))
    .expect_err("the mapper interrupts the actual pending claim before commit");
    assert!(
        matches!(first, CgError::Engine(message) if message == "interrupt inside the real pending durable claim")
    );
    let pending = interrupted
        .pending
        .lock()
        .expect("pending claim lock")
        .clone()
        .expect("mapper inspected the real pending claim");
    assert!(
        pending.end < i64::try_from(PRELUDE_EVENTS + 1).expect("prelude bound fits"),
        "the durable claim is cut by bytes before the row bound"
    );
    assert_claim_uncommitted(&retained_store, context_id, &pending);

    let resumed = Arc::new(ResumePendingClaim {
        store: Store::open(config(path.clone())).expect("open restart mapper inspection Store"),
        context_id,
        pending: pending.clone(),
        calls: Mutex::new(Vec::new()),
        recorder: None,
    });
    let resumed_error =
        ConnectorInterpretationRunner::build(request(path, context_id, resumed.clone()))
            .expect_err("stop once the resumed pending claim has committed");
    assert!(
        matches!(resumed_error, CgError::Engine(message) if message == "stop after committing the resumed pending claim")
    );
    assert_eq!(
        resumed.calls.lock().expect("mapper calls lock").as_slice(),
        pending
            .membership
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        "restart maps each member of the same pending claim once"
    );
    assert_eq!(
        generation_build_ordinal(&retained_store, context_id, pending.generation_id),
        pending.end,
        "commit advances exactly the pending claim cursor"
    );
    assert_claim_values_once(&retained_store, context_id, &pending);
}

#[test]
fn commit_uses_only_full_primary_key_membership_reads_for_the_resumed_pending_claim() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("bounded-claim.db");
    let recorder = Arc::new(QueryRecorder::default());
    // This retained fixture exists solely so runner reopens share the recorder.  It intentionally
    // does not prove Store release.
    let store = Store::open_with_database_plugin(config(path.clone()), recorder.clone())
        .expect("open plugin fixture Store");
    let context_id = capture(&store, UNCLAIMED_TAIL_EVENTS);
    let interrupted = Arc::new(InterruptPendingClaim {
        store: Store::open(config(path.clone())).expect("open mapper inspection Store"),
        context_id,
        pending: Mutex::new(None),
    });
    ConnectorInterpretationRunner::build(request(path.clone(), context_id, interrupted.clone()))
        .expect_err("interrupt inside the actual pending claim");
    let pending = interrupted
        .pending
        .lock()
        .expect("pending claim lock")
        .clone()
        .expect("pending claim");
    assert!(
        pending.end < i64::try_from(PRELUDE_EVENTS + 1).expect("prelude bound fits"),
        "the resumed claim ends before the separately appended unclaimed tail"
    );
    let tail = rows(
        &store,
        "SELECT ordinal FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal >= $tail_start",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(pending.generation_id)),
            (
                "tail_start".into(),
                Value::Int64(i64::try_from(PRELUDE_EVENTS + 1).expect("tail start fits")),
            ),
        ]),
    );
    assert_eq!(
        u64::try_from(tail.len()).expect("tail length fits"),
        UNCLAIMED_TAIL_EVENTS,
        "10,000 public-appended memberships remain beyond the actual resumed claim"
    );

    let resumed = Arc::new(ResumePendingClaim {
        store: Store::open(config(path.clone())).expect("open restart mapper inspection Store"),
        context_id,
        pending: pending.clone(),
        calls: Mutex::new(Vec::new()),
        recorder: Some(recorder.clone()),
    });
    ConnectorInterpretationRunner::build(request(path, context_id, resumed))
        .expect_err("stop once the actual resumed pending claim has committed");
    recorder.disable();
    let membership_reads = recorder
        .queries()
        .into_iter()
        .filter(|query| query.contains("event_interpretation_source_events"))
        .collect::<Vec<_>>();
    assert_eq!(
        membership_reads.len(),
        pending.membership.len(),
        "commit reads one persisted membership for every resumed claimed ordinal"
    );
    for query in membership_reads {
        assert!(
            query.contains("context_id = $context_id")
                && query.contains("generation_id = $generation_id")
                && query.contains("ordinal = $ordinal"),
            "commit membership read has the full context, generation, and ordinal key: {query}"
        );
        assert!(
            !query.contains("ordinal >=")
                && !query.contains("ordinal <=")
                && !query.contains("ORDER BY"),
            "commit rejects every membership range, ordered read, and unordered scan: {query}"
        );
    }
}

#[test]
fn reopened_validation_expiry_discards_the_first_result_then_reclaims_the_same_claim_once() {
    let clock = Arc::new(AtomicU64::new(T0));
    let clock_for_guard = Arc::clone(&clock);
    let _guard = Wallclock::test_clock_guard(move || clock_for_guard.load(Ordering::SeqCst));
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("reopened-expiry.db");
    let expiry = Arc::new(ExpiryOnReopenedValidationQuery::new(Arc::clone(&clock)));
    // This retained Store owns the test-only plugin for runner reopens; it is not a release proof.
    let retained_store = Store::open_with_database_plugin(config(path.clone()), expiry.clone())
        .expect("open plugin fixture Store");
    let context_id = capture(&retained_store, 0);
    let interrupted = Arc::new(InterruptPendingClaim {
        store: Store::open(config(path.clone())).expect("open mapper inspection Store"),
        context_id,
        pending: Mutex::new(None),
    });
    ConnectorInterpretationRunner::build(request(path.clone(), context_id, interrupted.clone()))
        .expect_err("leave the actual durable claim pending before restart");
    let pending = interrupted
        .pending
        .lock()
        .expect("pending claim lock")
        .clone()
        .expect("pending claim");
    assert_eq!(
        clock.load(Ordering::SeqCst),
        T0,
        "the restart begins before the plugin advances the guarded wallclock"
    );

    let resumed = Arc::new(ReclaimAfterReopenedExpiry {
        store: Store::open(config(path.clone())).expect("open restart mapper inspection Store"),
        context_id,
        pending: pending.clone(),
        expiry: expiry.clone(),
        calls: Mutex::new(Vec::new()),
        attempts: Mutex::new(Vec::new()),
        checked_reclaim: AtomicBool::new(false),
    });
    let resumed_error =
        ConnectorInterpretationRunner::build(request(path, context_id, resumed.clone()))
            .expect_err("stop once the reclaimed pending claim has committed");
    assert!(
        matches!(resumed_error, CgError::Engine(message) if message == "stop after committing the reclaimed pending claim")
    );
    assert!(
        expiry.advanced.load(Ordering::SeqCst) && resumed.checked_reclaim.load(Ordering::SeqCst),
        "the first mapped result expired at reopened validation and the next attempt reclaimed it"
    );
    assert!(
        expiry
            .fired_sql()
            .contains("FROM event_interpretation_generations"),
        "the plugin advanced the clock on the reopened commit validation query"
    );
    let expected_once = pending
        .membership
        .iter()
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    let calls = resumed.calls.lock().expect("mapper calls lock").clone();
    assert_eq!(calls, [expected_once.clone(), expected_once].concat());
    assert_eq!(
        resumed
            .attempts
            .lock()
            .expect("mapping attempts lock")
            .len(),
        2,
        "two renewed lease attempts each map the same pending membership once"
    );
    assert_eq!(
        generation_build_ordinal(&retained_store, context_id, pending.generation_id),
        pending.end,
        "only the reclaimed result advances the cursor"
    );
    assert_claim_values_once(&retained_store, context_id, &pending);
}
