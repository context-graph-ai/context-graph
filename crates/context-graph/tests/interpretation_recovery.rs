//! RED contracts for recovering event interpretation claims.
//!
//! Each control changes a row that the real public runner has already created. The mapper is the
//! only interruption point: it runs after the runner has released every Store handle and before
//! the claimed result can commit.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use context_graph::test_support::store_database_is_live_for_test;
use context_graph::{
    AppendEventBatch, CapturedEventAndAnchors, CgError, ConnectorInterpretationRunner,
    ConversationEvent, CreateContext, EffectiveEventFields, EventAdapterDeclaration, EventCoverage,
    EventDraft, EventRenderRole, InstalledInterpretationContract, InterpretationBuildRequest,
    InterpretationGenerationState, InterpretationLeaseRefusalReason,
    ReleasedConnectorInterpretation, ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use contextdb_core::{Value, Wallclock};
use serde_json::json;

const CONNECTOR: &str = "interpretation-recovery-fixture";
const TARGET: &str = "released-recovery-v1";

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
        adapter_version: "capture-recovery-v1".into(),
        anchor_declaration_version: "anchors-recovery-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(text: &str) -> EventDraft {
    EventDraft {
        session_ref: "recovery-session".into(),
        source_event_key: "retained-natural-key".into(),
        source_position: json!({"ordinal": 1}),
        occurred_at: None,
        author: "provider".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(text.into()),
        payload: json!({"fixture": text}),
        anchors: json!({"session": "recovery-session"}),
        source_remainder: json!({"unmapped": text}),
        coverage: EventCoverage::Complete,
    }
}

fn release() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        target_adapter_version: "capture-recovery-v1".into(),
        target_anchor_declaration_version: "anchors-recovery-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "capture-recovery-v1".into(),
            anchor_declaration_version: "anchors-recovery-v1".into(),
        }],
        mapping_contract_id: "recovery-fixture-contract".into(),
        mapping_contract_sha256: "recovery-fixture-contract-sha256".into(),
    }
}

fn expected() -> EffectiveEventFields {
    EffectiveEventFields {
        occurred_at: None,
        author: "interpreted-provider".into(),
        kind: "interpreted-message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("interpreted retained event".into()),
        payload: json!({"interpreted": true}),
        anchors: json!([{
            "key": "session",
            "canonical_value": "recovery-session",
            "original_value": "recovery-session"
        }]),
        coverage: EventCoverage::Complete,
    }
}

type MapperControl = Box<dyn Fn() + Send + Sync>;

struct RecoveryMapper {
    declaration: ReleasedInterpretationDeclaration,
    answer: EffectiveEventFields,
    controls: Mutex<Vec<MapperControl>>,
    calls: AtomicUsize,
    control_calls: AtomicUsize,
}

impl RecoveryMapper {
    fn new(control: impl FnOnce() + Send + 'static) -> Self {
        let control = Mutex::new(Some(control));
        Self::with_controls(vec![Box::new(move || {
            if let Some(control) = control.lock().expect("one-shot mapper control lock").take() {
                control();
            }
        })])
    }

    fn with_controls(controls: Vec<MapperControl>) -> Self {
        Self {
            declaration: release(),
            answer: expected(),
            controls: Mutex::new(controls),
            calls: AtomicUsize::new(0),
            control_calls: AtomicUsize::new(0),
        }
    }
}

impl ReleasedConnectorInterpretation for RecoveryMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn interpret_event(&self, _: &ConversationEvent) -> Result<EffectiveEventFields> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let control = {
            let mut controls = self.controls.lock().expect("mapper control lock");
            (!controls.is_empty()).then(|| controls.remove(0))
        };
        if let Some(control) = control {
            self.control_calls.fetch_add(1, Ordering::SeqCst);
            control();
        }
        Ok(self.answer.clone())
    }
}

struct RecoveryContract {
    declaration: ReleasedInterpretationDeclaration,
}

impl InstalledInterpretationContract for RecoveryContract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn validate(
        &self,
        captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        assert_eq!(captured.event.connector_id, CONNECTOR);
        assert_eq!(
            captured.anchors.len(),
            1,
            "complete captured anchors reach the contract"
        );
        assert_eq!(
            candidate,
            &expected(),
            "independent contract sees the complete candidate"
        );
        Ok(())
    }
}

fn request(
    path: PathBuf,
    context_id: context_graph::ContextId,
    mapper: Arc<RecoveryMapper>,
) -> InterpretationBuildRequest {
    InterpretationBuildRequest {
        store: config(path),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        mapper,
        contract: Arc::new(RecoveryContract {
            declaration: release(),
        }),
        max_events_per_batch: NonZeroU32::new(1).expect("nonzero batch size"),
        max_bytes_per_batch: NonZeroUsize::new(4096).expect("nonzero byte size"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease duration"),
    }
}

fn rows(store: &Store, sql: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    store
        .execute_sql_for_test(sql, &params)
        .expect("inspect or control real durable state")
        .rows
}

fn one_row(store: &Store, sql: &str, params: HashMap<String, Value>) -> Vec<Value> {
    let rows = rows(store, sql, params);
    assert_eq!(
        rows.len(),
        1,
        "control selects exactly one build-created row"
    );
    rows.into_iter().next().expect("one real row")
}

fn open_control_store(path: &Path) -> Store {
    assert!(
        !store_database_is_live_for_test(path),
        "the runner must release its Store and sync_database Arc before mapper execution"
    );
    Store::open(config(path.to_path_buf())).expect("open independent mapper control Store")
}

fn captured_store(path: PathBuf) -> (Store, context_graph::ContextId, ConversationEvent) {
    let store = Store::open(config(path)).expect("open capture Store");
    let context_id = store
        .create_context(CreateContext {
            name: "interpretation recovery context".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    let event = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "recovery-source",
            vec![draft("original retained content")],
        ))
        .expect("public capture")
        .events
        .into_iter()
        .next()
        .expect("one event")
        .event;
    (store, context_id, event)
}

#[derive(Clone, Debug, PartialEq)]
struct LiveGenerationSnapshot {
    id: uuid::Uuid,
    lease_token: String,
    lease_epoch: i64,
    lease_expires_at: i64,
    build_ordinal: i64,
    sealed_membership: Vec<Vec<Value>>,
}

fn now_millis() -> i64 {
    // The same seamed clock the store stamps leases with.
    i64::try_from(Wallclock::now().0).expect("current timestamp fits engine TIMESTAMP")
}

fn selected_live_generation(
    store: &Store,
    context_id: context_graph::ContextId,
) -> LiveGenerationSnapshot {
    let generation = one_row(
        store,
        "SELECT id, lease_token, lease_epoch, lease_expires_at, build_ordinal, state FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id AND lease_token IS NOT NULL AND lease_expires_at > $now ORDER BY started_at DESC LIMIT 1",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(CONNECTOR.into())),
            ("now".into(), Value::Timestamp(now_millis())),
        ]),
    );
    let id = match generation[0] {
        Value::Uuid(id) => id,
        ref other => panic!("live generation id: {other:?}"),
    };
    let lease_token = match &generation[1] {
        Value::Text(token) => token.clone(),
        other => panic!("live generation token: {other:?}"),
    };
    let lease_epoch = match generation[2] {
        Value::Int64(epoch) => epoch,
        ref other => panic!("live generation epoch: {other:?}"),
    };
    let lease_expires_at = match generation[3] {
        Value::Timestamp(expires_at) => expires_at,
        ref other => panic!("live generation expiry: {other:?}"),
    };
    assert!(
        lease_expires_at > now_millis(),
        "selected lease is unexpired"
    );
    let build_ordinal = match generation[4] {
        Value::Int64(ordinal) => ordinal,
        ref other => panic!("live generation build ordinal: {other:?}"),
    };
    assert!(
        matches!(&generation[5], Value::Text(state) if state == "building"),
        "selected live generation remains in building state"
    );
    let sealed_membership = rows(
        store,
        "SELECT event_id, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(id)),
        ]),
    );
    assert!(
        !sealed_membership.is_empty(),
        "selected live generation has sealed source membership before control mutation"
    );
    LiveGenerationSnapshot {
        id,
        lease_token,
        lease_epoch,
        lease_expires_at,
        build_ordinal,
        sealed_membership,
    }
}

fn assert_no_values_or_head(store: &Store, context_id: context_graph::ContextId) {
    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    assert!(
        rows(
            store,
            "SELECT event_id FROM event_interpretation_values WHERE context_id = $context_id",
            params.clone(),
        )
        .is_empty(),
        "a rejected claim writes no effective values"
    );
    assert!(
        rows(
            store,
            "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id",
            params,
        )
        .is_empty(),
        "a rejected claim changes no active head"
    );
}

fn assert_refused_claim_did_not_advance(
    store: &Store,
    context_id: context_graph::ContextId,
    selected: &LiveGenerationSnapshot,
) {
    assert_no_values_or_head(store, context_id);
    assert_eq!(
        one_row(
            store,
            "SELECT build_ordinal FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
            HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("id".into(), Value::Uuid(selected.id)),
            ]),
        ),
        vec![Value::Int64(selected.build_ordinal)],
        "refused claim does not advance the selected generation build cursor"
    );
}

#[test]
fn retained_shared_event_without_local_commit_is_included_by_public_build() {
    let root = tempfile::tempdir().expect("tempdir");
    let seed_path = root.path().join("seed.db");
    let fixture_path = root.path().join("fixture.db");
    let (seed, context_id, event) = captured_store(seed_path);
    let fixture = Store::open(config(fixture_path.clone())).expect("open shared-row fixture Store");
    let fixture_context_id = fixture
        .create_context(CreateContext {
            name: "interpretation recovery context".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create matching deterministic Context")
        .id;
    assert_eq!(
        fixture_context_id, context_id,
        "both Stores use the same Context ID"
    );

    for table in ["events", "event_anchors"] {
        let copied = seed
            .execute_sql_for_test(
                &format!("SELECT * FROM {table} WHERE context_id = $context_id"),
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
            )
            .expect("read production-created shared rows");
        assert!(!copied.rows.is_empty(), "seed has {table} rows");
        let placeholders = copied
            .columns
            .iter()
            .map(|column| format!("${column}"))
            .collect::<Vec<_>>()
            .join(", ");
        let insert = format!(
            "INSERT INTO {table} ({}) VALUES ({placeholders})",
            copied.columns.join(", ")
        );
        for row in &copied.rows {
            fixture
                .execute_sql_for_test(
                    &insert,
                    &copied
                        .columns
                        .iter()
                        .cloned()
                        .zip(row.iter().cloned())
                        .collect(),
                )
                .expect("copy exact production-created shared row without a local commit");
        }
        let fixture_rows = fixture
            .execute_sql_for_test(
                &format!("SELECT * FROM {table} WHERE context_id = $context_id"),
                &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
            )
            .expect("read copied shared rows");
        assert_eq!(fixture_rows.columns, copied.columns);
        assert_eq!(
            fixture_rows.rows, copied.rows,
            "{table} stays full-row equivalent"
        );
    }
    assert!(
        rows(
            &fixture,
            "SELECT event_id FROM event_local_commits",
            HashMap::new()
        )
        .is_empty(),
        "fixture deliberately has no local bookkeeping row"
    );
    drop(seed);
    drop(fixture);

    let mapper = Arc::new(RecoveryMapper::new(|| {}));
    let receipt =
        ConnectorInterpretationRunner::build(request(fixture_path.clone(), context_id, mapper))
            .expect("public build includes the retained shared event");
    assert_eq!(receipt.eligible_event_count, 1);
    assert_eq!(receipt.interpreted_event_count, 1);
    assert_eq!(receipt.state, InterpretationGenerationState::Ready);
    let inspected = Store::open(config(fixture_path)).expect("reopen completed build");
    assert_eq!(
        rows(
            &inspected,
            "SELECT event_id FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id",
            HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                (
                    "generation_id".into(),
                    Value::Uuid(receipt.generation_id.as_uuid())
                ),
            ]),
        ),
        vec![vec![Value::Uuid(event.id.as_uuid())]],
        "sealed membership contains the exact shared event ID"
    );
}

#[test]
fn stale_claim_changed_during_mapper_is_refused_without_values_or_head() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("stale.db");
    let (store, context_id, _) = captured_store(path.clone());
    drop(store);
    let control_path = path.clone();
    let selected_claim = Arc::new(Mutex::new(None));
    let callback_claim = Arc::clone(&selected_claim);
    let mapper = Arc::new(RecoveryMapper::new(move || {
        let control = open_control_store(&control_path);
        let generation = selected_live_generation(&control, context_id);
        *callback_claim
            .lock()
            .expect("capture selected real generation") = Some(generation.clone());
        rows(
            &control,
            "UPDATE event_interpretation_generations SET lease_token = $lease_token, lease_epoch = $lease_epoch WHERE context_id = $context_id AND id = $id",
            HashMap::from([
                (
                    "lease_token".into(),
                    Value::Text("replacement-real-lease-token".into()),
                ),
                (
                    "lease_epoch".into(),
                    Value::Int64(generation.lease_epoch + 1),
                ),
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("id".into(), Value::Uuid(generation.id)),
            ]),
        );
    }));
    let error =
        ConnectorInterpretationRunner::build(request(path.clone(), context_id, mapper.clone()))
            .expect_err("late mapper result is refused against the actual replaced lease");
    assert!(matches!(
        error,
        CgError::InterpretationLeaseRefused {
            reason: InterpretationLeaseRefusalReason::StaleClaim,
            ..
        }
    ));
    assert_eq!(
        mapper.calls.load(Ordering::SeqCst),
        1,
        "mapper callback ran before refusal"
    );
    assert_eq!(
        mapper.control_calls.load(Ordering::SeqCst),
        1,
        "stale-claim control callback ran before typed refusal"
    );
    let inspected = Store::open(config(path)).expect("inspect stale refusal");
    assert_refused_claim_did_not_advance(
        &inspected,
        context_id,
        selected_claim
            .lock()
            .expect("read selected real generation")
            .as_ref()
            .expect("control captured a real live generation"),
    );
}

#[test]
fn expired_claim_reclaims_once_and_commits_one_value_set() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("expired.db");
    let (store, context_id, _) = captured_store(path.clone());
    drop(store);
    let first_control_path = path.clone();
    let first_claim = Arc::new(Mutex::new(None));
    let first_callback_claim = Arc::clone(&first_claim);
    let second_control_path = path.clone();
    let second_callback_claim = Arc::clone(&first_claim);
    let reclaimed_claim = Arc::new(Mutex::new(None));
    let reclaimed_callback_claim = Arc::clone(&reclaimed_claim);
    let mapper = Arc::new(RecoveryMapper::with_controls(vec![
        Box::new(move || {
            let control = open_control_store(&first_control_path);
            let generation = selected_live_generation(&control, context_id);
            *first_callback_claim
                .lock()
                .expect("capture first real generation") = Some(generation.clone());
            rows(
                &control,
                "UPDATE event_interpretation_generations SET lease_expires_at = $expired WHERE context_id = $context_id AND id = $id",
                HashMap::from([
                    ("expired".into(), Value::Timestamp(0)),
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(generation.id)),
                ]),
            );
        }),
        Box::new(move || {
            let control = open_control_store(&second_control_path);
            let renewed = selected_live_generation(&control, context_id);
            let first = second_callback_claim
                .lock()
                .expect("read first real generation")
                .clone()
                .expect("first control captured a real live generation");
            assert_eq!(
                renewed.id, first.id,
                "reclaim renews the selected real generation"
            );
            assert_eq!(
                renewed.sealed_membership, first.sealed_membership,
                "reclaim retains the same sealed source membership"
            );
            assert_eq!(
                renewed.build_ordinal, first.build_ordinal,
                "expired first attempt did not advance the selected generation build cursor"
            );
            assert_ne!(
                renewed.lease_token, first.lease_token,
                "reclaim renews the live lease token"
            );
            assert!(
                renewed.lease_epoch > first.lease_epoch,
                "reclaim advances the live lease epoch"
            );
            assert!(
                renewed.lease_expires_at > now_millis(),
                "reclaim gives the renewed lease a future expiry"
            );
            assert_refused_claim_did_not_advance(&control, context_id, &first);
            *reclaimed_callback_claim
                .lock()
                .expect("capture reclaimed real generation") = Some(renewed);
        }),
    ]));
    let receipt =
        ConnectorInterpretationRunner::build(request(path.clone(), context_id, mapper.clone()))
            .expect("expired first claim is reclaimed and the second attempt commits");
    assert_eq!(
        mapper.calls.load(Ordering::SeqCst),
        2,
        "only the reclaimed attempt remaps"
    );
    assert_eq!(
        mapper.control_calls.load(Ordering::SeqCst),
        2,
        "both expiry controls ran across the rejected and reclaimed mapper attempts"
    );
    assert_eq!(receipt.interpreted_event_count, 1);
    let inspected = Store::open(config(path)).expect("inspect reclaimed build");
    let reclaimed = reclaimed_claim
        .lock()
        .expect("read reclaimed real generation")
        .clone()
        .expect("second control captured the reclaimed live generation");
    assert_eq!(
        receipt.generation_id.as_uuid(),
        reclaimed.id,
        "successful retry returns the generation reclaimed during its callback"
    );
    assert_eq!(
        one_row(
            &inspected,
            "SELECT build_ordinal FROM event_interpretation_generations WHERE context_id = $context_id AND id = $generation_id",
            HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(reclaimed.id)),
            ]),
        ),
        vec![Value::Int64(
            reclaimed.build_ordinal
                + i64::try_from(reclaimed.sealed_membership.len())
                    .expect("sealed membership count fits build ordinal"),
        )],
        "successful reclaimed claim advances its persisted cursor through its actual sealed membership"
    );
    assert_eq!(
        rows(
            &inspected,
            "SELECT event_id FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id",
            HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(receipt.generation_id.as_uuid())),
            ]),
        )
        .len(),
        1,
        "reclaimed lease commits one value set"
    );
    let generation = one_row(
        &inspected,
        "SELECT lease_token FROM event_interpretation_generations WHERE context_id = $context_id AND id = $generation_id",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            (
                "generation_id".into(),
                Value::Uuid(receipt.generation_id.as_uuid()),
            ),
        ]),
    );
    assert!(
        matches!(generation[0], Value::Null),
        "committed claim consumes its live lease token"
    );
}

#[test]
fn changed_source_or_membership_during_mapper_is_refused_against_sealed_real_rows() {
    for control in ["source", "membership"] {
        let root = tempfile::tempdir().expect("tempdir");
        let path = root.path().join(format!("{control}.db"));
        let (store, context_id, original) = captured_store(path.clone());
        drop(store);
        let control_path = path.clone();
        let selected_claim = Arc::new(Mutex::new(None));
        let callback_claim = Arc::clone(&selected_claim);
        let mapper = Arc::new(RecoveryMapper::new(move || {
            let control_store = open_control_store(&control_path);
            let generation = selected_live_generation(&control_store, context_id);
            *callback_claim
                .lock()
                .expect("capture selected real generation") = Some(generation.clone());
            if control == "membership" {
                rows(
                    &control_store,
                    "UPDATE event_interpretation_generations SET event_count = event_count + 1 WHERE context_id = $context_id AND id = $id",
                    HashMap::from([
                        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                        ("id".into(), Value::Uuid(generation.id)),
                    ]),
                );
            } else {
                let event_params = HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("event_id".into(), Value::Uuid(original.id.as_uuid())),
                ]);
                for (table, event_column) in [
                    ("event_local_commits", "event_id"),
                    ("event_anchors", "event_id"),
                    ("events", "id"),
                ] {
                    control_store
                        .execute_sql_for_test(
                            &format!("PURGE FROM {table} WHERE context_id = $context_id AND {event_column} = $event_id"),
                            &event_params,
                        )
                        .expect("purge immutable source row only through the engine's purge path");
                }
                let replacement = control_store
                    .append_event_batch(AppendEventBatch::live(
                        context_id,
                        declaration(),
                        "recovery-source",
                        vec![draft("replacement source content")],
                    ))
                    .expect("publicly append replacement under the same natural key")
                    .events
                    .into_iter()
                    .next()
                    .expect("replacement event")
                    .event;
                assert_eq!(
                    replacement.id, original.id,
                    "real Store reuses the natural event ID"
                );
                assert_ne!(
                    replacement.content_hash, original.content_hash,
                    "full source row changed"
                );
            }
        }));
        let error =
            ConnectorInterpretationRunner::build(request(path.clone(), context_id, mapper.clone()))
                .expect_err("sealed claim cannot commit after real source or membership change");
        let expected = if control == "source" {
            InterpretationLeaseRefusalReason::SourceChanged
        } else {
            InterpretationLeaseRefusalReason::MembershipChanged
        };
        assert!(matches!(
            error,
            CgError::InterpretationLeaseRefused { reason, .. } if reason == expected
        ));
        assert_eq!(
            mapper.calls.load(Ordering::SeqCst),
            1,
            "mapper callback ran before refusal"
        );
        assert_eq!(
            mapper.control_calls.load(Ordering::SeqCst),
            1,
            "{control} control callback ran before typed refusal"
        );
        let inspected = Store::open(config(path)).expect("inspect revalidation refusal");
        assert_refused_claim_did_not_advance(
            &inspected,
            context_id,
            selected_claim
                .lock()
                .expect("read selected real generation")
                .as_ref()
                .expect("control captured a real live generation"),
        );
    }
}
