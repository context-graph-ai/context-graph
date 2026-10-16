//! regression regressions for concurrent public interpretation runners.
//!
//! The mappers are the only barriers.  Each captures the durable claim made by a real runner;
//! the control expires only that row through the existing test SQL seam.  No test invents a
//! generation, member, event ID, or lease token.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use context_graph::{
    AppendEventBatch, CapturedEventAndAnchors, CgError, ConnectorInterpretationRunner,
    ConversationEvent, CreateContext, EffectiveEventFields, EventAdapterDeclaration, EventCoverage,
    EventDraft, EventRenderRole, InstalledInterpretationContract, InterpretationBuildRequest,
    ReleasedConnectorInterpretation, ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "interpretation-claim-concurrency-red";
const TARGET: &str = "claim-concurrency-v1";
const CHANNEL_TIMEOUT: Duration = Duration::from_secs(3);

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
        adapter_version: "claim-concurrency-capture-v1".into(),
        anchor_declaration_version: "claim-concurrency-anchors-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn release() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        target_adapter_version: "claim-concurrency-capture-v1".into(),
        target_anchor_declaration_version: "claim-concurrency-anchors-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "claim-concurrency-capture-v1".into(),
            anchor_declaration_version: "claim-concurrency-anchors-v1".into(),
        }],
        mapping_contract_id: "claim-concurrency-contract".into(),
        mapping_contract_sha256: "claim-concurrency-contract-sha256".into(),
    }
}

fn effective() -> EffectiveEventFields {
    EffectiveEventFields {
        occurred_at: None,
        author: "interpreted-provider".into(),
        kind: "interpreted-message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("concurrent interpreted event".into()),
        payload: json!({"interpreted": true}),
        anchors: json!([{
            "key": "session",
            "canonical_value": "claim-concurrency-session",
            "original_value": "claim-concurrency-session"
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
        max_events_per_batch: NonZeroU32::new(1).expect("nonzero row bound"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte bound"),
        lease_duration_ms: NonZeroU64::new(60_000).expect("nonzero lease duration"),
    }
}

fn rows(store: &Store, sql: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    store
        .execute_sql_for_test(sql, &params)
        .expect("inspect or control durable public-runner rows")
        .rows
}

fn capture(path: PathBuf) -> (Store, context_graph::ContextId) {
    let store = Store::open(config(path)).expect("open capture Store");
    let context_id = store
        .create_context(CreateContext {
            name: "interpretation claim concurrency".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "claim-concurrency-source",
            vec![EventDraft {
                session_ref: "claim-concurrency-session".into(),
                source_event_key: "real-source-key".into(),
                source_position: json!({"ordinal": 1}),
                occurred_at: None,
                author: "provider".into(),
                kind: "message".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some("real captured source".into()),
                payload: json!({"fixture": "concurrency"}),
                anchors: json!({"session": "claim-concurrency-session"}),
                source_remainder: json!({"retained": true}),
                coverage: EventCoverage::Complete,
            }],
        ))
        .expect("capture source through the public Store");
    (store, context_id)
}

#[derive(Clone, Debug, PartialEq)]
struct ClaimSnapshot {
    generation_id: uuid::Uuid,
    lease_token: String,
    lease_epoch: i64,
    build_ordinal: i64,
    claim_end_ordinal: i64,
    membership: Vec<Vec<Value>>,
}

fn live_claim(store: &Store, context_id: context_graph::ContextId) -> ClaimSnapshot {
    let generation = rows(
        store,
        "SELECT id, lease_token, lease_epoch, build_ordinal, claim_end_ordinal FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id AND state = $state AND lease_token IS NOT NULL AND claim_end_ordinal IS NOT NULL",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(CONNECTOR.into())),
            ("state".into(), Value::Text("building".into())),
        ]),
    );
    assert_eq!(generation.len(), 1, "one real runner-created live claim");
    let row = &generation[0];
    let generation_id = match row[0] {
        Value::Uuid(id) => id,
        ref value => panic!("real generation ID: {value:?}"),
    };
    let lease_token = match &row[1] {
        Value::Text(token) => token.clone(),
        value => panic!("real lease token: {value:?}"),
    };
    let lease_epoch = match row[2] {
        Value::Int64(epoch) => epoch,
        ref value => panic!("real lease epoch: {value:?}"),
    };
    let build_ordinal = match row[3] {
        Value::Int64(ordinal) => ordinal,
        ref value => panic!("real build ordinal: {value:?}"),
    };
    let claim_end_ordinal = match row[4] {
        Value::Int64(ordinal) => ordinal,
        ref value => panic!("real claim end ordinal: {value:?}"),
    };
    let membership = rows(
        store,
        "SELECT ordinal, event_id, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal >= $start AND ordinal < $end ORDER BY ordinal",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id)),
            ("start".into(), Value::Int64(build_ordinal)),
            ("end".into(), Value::Int64(claim_end_ordinal)),
        ]),
    );
    assert_eq!(membership.len(), 1, "real live claim owns one source row");
    ClaimSnapshot {
        generation_id,
        lease_token,
        lease_epoch,
        build_ordinal,
        claim_end_ordinal,
        membership,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RelevantRows {
    generation: Vec<Vec<Value>>,
    membership: Vec<Vec<Value>>,
    values: Vec<Vec<Value>>,
    anchors: Vec<Vec<Value>>,
    heads: Vec<Vec<Value>>,
}

fn relevant_rows(
    store: &Store,
    context_id: context_graph::ContextId,
    generation_id: uuid::Uuid,
) -> RelevantRows {
    let common = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("generation_id".into(), Value::Uuid(generation_id)),
    ]);
    RelevantRows {
        generation: rows(
            store,
            "SELECT id, context_id, connector_id, accepted_source_declarations, target_interpretation_version, target_adapter_version, target_anchor_declaration_version, mapping_contract_id, mapping_contract_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, build_ordinal, claim_end_ordinal, claim_membership_sha256, lease_epoch, lease_token, lease_expires_at, state, eligible_event_count, interpreted_event_count, failed_event_count, effective_content_sha256, sanitized_failure, started_at, completed_at, activated_at, activation_receipt FROM event_interpretation_generations WHERE context_id = $context_id AND id = $generation_id ORDER BY id",
            common.clone(),
        ),
        membership: rows(
            store,
            "SELECT context_id, generation_id, ordinal, event_id, source_instance_ref, source_sequence, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY ordinal, event_id",
            common.clone(),
        ),
        values: rows(
            store,
            "SELECT context_id, generation_id, event_id, captured_content_sha256, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, coverage, effective_sha256, created_at FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY event_id",
            common.clone(),
        ),
        anchors: rows(
            store,
            "SELECT id, context_id, generation_id, event_id, declaration_version, key, canonical_value, identity_value, original_value FROM event_interpretation_anchors WHERE context_id = $context_id AND generation_id = $generation_id ORDER BY event_id, declaration_version, key, identity_value, id",
            common.clone(),
        ),
        heads: rows(
            store,
            "SELECT context_id, connector_id, generation_id, activated_at FROM event_interpretation_heads WHERE context_id = $context_id ORDER BY connector_id, generation_id",
            HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        ),
    }
}

struct FirstRunnerMapper {
    path: PathBuf,
    context_id: context_graph::ContextId,
    paused: Sender<ClaimSnapshot>,
    resume: Mutex<Receiver<()>>,
}

impl ReleasedConnectorInterpretation for FirstRunnerMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        static DECLARATION: std::sync::OnceLock<ReleasedInterpretationDeclaration> =
            std::sync::OnceLock::new();
        DECLARATION.get_or_init(release)
    }

    fn interpret_event(&self, _: &ConversationEvent) -> Result<EffectiveEventFields> {
        let control = Store::open(config(self.path.clone())).expect("open mapper control Store");
        let claim = live_claim(&control, self.context_id);
        rows(
            &control,
            "UPDATE event_interpretation_generations SET lease_expires_at = $expired WHERE context_id = $context_id AND id = $generation_id",
            HashMap::from([
                ("expired".into(), Value::Timestamp(0)),
                ("context_id".into(), Value::Uuid(self.context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(claim.generation_id)),
            ]),
        );
        self.paused
            .send(claim)
            .expect("test receives the expired real A claim");
        self.resume
            .lock()
            .expect("A resume lock")
            .recv_timeout(CHANNEL_TIMEOUT)
            .expect("test releases A after B has reclaimed");
        Err(CgError::Engine(
            "late A mapper error after B reclaim".into(),
        ))
    }
}

struct SecondRunnerMapper {
    path: PathBuf,
    context_id: context_graph::ContextId,
    first: ClaimSnapshot,
    paused: Sender<ClaimSnapshot>,
    resume: Mutex<Receiver<()>>,
}

impl ReleasedConnectorInterpretation for SecondRunnerMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        static DECLARATION: std::sync::OnceLock<ReleasedInterpretationDeclaration> =
            std::sync::OnceLock::new();
        DECLARATION.get_or_init(release)
    }

    fn interpret_event(&self, _: &ConversationEvent) -> Result<EffectiveEventFields> {
        let control =
            Store::open(config(self.path.clone())).expect("open B mapper inspection Store");
        let claim = live_claim(&control, self.context_id);
        assert_eq!(
            claim.generation_id, self.first.generation_id,
            "B reclaims A's real generation"
        );
        assert_eq!(
            claim.membership, self.first.membership,
            "B reclaims A's real membership rows"
        );
        assert_eq!(
            claim.build_ordinal, self.first.build_ordinal,
            "B begins at A's real cursor"
        );
        assert_eq!(
            claim.claim_end_ordinal, self.first.claim_end_ordinal,
            "B keeps A's durable claim bound"
        );
        assert_ne!(
            claim.lease_token, self.first.lease_token,
            "B owns a renewed real lease token"
        );
        assert!(
            claim.lease_epoch > self.first.lease_epoch,
            "B owns a later lease epoch"
        );
        self.paused
            .send(claim)
            .expect("test receives B's reclaimed real lease");
        self.resume
            .lock()
            .expect("B resume lock")
            .recv_timeout(CHANNEL_TIMEOUT)
            .expect("test releases B to commit");
        Ok(effective())
    }
}

fn late_a_error_preserves_b_claim(b_completes_before_a_error: bool) {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("concurrent-claims.db");
    let (store, context_id) = capture(path.clone());
    drop(store);

    let (a_paused_tx, a_paused_rx) = channel();
    let (a_resume_tx, a_resume_rx) = channel();
    let first: Arc<dyn ReleasedConnectorInterpretation> = Arc::new(FirstRunnerMapper {
        path: path.clone(),
        context_id,
        paused: a_paused_tx,
        resume: Mutex::new(a_resume_rx),
    });
    let a_path = path.clone();
    let a = thread::spawn(move || {
        ConnectorInterpretationRunner::build(request(a_path, context_id, first))
    });
    let first_claim = a_paused_rx
        .recv_timeout(CHANNEL_TIMEOUT)
        .expect("A paused after expiring its real lease with test SQL");

    let (b_paused_tx, b_paused_rx) = channel();
    let (b_resume_tx, b_resume_rx) = channel();
    let second: Arc<dyn ReleasedConnectorInterpretation> = Arc::new(SecondRunnerMapper {
        path: path.clone(),
        context_id,
        first: first_claim,
        paused: b_paused_tx,
        resume: Mutex::new(b_resume_rx),
    });
    let b_path = path.clone();
    let mut b = Some(thread::spawn(move || {
        ConnectorInterpretationRunner::build(request(b_path, context_id, second))
    }));
    let b_claim = b_paused_rx
        .recv_timeout(CHANNEL_TIMEOUT)
        .expect("B reclaimed and paused on the renewed real lease");

    if b_completes_before_a_error {
        b_resume_tx.send(()).expect("release B to complete");
        b.take()
            .expect("B thread is still held by its mapper")
            .join()
            .expect("B runner thread joins")
            .expect("B completes its reclaimed claim before A returns an error");
    }

    let before_a_error = {
        let inspect =
            Store::open(config(path.clone())).expect("inspect B's real rows before A error");
        relevant_rows(&inspect, context_id, b_claim.generation_id)
    };
    a_resume_tx.send(()).expect("release A's late mapper error");
    let a_error = a.join().expect("A runner thread joins");
    assert!(
        matches!(a_error, Err(CgError::Engine(message)) if message == "late A mapper error after B reclaim")
    );
    let after_a_error = {
        let inspect =
            Store::open(config(path.clone())).expect("inspect B's real rows after A error");
        relevant_rows(&inspect, context_id, b_claim.generation_id)
    };

    let b_result_after_a_error = if !b_completes_before_a_error {
        b_resume_tx.send(()).expect("release B to complete");
        Some(
            b.take()
                .expect("B thread is still held by its mapper")
                .join()
                .expect("B runner thread joins"),
        )
    } else {
        None
    };

    assert_eq!(
        after_a_error, before_a_error,
        "A's late error leaves every relevant row of B's reclaimed lease or completed generation intact"
    );
    if let Some(result) = b_result_after_a_error {
        result.expect("B completes its reclaimed claim after A returns an error");
    }
}

#[test]
fn late_mapper_error_cannot_clear_a_reclaimed_live_lease_or_its_progress() {
    late_a_error_preserves_b_claim(false);
}

#[test]
fn late_mapper_error_cannot_turn_a_newer_completed_generation_failed() {
    late_a_error_preserves_b_claim(true);
}
