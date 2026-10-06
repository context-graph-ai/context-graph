//! Prompt recovery of an interrupted connector interpretation build — the second kind of
//! background work, recovered by the same shared claim mechanism as search projection.
//!
//! The user-facing contract: a build whose process dies after claiming a batch is taken over by
//! the next build at once, and only the uncommitted batch is mapped again; a build that meets a
//! live builder waits for it to finish instead of refusing or polling; status reports work nobody
//! is doing as interrupted.
//!
//! The interrupted builder is a real child copy of this test binary. It exits or waits inside the
//! released mapper, which the runner calls after committing its claim and closing its Store.
//! Ordering comes from pipes and channels; nothing sleeps and no clock is advanced.

#![cfg(feature = "test-support")]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::test_support::observe_background_claim_waits_for_test;
use context_graph::{
    AppendEventBatch, BackgroundWork, BackgroundWorkState, CapturedEventAndAnchors,
    ConnectorInterpretationRunner, ContextId, ContextSelector, ConversationEvent, CreateContext,
    EffectiveEventFields, EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole,
    InstalledInterpretationContract, InterpretationBuildRequest, InterpretationGenerationState,
    ReleasedConnectorInterpretation, ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "interpretation-crash-recovery";
const TARGET: &str = "crash-recovery-v1";
const SESSION: &str = "interpretation-crash-recovery-session";
const INTERPRETATION_WORK: &str = "interpretation";
const EVENTS: usize = 3;
const RENDEZVOUS: Duration = Duration::from_secs(120);

const CHILD_ROLE: &str = "CG_INTERPRETATION_CRASH_CHILD_ROLE";
const CHILD_PATH: &str = "CG_INTERPRETATION_CRASH_CHILD_PATH";
const CHILD_CONTEXT: &str = "CG_INTERPRETATION_CRASH_CHILD_CONTEXT";
const CHILD_MARK: &str = "INTERPRETATION-CRASH-CHILD:";
const EXIT_AFTER_CLAIM: i32 = 72;

fn config(path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn open(config: &StoreConfig) -> Store {
    Store::open(config.clone()).expect("open Store")
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "crash-recovery-capture-v1".into(),
        anchor_declaration_version: "crash-recovery-anchors-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn release() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        target_adapter_version: "crash-recovery-capture-v1".into(),
        target_anchor_declaration_version: "crash-recovery-anchors-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "crash-recovery-capture-v1".into(),
            anchor_declaration_version: "crash-recovery-anchors-v1".into(),
        }],
        mapping_contract_id: "crash-recovery-contract".into(),
        mapping_contract_sha256: "crash-recovery-contract-sha256".into(),
    }
}

fn effective() -> EffectiveEventFields {
    EffectiveEventFields {
        occurred_at: None,
        author: "interpreted-provider".into(),
        kind: "interpreted-message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("recovered interpreted event".into()),
        payload: json!({"interpreted": true}),
        anchors: json!([{
            "key": "session",
            "canonical_value": SESSION,
            "original_value": SESSION
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

type MapperHook = Box<dyn Fn(usize) + Send + Sync>;

/// A released mapper that counts its calls and runs `hook(call_number)` before answering.
struct CountingMapper {
    calls: AtomicUsize,
    hook: MapperHook,
}

impl CountingMapper {
    fn new(hook: impl Fn(usize) + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            hook: Box::new(hook),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ReleasedConnectorInterpretation for CountingMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        static DECLARATION: std::sync::OnceLock<ReleasedInterpretationDeclaration> =
            std::sync::OnceLock::new();
        DECLARATION.get_or_init(release)
    }

    fn interpret_event(&self, _: &ConversationEvent) -> Result<EffectiveEventFields> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        (self.hook)(call);
        Ok(effective())
    }
}

fn request(
    config: &StoreConfig,
    context_id: ContextId,
    mapper: Arc<dyn ReleasedConnectorInterpretation>,
) -> InterpretationBuildRequest {
    InterpretationBuildRequest {
        store: config.clone(),
        context_id,
        connector_id: CONNECTOR.into(),
        target_interpretation_version: TARGET.into(),
        mapper,
        contract: Arc::new(FixtureContract),
        max_events_per_batch: NonZeroU32::new(1).expect("one event per claimed batch"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte bound"),
    }
}

/// Three captured events: each is one claimed batch.
fn captured_store(root: &std::path::Path, name: &str) -> (StoreConfig, ContextId) {
    let config = config(root.join(format!("{name}.db")));
    let store = open(&config);
    let context_id = store
        .create_context(CreateContext {
            name: name.into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "crash-recovery-source",
            (1..=EVENTS)
                .map(|ordinal| EventDraft {
                    session_ref: SESSION.into(),
                    source_event_key: format!("source-key-{ordinal}"),
                    source_position: json!({"ordinal": ordinal}),
                    occurred_at: None,
                    author: "provider".into(),
                    kind: "message".into(),
                    render_role: EventRenderRole::HumanMessage,
                    interaction_ref: None,
                    text: Some(format!("captured source {ordinal}")),
                    payload: json!({"fixture": ordinal}),
                    anchors: json!({"session": SESSION}),
                    source_remainder: json!({"retained": true}),
                    coverage: EventCoverage::Complete,
                })
                .collect(),
        ))
        .expect("capture source events");
    (config, context_id)
}

fn claim_epochs(store: &Store, context_id: ContextId) -> Vec<i64> {
    store
        .execute_sql_for_test(
            "SELECT owner_epoch FROM background_claims WHERE work = $work AND context_id = $context_id",
            &HashMap::from([
                ("work".into(), Value::Text(INTERPRETATION_WORK.into())),
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ]),
        )
        .expect("inspect interpretation claims")
        .rows
        .iter()
        .map(|row| match row[0] {
            Value::Int64(epoch) => epoch,
            ref other => panic!("unexpected claim epoch {other:?}"),
        })
        .collect()
}

fn committed_value_count(config: &StoreConfig, context_id: ContextId) -> usize {
    open(config)
        .execute_sql_for_test(
            "SELECT event_id FROM event_interpretation_values WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect committed values")
        .rows
        .len()
}

fn interpretation_work(config: &StoreConfig, context_id: ContextId) -> BackgroundWorkState {
    open(config)
        .interpretation_status(ContextSelector::one(context_id), CONNECTOR.into())
        .expect("interpretation status")
        .contexts
        .remove(0)
        .work
}

fn tell_parent(marker: &str) {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{CHILD_MARK}{marker}").expect("write to the parent pipe");
    stdout.flush().expect("flush the parent pipe");
}

#[test]
fn interpretation_crash_child() {
    let Ok(role) = std::env::var(CHILD_ROLE) else {
        return;
    };
    let config = config(PathBuf::from(
        std::env::var_os(CHILD_PATH).expect("child store path"),
    ));
    let context_id = ContextId::from(
        uuid::Uuid::parse_str(&std::env::var(CHILD_CONTEXT).expect("child Context"))
            .expect("child Context id"),
    );
    // The second call runs inside the second claimed batch, after the first batch committed.
    let mapper = CountingMapper::new(move |call| {
        if call != 2 {
            return;
        }
        match role.as_str() {
            "exit-after-claim" => std::process::exit(EXIT_AFTER_CLAIM),
            "hold-after-claim" => {
                tell_parent("HOLDING");
                let mut byte = [0_u8; 1];
                let _ = std::io::stdin().read(&mut byte);
            }
            other => panic!("unknown child role {other}"),
        }
    });
    let receipt =
        ConnectorInterpretationRunner::build(request(&config, context_id, mapper.clone()))
            .expect("the held builder finishes its own claim");
    tell_parent(&format!("MAPPED:{}", mapper.calls()));
    tell_parent(&format!("INTERPRETED:{}", receipt.interpreted_event_count));
}

struct BuilderChild {
    process: Child,
    stdout: BufReader<ChildStdout>,
    reaped: bool,
}

impl BuilderChild {
    fn spawn(role: &str, config: &StoreConfig, context_id: ContextId) -> Self {
        let mut process = Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "interpretation_crash_child", "--nocapture"])
            .env(CHILD_ROLE, role)
            .env(CHILD_PATH, &config.db_path)
            .env(CHILD_CONTEXT, context_id.as_uuid().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn the interpretation child");
        let stdout = BufReader::new(process.stdout.take().expect("child stdout"));
        Self {
            process,
            stdout,
            reaped: false,
        }
    }

    fn next_mark(&mut self) -> Option<String> {
        let mut line = String::new();
        loop {
            line.clear();
            if self
                .stdout
                .read_line(&mut line)
                .expect("read the child pipe")
                == 0
            {
                return None;
            }
            if let Some(mark) = line.trim_end().strip_prefix(CHILD_MARK) {
                return Some(mark.to_string());
            }
        }
    }

    fn expect_mark(&mut self, expected: &str) {
        let mut seen = Vec::new();
        loop {
            match self.next_mark() {
                Some(mark) if mark == expected => return,
                Some(mark) => seen.push(mark),
                None => panic!("the child exited before {expected}; it said {seen:?}"),
            }
        }
    }

    fn release(&mut self) {
        let stdin = self.process.stdin.as_mut().expect("child stdin");
        stdin.write_all(b"\n").expect("release the child");
        stdin.flush().expect("flush the child release");
    }

    fn finish(mut self) -> (ExitStatus, Vec<String>) {
        let mut marks = Vec::new();
        while let Some(mark) = self.next_mark() {
            marks.push(mark);
        }
        let status = self.process.wait().expect("wait for the child");
        self.reaped = true;
        (status, marks)
    }
}

impl Drop for BuilderChild {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.process.kill();
            let _ = self.process.wait();
        }
    }
}

fn crash_builder_after_its_second_claim(config: &StoreConfig, context_id: ContextId) {
    let (status, marks) = BuilderChild::spawn("exit-after-claim", config, context_id).finish();
    assert_eq!(
        status.code(),
        Some(EXIT_AFTER_CLAIM),
        "the builder died inside its second claimed batch: {marks:?}"
    );
    assert_eq!(
        committed_value_count(config, context_id),
        1,
        "the first claimed batch committed before the crash"
    );
    assert_eq!(
        claim_epochs(&open(config), context_id),
        vec![1],
        "the dead builder left its own first-epoch claim"
    );
}

#[test]
fn an_interpretation_build_killed_after_its_claim_is_taken_over_at_once() {
    // The next build takes over a dead builder's claim at once and maps only the uncommitted batch.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = captured_store(root.path(), "killed-builder");
    crash_builder_after_its_second_claim(&config, context_id);

    let epochs = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&epochs);
    let path = config.clone();
    let mapper = CountingMapper::new(move |_| {
        let control = open(&path);
        observed
            .lock()
            .expect("epochs")
            .extend(claim_epochs(&control, context_id));
    });
    let receipt =
        ConnectorInterpretationRunner::build(request(&config, context_id, mapper.clone()))
            .expect("the immediate rebuild resumes the dead builder's claim");
    assert_eq!(
        mapper.calls(),
        EVENTS - 1,
        "only the batches that never committed are mapped again"
    );
    assert_eq!(
        epochs.lock().expect("epochs").first(),
        Some(&2),
        "the interrupted batch is mapped under the taken-over claim's next epoch"
    );
    assert_eq!(receipt.state, InterpretationGenerationState::Ready);
    assert_eq!(receipt.interpreted_event_count, EVENTS as u64);
    assert_eq!(
        committed_value_count(&config, context_id),
        EVENTS,
        "one value set per event, none duplicated"
    );
    assert!(claim_epochs(&open(&config), context_id).is_empty());
}

#[test]
fn an_interpretation_build_waits_on_a_live_holder_then_finishes() {
    // A build that meets a live builder waits for it, then finishes without remapping its work.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = captured_store(root.path(), "live-builder");
    let mut child = BuilderChild::spawn("hold-after-claim", &config, context_id);
    child.expect_mark("HOLDING");
    assert_eq!(
        interpretation_work(&config, context_id),
        BackgroundWorkState::Running
    );

    let (waiting, waits) = channel();
    let mapper = CountingMapper::new(|_| {});
    let waiter_mapper = mapper.clone();
    let waiter_config = config.clone();
    let waiter = std::thread::spawn(move || {
        observe_background_claim_waits_for_test(move |work, waited| {
            let _ = waiting.send((work, waited));
        });
        ConnectorInterpretationRunner::build(request(&waiter_config, context_id, waiter_mapper))
    });
    assert_eq!(
        waits
            .recv_timeout(RENDEZVOUS)
            .expect("the second build starts waiting"),
        (BackgroundWork::Interpretation, context_id)
    );
    assert!(
        !waiter.is_finished(),
        "the second build waits while the holder lives"
    );
    assert_eq!(
        claim_epochs(&open(&config), context_id),
        vec![1],
        "nothing was taken over"
    );

    child.release();
    let (status, marks) = child.finish();
    assert!(
        status.success(),
        "the live builder finishes its own claim: {marks:?}"
    );
    assert!(
        marks.contains(&format!("INTERPRETED:{EVENTS}")),
        "{marks:?}"
    );
    let receipt = waiter
        .join()
        .expect("the waiting build joins")
        .expect("the waiting build wakes when the holder releases and completes");
    assert_eq!(mapper.calls(), 0, "the waiting build remaps nothing");
    assert_eq!(receipt.state, InterpretationGenerationState::Ready);
    assert!(waits.try_recv().is_err(), "one event-driven wait");
    assert_eq!(committed_value_count(&config, context_id), EVENTS);
}

#[test]
fn interpretation_status_reports_interrupted_after_a_crash() {
    // Interpretation status says running while a builder lives, interrupted once it dies, idle after resume.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = captured_store(root.path(), "interpretation-status");
    let mut child = BuilderChild::spawn("hold-after-claim", &config, context_id);
    child.expect_mark("HOLDING");
    assert_eq!(
        interpretation_work(&config, context_id),
        BackgroundWorkState::Running
    );

    // SAFETY: the pid is this test's own live child; SIGKILL leaves no chance to release anything.
    assert_eq!(
        unsafe {
            libc::kill(
                i32::try_from(child.process.id()).expect("pid fits"),
                libc::SIGKILL,
            )
        },
        0
    );
    let (status, _) = child.finish();
    assert!(!status.success(), "the builder was killed");
    assert_eq!(
        interpretation_work(&config, context_id),
        BackgroundWorkState::Interrupted,
        "a dead builder's work is interrupted, never running"
    );

    ConnectorInterpretationRunner::build(request(&config, context_id, CountingMapper::new(|_| {})))
        .expect("the next build resumes at once");
    assert_eq!(
        interpretation_work(&config, context_id),
        BackgroundWorkState::Idle
    );
}
