//! Prompt recovery of interrupted event-search projection.
//!
//! The user-facing contract: after a worker crash, kill or machine restart, the next run resumes
//! the pending projection at once — no lease to outwait, no cleanup — and redoes only the batch
//! that never committed. A worker that is genuinely alive (suspended, stopped, or computing with
//! its Store closed) is never taken over; a run that finds one waits for it to finish instead of
//! polling. A worker that lost ownership can never publish. Status says `interrupted`, not
//! `running`, for work nobody is doing. A store whose filesystem cannot take the holder lock
//! refuses with a typed error and writes nothing.
//!
//! Every proof uses a real process: a child copy of this test binary claims real work and then
//! exits at an exact cut, is stopped, or waits on its stdin pipe. Ordering comes from pipes,
//! channels and the projection's own commit hooks. Nothing sleeps and no clock is advanced.

#![cfg(feature = "test-support")]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::test_support::{
    after_next_event_search_batch_commit_for_test, before_next_background_claim_commit_for_test,
    before_next_event_search_batch_commit_for_test, fail_next_background_claim_presence_for_test,
    lose_held_claim_presence_for_test, observe_background_claim_waits_for_test,
    store_database_is_live_for_test,
};
use context_graph::{
    AppendEventBatch, BackgroundWork, BackgroundWorkState, CgError, ContextId, ContextSelector,
    CreateContext, DistanceMetric, Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput,
    EmbeddingSpace, EmbeddingSpaceBinding, EventAdapterDeclaration, EventCoverage, EventDraft,
    EventHistorySearchState, EventRenderRole, EventSearchBuildReceipt, EventSearchBuildRequest,
    EventSearchFailureDisposition, EventSearchProfile, EventSearchProjectionRunner,
    EventSearchRunRequest, EvidenceKind, EvidenceProducer, ForgetSelection, Result, Store,
    StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::persistence::holder_presence::{self, Holder};
use serde_json::json;

const CONNECTOR: &str = "background-claim-recovery";
const SESSION: &str = "background-claim-recovery-session";
const SPACE: &str = "background-claim-recovery-768";
const EVENT_SEARCH_WORK: &str = "event_search";

/// A hang guard for a rendezvous that must happen; the test fails rather than waits forever.
const RENDEZVOUS: Duration = Duration::from_secs(120);

const CHILD_ROLE: &str = "CG_BACKGROUND_CLAIM_CHILD_ROLE";
const CHILD_PATH: &str = "CG_BACKGROUND_CLAIM_CHILD_PATH";
const CHILD_CONTEXT: &str = "CG_BACKGROUND_CLAIM_CHILD_CONTEXT";
const CHILD_MARK: &str = "BACKGROUND-CLAIM-CHILD:";

/// Exit codes of the child's crash cuts.
const EXIT_AFTER_CLAIM: i32 = 72;
const EXIT_MID_EMBED: i32 = 73;
const EXIT_BEFORE_CLAIM_COMMIT: i32 = 75;
/// The child found its Store still open while it was embedding.
const STORE_OPEN_WHILE_EMBEDDING: i32 = 74;

type TextHook = Arc<dyn Fn(&str) + Send + Sync>;

/// Role-aware fixture tokenization (five model tokens plus two per scalar). `on_prepare` runs at
/// every document preparation, which happens after a batch is claimed and before it is embedded;
/// `on_embed` runs at every forward call.
struct FixtureEmbedder {
    on_prepare: Mutex<Option<TextHook>>,
    on_embed: Mutex<Option<TextHook>>,
    embedded: Mutex<Vec<String>>,
}

impl FixtureEmbedder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            on_prepare: Mutex::new(None),
            on_embed: Mutex::new(None),
            embedded: Mutex::new(Vec::new()),
        })
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn on_prepare(&self, hook: impl Fn(&str) + Send + Sync + 'static) {
        *self.on_prepare.lock().expect("prepare hook") = Some(Arc::new(hook));
    }

    fn on_embed(&self, hook: impl Fn(&str) + Send + Sync + 'static) {
        *self.on_embed.lock().expect("embed hook") = Some(Arc::new(hook));
    }

    fn embedded(&self) -> Vec<String> {
        self.embedded.lock().expect("embedded log").clone()
    }

    fn embed_one(&self, inference: &EventSearchInference) -> Result<EmbeddingOutput> {
        let text: String = inference
            .token_ids()
            .iter()
            .filter(|token| **token != 0)
            .map(|token| char::from_u32(*token).expect("fixture token is Unicode"))
            .collect();
        let hook = self.on_embed.lock().expect("embed hook").clone();
        if let Some(hook) = hook {
            hook(&text);
        }
        self.embedded.lock().expect("embedded log").push(text);
        Ok(vector(inference))
    }
}

fn vector(inference: &EventSearchInference) -> EmbeddingOutput {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for token in inference.token_ids() {
        hash.update(token.to_le_bytes());
    }
    let hash = hash.finalize();
    EmbeddingOutput {
        embedding_space_id: SPACE.into(),
        dimension: 768,
        producer: EvidenceProducer {
            system: CONNECTOR.into(),
            model_name: "recovery-fixture".into(),
            model_version: "1".into(),
            pipeline_version: "1".into(),
        },
        vector: (0..768)
            .map(|i| f32::from(hash[i % hash.len()]) / 255.0)
            .collect(),
    }
}

impl Embedder for FixtureEmbedder {
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
            model_name: "recovery-fixture".into(),
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
        "background-claim-recovery-v1".into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(256)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let hook = self.on_prepare.lock().expect("prepare hook").clone();
        if let Some(hook) = hook {
            hook(text);
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
        self.embed_one(inference)
    }

    fn embed_prepared_documents_hook(
        &self,
        inferences: &[EventSearchInference],
    ) -> Result<Vec<EmbeddingOutput>> {
        inferences
            .iter()
            .map(|inference| self.embed_one(inference))
            .collect()
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(vector(inference))
    }
}

fn config(path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        extra_embedding_spaces: vec![FixtureEmbedder::new().embedding_space()],
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
        adapter_version: "recovery-v1".into(),
        anchor_declaration_version: "recovery-v1".into(),
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
        interaction_ref: Some("recovery-pair".into()),
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

fn run_request(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<FixtureEmbedder>,
) -> EventSearchRunRequest {
    EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: profile(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
        max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
    }
}

fn run(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<FixtureEmbedder>,
) -> Result<EventSearchBuildReceipt> {
    EventSearchProjectionRunner::catch_up(run_request(config, context_id, embedder))
}

fn held_store_request(
    context_id: ContextId,
    embedder: &Arc<FixtureEmbedder>,
) -> EventSearchBuildRequest {
    EventSearchBuildRequest {
        context_id,
        embedder: embedder.registration(),
        profile: profile(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
        max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
    }
}

/// Two streams of two events each: every stream is one claimed batch and one passage, so the
/// first committed batch and the interrupted second batch are told apart by their markers.
fn two_stream_store(root: &Path, name: &str) -> (StoreConfig, ContextId) {
    let config = config(root.join(format!("{name}.db")));
    let store = open(&config);
    let context_id = store
        .create_context(CreateContext {
            name: name.into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
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
    (config, context_id)
}

/// Every current passage as (passage id, rendered text), in passage order.
fn current_passages(store: &Store, context_id: ContextId) -> Vec<(uuid::Uuid, String)> {
    store
        .execute_sql_for_test(
            "SELECT id, rendered_text, passage_ordinal FROM event_search_passages WHERE context_id = $context_id AND is_current = $current ORDER BY passage_ordinal ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("current".into(), Value::Bool(true)),
            ]),
        )
        .expect("inspect current passages")
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Uuid(id), Value::Text(text)) => (*id, text.clone()),
            other => panic!("unexpected passage row {other:?}"),
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

/// One durable claim row: the unit of work, who owns it and at which ownership epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ClaimRow {
    unit_id: uuid::Uuid,
    owner_token: uuid::Uuid,
    owner_epoch: i64,
}

fn claim_rows(store: &Store, context_id: ContextId) -> Vec<ClaimRow> {
    store
        .execute_sql_for_test(
            "SELECT unit_id, owner_token, owner_epoch FROM background_claims WHERE work = $work AND context_id = $context_id ORDER BY unit_id ASC",
            &HashMap::from([
                ("work".into(), Value::Text(EVENT_SEARCH_WORK.into())),
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ]),
        )
        .expect("inspect durable background claims")
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1], &row[2]) {
            (Value::Uuid(unit_id), Value::Uuid(owner_token), Value::Int64(owner_epoch)) => {
                ClaimRow {
                    unit_id: *unit_id,
                    owner_token: *owner_token,
                    owner_epoch: *owner_epoch,
                }
            }
            other => panic!("unexpected claim row {other:?}"),
        })
        .collect()
}

fn claim_rows_at(config: &StoreConfig, context_id: ContextId) -> Vec<ClaimRow> {
    claim_rows(&open(config), context_id)
}

/// Everything a projection publishes for the Context, so a refused publication can be proved to
/// have changed nothing.
#[derive(Debug, PartialEq)]
struct PublishedProjection {
    generations: Vec<Vec<Value>>,
    cursors: Vec<Vec<Value>>,
    passages: Vec<Vec<Value>>,
    claims: Vec<ClaimRow>,
}

fn published_projection(config: &StoreConfig, context_id: ContextId) -> PublishedProjection {
    let store = open(config);
    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    let rows = |sql: &str| {
        store
            .execute_sql_for_test(sql, &params)
            .expect("inspect published projection")
            .rows
    };
    PublishedProjection {
        generations: rows(
            "SELECT id, state, started_at, last_attempt_at, last_attempt_token, completed_at, sanitized_failure FROM event_search_generations WHERE context_id = $context_id ORDER BY id ASC",
        ),
        cursors: rows(
            "SELECT generation_id, connector_id, source_instance_ref, session_ref, committed_sequence FROM event_search_cursors WHERE context_id = $context_id ORDER BY generation_id ASC, source_instance_ref ASC",
        ),
        passages: rows(
            "SELECT id, rendered_text, is_current FROM event_search_passages WHERE context_id = $context_id ORDER BY id ASC",
        ),
        claims: claim_rows(&store, context_id),
    }
}

fn search_status(
    config: &StoreConfig,
    context_id: ContextId,
) -> context_graph::EventHistorySearchStatus {
    open(config)
        .event_history_status(ContextSelector::one(context_id))
        .expect("event history status")
        .contexts
        .remove(0)
        .search
}

/// The marker pair of the stream whose passage did NOT commit before an interruption.
fn pending_markers(committed: &[(uuid::Uuid, String)]) -> (char, char) {
    assert_eq!(
        committed.len(),
        1,
        "exactly the first claimed batch committed before the interruption: {committed:?}"
    );
    if committed[0].1.contains('α') {
        ('γ', 'δ')
    } else {
        ('α', 'β')
    }
}

/// After a resumed run: the committed passage is untouched, the interrupted batch is covered
/// exactly once, and nothing was rebuilt.
fn assert_resumed_exactly_once(
    config: &StoreConfig,
    context_id: ContextId,
    committed_before: &[(uuid::Uuid, String)],
    resumed_embeds: &[String],
) {
    let (first, second) = pending_markers(committed_before);
    assert_eq!(
        resumed_embeds.len(),
        1,
        "the resumed run embeds only the batch that never committed: {resumed_embeds:?}"
    );
    assert!(
        resumed_embeds[0].contains(first) && resumed_embeds[0].contains(second),
        "the resumed embedding is the interrupted batch: {resumed_embeds:?}"
    );
    let store = open(config);
    let current = current_passages(&store, context_id);
    assert_eq!(
        current.len(),
        2,
        "one current passage per stream: {current:?}"
    );
    assert!(
        current.contains(&committed_before[0]),
        "the committed passage keeps its identity and text: {current:?}"
    );
    assert_eq!(
        all_passage_count(&store, context_id),
        2,
        "no duplicate passage rows, current or retired"
    );
    assert!(
        claim_rows(&store, context_id).is_empty(),
        "a finished projection holds no claim"
    );
}

// ---------------------------------------------------------------------------------------------
// The child process: one copy of this binary that claims real work and stops at an exact cut.
// ---------------------------------------------------------------------------------------------

fn tell_parent(marker: &str) {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{CHILD_MARK}{marker}").expect("write to the parent pipe");
    stdout.flush().expect("flush the parent pipe");
}

/// Block until the parent writes one byte or closes the pipe.
fn wait_for_parent() {
    let mut byte = [0_u8; 1];
    let _ = std::io::stdin().read(&mut byte);
}

/// Arms `armed` once the first batch has committed, so a cut fires inside the second claim
/// whichever stream the runner happens to claim first.
fn arm_after_first_commit() -> Arc<AtomicBool> {
    let armed = Arc::new(AtomicBool::new(false));
    let arm = Arc::clone(&armed);
    after_next_event_search_batch_commit_for_test(move |_| arm.store(true, Ordering::SeqCst));
    armed
}

#[test]
fn background_claim_child() {
    let Ok(role) = std::env::var(CHILD_ROLE) else {
        return;
    };
    let path = PathBuf::from(std::env::var_os(CHILD_PATH).expect("child store path"));
    let config = config(path.clone());
    let context_id = ContextId::from(
        uuid::Uuid::parse_str(&std::env::var(CHILD_CONTEXT).expect("child Context"))
            .expect("child Context id"),
    );
    let embedder = FixtureEmbedder::new();
    match role.as_str() {
        "exit-after-claim" => {
            let armed = arm_after_first_commit();
            embedder.on_prepare(move |_| {
                if armed.load(Ordering::SeqCst) {
                    std::process::exit(EXIT_AFTER_CLAIM);
                }
            });
        }
        "exit-mid-embed" => {
            let armed = arm_after_first_commit();
            let path = path.clone();
            embedder.on_embed(move |_| {
                if armed.load(Ordering::SeqCst) {
                    if store_database_is_live_for_test(&path) {
                        std::process::exit(STORE_OPEN_WHILE_EMBEDDING);
                    }
                    std::process::exit(EXIT_MID_EMBED);
                }
            });
        }
        "exit-before-claim-commit" => {
            before_next_background_claim_commit_for_test(|_| {
                std::process::exit(EXIT_BEFORE_CLAIM_COMMIT)
            });
        }
        "hold-after-claim" | "hold-after-losing-lock" | "hold-at-first-claim" => {
            // `hold-at-first-claim` holds inside the first batch it claims, which is the taken-over
            // batch when an earlier worker already committed the other stream; the other roles
            // hold inside the batch claimed after their own first commit.
            let armed = if role == "hold-at-first-claim" {
                Arc::new(AtomicBool::new(true))
            } else {
                arm_after_first_commit()
            };
            let lose_lock = role == "hold-after-losing-lock";
            let path = path.clone();
            let held = AtomicBool::new(false);
            embedder.on_prepare(move |_| {
                if armed.load(Ordering::SeqCst) && !held.swap(true, Ordering::SeqCst) {
                    if lose_lock {
                        let lost = lose_held_claim_presence_for_test(&path);
                        tell_parent(&format!("LOST:{lost}"));
                    }
                    tell_parent("HOLDING");
                    wait_for_parent();
                }
            });
        }
        "wait-on-live-holder" => {
            observe_background_claim_waits_for_test(|work, _| {
                tell_parent(&format!("WAITING:{work:?}"));
            });
            before_next_background_claim_commit_for_test(|_| tell_parent("TAKEOVER-STAGED"));
        }
        other => panic!("unknown child role {other}"),
    }
    report_every_batch_commit(context_id);
    let outcome =
        EventSearchProjectionRunner::catch_up_observed(run_request(&config, context_id, &embedder));
    tell_parent(&format!("EMBEDDED:{}", embedder.embedded().len()));
    tell_parent(&match (&outcome.result, outcome.failure_disposition) {
        (Ok(_), _) => "OUTCOME:published".to_string(),
        (Err(_), disposition) => format!("OUTCOME:refused:{disposition:?}"),
    });
}

/// Tell the parent the claim epoch of every batch commit this thread makes.
fn report_every_batch_commit(context_id: ContextId) {
    before_next_event_search_batch_commit_for_test(move |store| {
        let epochs = claim_rows(store, context_id)
            .iter()
            .map(|claim| claim.owner_epoch.to_string())
            .collect::<Vec<_>>()
            .join(",");
        tell_parent(&format!("COMMITTING-AT-EPOCH:{epochs}"));
        report_every_batch_commit(context_id);
    });
}

/// A running child copy of this binary, read through its stdout pipe.
struct ClaimChild {
    process: Child,
    stdout: BufReader<ChildStdout>,
    reaped: bool,
}

impl ClaimChild {
    fn spawn(role: &str, config: &StoreConfig, context_id: ContextId) -> Self {
        let mut process = Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "background_claim_child", "--nocapture"])
            .env(CHILD_ROLE, role)
            .env(CHILD_PATH, &config.db_path)
            .env(CHILD_CONTEXT, context_id.as_uuid().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn the claiming child");
        let stdout = BufReader::new(process.stdout.take().expect("child stdout"));
        Self {
            process,
            stdout,
            reaped: false,
        }
    }

    fn pid(&self) -> i32 {
        i32::try_from(self.process.id()).expect("pid fits")
    }

    /// The next protocol line from the child, or `None` once the child closed its stdout.
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

    /// Read until `expected` arrives, returning every mark seen before it.
    fn expect_mark(&mut self, expected: &str) -> Vec<String> {
        let mut seen = Vec::new();
        loop {
            match self.next_mark() {
                Some(mark) if mark == expected => return seen,
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

    /// Wait for exit and return the exit status plus every mark the child still had to say.
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

impl Drop for ClaimChild {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.process.kill();
            let _ = self.process.wait();
        }
    }
}

/// Run a child to its crash cut and return the passages it committed before dying.
fn crash_child_at(
    role: &str,
    expected_code: i32,
    config: &StoreConfig,
    context_id: ContextId,
) -> Vec<(uuid::Uuid, String)> {
    let (status, marks) = ClaimChild::spawn(role, config, context_id).finish();
    assert_eq!(
        status.code(),
        Some(expected_code),
        "the child reached its exact crash cut `{role}`: {marks:?}"
    );
    current_passages(&open(config), context_id)
}

/// Record the owner epoch of the claim the next batch commit on this thread publishes under.
fn observe_commit_epochs(context_id: ContextId) -> Receiver<Vec<i64>> {
    let (sender, receiver) = channel();
    before_next_event_search_batch_commit_for_test(move |store| {
        let _ = sender.send(
            claim_rows(store, context_id)
                .iter()
                .map(|claim| claim.owner_epoch)
                .collect(),
        );
    });
    receiver
}

fn mark_value<'a>(marks: &'a [String], prefix: &str) -> Vec<&'a str> {
    marks
        .iter()
        .filter_map(|mark| mark.strip_prefix(prefix))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Proofs
// ---------------------------------------------------------------------------------------------

#[test]
fn a_worker_killed_after_its_claim_is_taken_over_at_once() {
    // A crashed worker's claim is taken over by the very next run; only its uncommitted batch repeats.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "killed-after-claim");
    let committed = crash_child_at("exit-after-claim", EXIT_AFTER_CLAIM, &config, context_id);
    let orphaned = claim_rows_at(&config, context_id);
    assert_eq!(
        orphaned
            .iter()
            .map(|claim| claim.owner_epoch)
            .collect::<Vec<_>>(),
        vec![1],
        "the dead worker left exactly its own first-epoch claim: {orphaned:?}"
    );

    let embedder = FixtureEmbedder::new();
    let epochs = observe_commit_epochs(context_id);
    run(&config, context_id, &embedder)
        .expect("the immediate rerun resumes the dead worker's claim");
    assert_eq!(
        epochs
            .try_recv()
            .expect("the rerun committed the interrupted batch"),
        vec![2],
        "the rerun published under the taken-over claim's next epoch"
    );
    assert_resumed_exactly_once(&config, context_id, &committed, &embedder.embedded());
}

#[test]
fn a_worker_killed_mid_embed_with_its_store_closed_resumes_without_waiting() {
    // A worker killed inside a forward call, with its Store closed, is resumed by the next run at once.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "killed-mid-embed");
    let committed = crash_child_at("exit-mid-embed", EXIT_MID_EMBED, &config, context_id);

    let embedder = FixtureEmbedder::new();
    let epochs = observe_commit_epochs(context_id);
    run(&config, context_id, &embedder).expect("the immediate rerun resumes the killed embedding");
    assert_eq!(
        epochs
            .try_recv()
            .expect("the rerun committed the interrupted batch"),
        vec![2]
    );
    assert_resumed_exactly_once(&config, context_id, &committed, &embedder.embedded());
}

#[test]
fn a_crash_between_presence_and_claim_commit_leaves_no_claim() {
    // A crash after taking the holder lock but before the claim commits leaves nothing to take over.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "crash-before-claim-commit");
    let committed = crash_child_at(
        "exit-before-claim-commit",
        EXIT_BEFORE_CLAIM_COMMIT,
        &config,
        context_id,
    );
    assert!(
        committed.is_empty(),
        "no batch was claimed, so none committed"
    );
    assert!(
        claim_rows_at(&config, context_id).is_empty(),
        "an uncommitted claim leaves no durable claim row"
    );

    let embedder = FixtureEmbedder::new();
    let epochs = observe_commit_epochs(context_id);
    run(&config, context_id, &embedder).expect("the next run claims afresh at once");
    assert_eq!(
        epochs
            .try_recv()
            .expect("the next run committed its first batch"),
        vec![1],
        "a fresh claim starts at the first epoch"
    );
    assert_eq!(
        embedder.embedded().len(),
        2,
        "both streams embed exactly once"
    );
    let store = open(&config);
    assert_eq!(current_passages(&store, context_id).len(), 2);
    assert!(claim_rows(&store, context_id).is_empty());
}

#[test]
fn a_live_worker_computing_with_its_store_closed_is_never_taken_over() {
    // A live worker is never taken over: a held-Store door is refused, a standalone run waits, then finishes.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "live-holder");
    let mut child = ClaimChild::spawn("hold-after-claim", &config, context_id);
    child.expect_mark("HOLDING");
    let held = claim_rows_at(&config, context_id);
    assert_eq!(held.len(), 1, "the live worker holds one claim: {held:?}");

    let door_embedder = FixtureEmbedder::new();
    let store = open(&config);
    let refusal = store
        .catch_up_event_search(held_store_request(context_id, &door_embedder))
        .expect_err("a held-Store door never takes over a live worker");
    assert!(
        matches!(
            &refusal,
            CgError::BackgroundWorkHeldByLiveWorker {
                work: BackgroundWork::EventSearch,
                context_id: refused,
                holder,
            } if *refused == context_id && *holder == held[0].owner_token
        ),
        "the refusal names the live holder: {refusal:?}"
    );
    drop(store);
    assert!(door_embedder.embedded().is_empty());
    let status = search_status(&config, context_id);
    assert_eq!(status.state, EventHistorySearchState::Running);
    assert_eq!(status.work, BackgroundWorkState::Running);
    assert_eq!(
        status.last_failure, None,
        "a refused door records no failure"
    );
    assert_eq!(
        claim_rows_at(&config, context_id),
        held,
        "the live claim is untouched"
    );

    let (waiting, waits) = channel();
    let waiter_embedder = FixtureEmbedder::new();
    let waiter_config = config.clone();
    let waiter_registration = Arc::clone(&waiter_embedder);
    let waiter = std::thread::spawn(move || {
        observe_background_claim_waits_for_test(move |work, waited| {
            let _ = waiting.send((work, waited));
        });
        run(&waiter_config, context_id, &waiter_registration)
    });
    assert_eq!(
        waits
            .recv_timeout(RENDEZVOUS)
            .expect("the standalone run starts waiting"),
        (BackgroundWork::EventSearch, context_id)
    );
    assert!(
        !waiter.is_finished(),
        "the standalone run waits while the holder lives"
    );
    assert_eq!(
        claim_rows_at(&config, context_id),
        held,
        "waiting takes nothing over"
    );

    child.release();
    let (status, marks) = child.finish();
    assert!(
        status.success(),
        "the live worker finishes its own claim: {marks:?}"
    );
    assert_eq!(
        mark_value(&marks, "COMMITTING-AT-EPOCH:"),
        vec!["1"],
        "the live worker publishes its held batch under its own first epoch"
    );
    assert_eq!(mark_value(&marks, "OUTCOME:"), vec!["published"]);
    waiter
        .join()
        .expect("the waiting run joins")
        .expect("the waiting run wakes when the holder releases and completes");
    assert!(
        waiter_embedder.embedded().is_empty(),
        "the waiting run re-embeds nothing the live worker published"
    );
    assert!(
        waits.try_recv().is_err(),
        "one event-driven wait, not a polling loop"
    );
    let store = open(&config);
    assert_eq!(current_passages(&store, context_id).len(), 2);
    assert_eq!(all_passage_count(&store, context_id), 2);
    assert!(claim_rows(&store, context_id).is_empty());
}

#[test]
fn a_stopped_process_keeps_its_claim() {
    // A SIGSTOPped worker is alive: its holder lock stays present and nothing takes its claim.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "stopped-holder");
    let mut child = ClaimChild::spawn("hold-after-claim", &config, context_id);
    child.expect_mark("HOLDING");
    let held = claim_rows_at(&config, context_id);
    assert_eq!(held.len(), 1);

    // SAFETY: the pid is this test's own live child.
    assert_eq!(unsafe { libc::kill(child.pid(), libc::SIGSTOP) }, 0);
    let mut wait_status = 0;
    // SAFETY: waiting on this test's own child for its stop report.
    assert_eq!(
        unsafe { libc::waitpid(child.pid(), &mut wait_status, libc::WUNTRACED) },
        child.pid()
    );
    assert!(libc::WIFSTOPPED(wait_status), "the child is stopped");

    assert!(
        matches!(
            holder_presence::probe(&config.db_path, *held[0].owner_token.as_bytes())
                .expect("probe the holder"),
            Holder::Present
        ),
        "a stopped holder is still present"
    );
    let store = open(&config);
    let refusal = store
        .catch_up_event_search(held_store_request(context_id, &FixtureEmbedder::new()))
        .expect_err("a stopped worker is never taken over");
    assert!(
        matches!(refusal, CgError::BackgroundWorkHeldByLiveWorker { .. }),
        "{refusal:?}"
    );
    drop(store);
    assert_eq!(
        claim_rows_at(&config, context_id),
        held,
        "the stopped worker keeps its claim"
    );
    assert_eq!(
        search_status(&config, context_id).work,
        BackgroundWorkState::Running
    );

    // SAFETY: the pid is this test's own stopped child.
    assert_eq!(unsafe { libc::kill(child.pid(), libc::SIGCONT) }, 0);
    child.release();
    let (status, marks) = child.finish();
    assert!(
        status.success(),
        "the resumed worker commits its own claim: {marks:?}"
    );
    assert_eq!(
        mark_value(&marks, "COMMITTING-AT-EPOCH:"),
        vec!["1"],
        "the stopped worker publishes its held batch under its own first epoch"
    );
    assert_eq!(mark_value(&marks, "OUTCOME:"), vec!["published"]);
    let store = open(&config);
    assert_eq!(current_passages(&store, context_id).len(), 2);
    assert!(claim_rows(&store, context_id).is_empty());
}

#[test]
fn a_rebooted_store_is_taken_over_immediately() {
    // After a restart no process holds a lock, so the copied store's claim is taken over at once.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "rebooted");
    let mut child = ClaimChild::spawn("hold-after-claim", &config, context_id);
    child.expect_mark("HOLDING");
    let held = claim_rows_at(&config, context_id);
    assert_eq!(held.len(), 1);
    let committed = current_passages(&open(&config), context_id);

    // Copying every file of the store reproduces the state a reboot leaves: the same durable rows
    // and companion files, and no kernel locks on any of them.
    let rebooted_root = tempfile::tempdir().expect("rebooted tempdir");
    for entry in std::fs::read_dir(root.path()).expect("list the store folder") {
        let entry = entry.expect("store folder entry");
        if entry.file_type().expect("entry type").is_file() {
            std::fs::copy(entry.path(), rebooted_root.path().join(entry.file_name()))
                .expect("copy one store file");
        }
    }
    let rebooted = StoreConfig {
        db_path: rebooted_root
            .path()
            .join(config.db_path.file_name().expect("store file name")),
        ..config.clone()
    };
    assert_eq!(
        claim_rows_at(&rebooted, context_id),
        held,
        "the copy carries the claim"
    );

    let embedder = FixtureEmbedder::new();
    let epochs = observe_commit_epochs(context_id);
    run(&rebooted, context_id, &embedder).expect("the rebooted store resumes at once");
    assert_eq!(
        epochs
            .try_recv()
            .expect("the rebooted run committed the interrupted batch"),
        vec![2]
    );
    assert_resumed_exactly_once(&rebooted, context_id, &committed, &embedder.embedded());

    child.release();
    let (status, marks) = child.finish();
    assert!(
        status.success(),
        "the original store's worker is unaffected: {marks:?}"
    );
    assert_eq!(mark_value(&marks, "OUTCOME:"), vec!["published"]);
}

#[test]
fn an_obsolete_worker_never_publishes_after_takeover() {
    // A worker that lost its lock while alive is taken over, and its late commit publishes nothing.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "obsolete-after-takeover");
    let mut child = ClaimChild::spawn("hold-after-losing-lock", &config, context_id);
    let before_holding = child.expect_mark("HOLDING");
    assert_eq!(
        mark_value(&before_holding, "LOST:"),
        vec!["1"],
        "the child released the kernel lock of its one claim"
    );
    let obsolete = claim_rows_at(&config, context_id);
    assert_eq!(obsolete.len(), 1);
    let committed = current_passages(&open(&config), context_id);

    let embedder = FixtureEmbedder::new();
    let epochs = observe_commit_epochs(context_id);
    run(&config, context_id, &embedder).expect("the lost claim is taken over and finished");
    assert_eq!(
        epochs
            .try_recv()
            .expect("the taker committed the interrupted batch"),
        vec![2]
    );
    assert_resumed_exactly_once(&config, context_id, &committed, &embedder.embedded());
    let after_takeover = published_projection(&config, context_id);

    child.release();
    let (status, marks) = child.finish();
    assert!(status.success(), "{marks:?}");
    assert_eq!(
        mark_value(&marks, "OUTCOME:"),
        vec!["refused:Stale"],
        "the obsolete worker is fenced out as stale and records no failure"
    );
    assert_eq!(
        published_projection(&config, context_id),
        after_takeover,
        "the obsolete worker's late commit changed nothing"
    );
    assert!(
        claim_rows_at(&config, context_id)
            .iter()
            .all(|claim| claim.owner_token != obsolete[0].owner_token),
        "no row carries the obsolete worker's token"
    );
}

#[test]
fn an_obsolete_worker_never_publishes_after_forget_deletes_its_claim() {
    // Forgetting the Context deletes a live worker's claim, and the worker's late commit publishes nothing.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "obsolete-after-forget");
    let mut child = ClaimChild::spawn("hold-after-claim", &config, context_id);
    child.expect_mark("HOLDING");
    let held = claim_rows_at(&config, context_id);
    assert_eq!(held.len(), 1);

    open(&config)
        .forget_event_history(ForgetSelection::Context { context_id })
        .expect("forget the Context while its worker computes");
    assert!(
        claim_rows_at(&config, context_id).is_empty(),
        "forget deletes the Context's claims"
    );
    let after_forget = published_projection(&config, context_id);

    child.release();
    let (status, marks) = child.finish();
    assert!(status.success(), "{marks:?}");
    assert_eq!(mark_value(&marks, "OUTCOME:"), vec!["refused:Stale"]);
    assert_eq!(
        published_projection(&config, context_id),
        after_forget,
        "nothing reappears after forget"
    );
    assert!(current_passages(&open(&config), context_id).is_empty());
}

#[test]
fn concurrent_takeovers_between_threads_admit_exactly_one() {
    // Two runs racing to take over one dead worker's claim: exactly one wins, the other waits.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "thread-takeovers");
    let committed = crash_child_at("exit-after-claim", EXIT_AFTER_CLAIM, &config, context_id);
    let (pending_first, _) = pending_markers(&committed);

    // The winner holds inside its preparation until the loser is seen waiting on it, so the loser
    // meets a live holder whichever thread wins.
    let (open_gate, gate) = channel::<()>();
    let gate = Mutex::new(gate);
    let embedder = FixtureEmbedder::new();
    embedder.on_prepare(move |text| {
        if text.contains(pending_first) {
            gate.lock()
                .expect("gate")
                .recv_timeout(RENDEZVOUS)
                .expect("the losing run is seen waiting");
        }
    });

    let (arrived, arrivals) = channel::<()>();
    let (waited, waits) = channel::<(BackgroundWork, ContextId)>();
    let (committed_at, commits) = channel::<Vec<i64>>();
    let mut releases = Vec::new();
    let mut runners = Vec::new();
    for _ in 0..2 {
        let (release, released) = channel::<()>();
        releases.push(release);
        let arrived = arrived.clone();
        let waited = waited.clone();
        let committed_at = committed_at.clone();
        let config = config.clone();
        let embedder = Arc::clone(&embedder);
        runners.push(std::thread::spawn(move || {
            before_next_background_claim_commit_for_test(move |_| {
                arrived
                    .send(())
                    .expect("report arrival at the takeover commit");
                released
                    .recv_timeout(RENDEZVOUS)
                    .expect("both takeovers are staged before either commits");
            });
            observe_background_claim_waits_for_test(move |work, waited_on| {
                let _ = waited.send((work, waited_on));
            });
            before_next_event_search_batch_commit_for_test(move |store| {
                let _ = committed_at.send(
                    claim_rows(store, context_id)
                        .iter()
                        .map(|claim| claim.owner_epoch)
                        .collect(),
                );
            });
            run(&config, context_id, &embedder)
        }));
    }
    for _ in 0..2 {
        arrivals
            .recv_timeout(RENDEZVOUS)
            .expect("both runs stage a takeover of the dead worker's claim concurrently");
    }
    for release in &releases {
        release.send(()).expect("release one staged takeover");
    }
    assert_eq!(
        waits
            .recv_timeout(RENDEZVOUS)
            .expect("the losing run waits on the winner"),
        (BackgroundWork::EventSearch, context_id)
    );
    open_gate.send(()).expect("let the winner finish");
    for runner in runners {
        runner
            .join()
            .expect("runner joins")
            .expect("winner and waiter both complete");
    }
    assert_eq!(
        commits.try_iter().collect::<Vec<_>>(),
        vec![vec![2]],
        "exactly one takeover published, at the next epoch"
    );
    assert!(waits.try_recv().is_err(), "only the loser waited, once");
    assert_resumed_exactly_once(&config, context_id, &committed, &embedder.embedded());
}

#[test]
fn concurrent_takeovers_between_processes_admit_exactly_one() {
    // A second process meeting the process that took over a dead claim waits instead of taking it.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "process-takeovers");
    let committed = crash_child_at("exit-after-claim", EXIT_AFTER_CLAIM, &config, context_id);

    let mut winner = ClaimChild::spawn("hold-at-first-claim", &config, context_id);
    winner.expect_mark("HOLDING");
    let taken = claim_rows_at(&config, context_id);
    assert_eq!(
        taken
            .iter()
            .map(|claim| claim.owner_epoch)
            .collect::<Vec<_>>(),
        vec![2],
        "the first process took the dead claim over"
    );

    let mut loser = ClaimChild::spawn("wait-on-live-holder", &config, context_id);
    let before_wait = loser.expect_mark("WAITING:EventSearch");
    assert!(
        !before_wait.iter().any(|mark| mark == "TAKEOVER-STAGED"),
        "the second process never stages a takeover of a live holder: {before_wait:?}"
    );
    assert_eq!(claim_rows_at(&config, context_id), taken);

    winner.release();
    let (winner_status, winner_marks) = winner.finish();
    assert!(winner_status.success(), "{winner_marks:?}");
    assert_eq!(mark_value(&winner_marks, "COMMITTING-AT-EPOCH:"), vec!["2"]);
    assert_eq!(mark_value(&winner_marks, "EMBEDDED:"), vec!["1"]);
    let (loser_status, loser_marks) = loser.finish();
    assert!(loser_status.success(), "{loser_marks:?}");
    assert!(
        mark_value(&loser_marks, "COMMITTING-AT-EPOCH:").is_empty()
            && !loser_marks.iter().any(|mark| mark == "TAKEOVER-STAGED"),
        "the waiting process wakes to finished work: {loser_marks:?}"
    );
    assert_eq!(mark_value(&loser_marks, "EMBEDDED:"), vec!["0"]);
    assert_eq!(mark_value(&loser_marks, "OUTCOME:"), vec!["published"]);
    let store = open(&config);
    let current = current_passages(&store, context_id);
    assert_eq!(current.len(), 2);
    assert!(current.contains(&committed[0]));
    assert_eq!(all_passage_count(&store, context_id), 2);
    assert!(claim_rows(&store, context_id).is_empty());
}

#[test]
fn a_worker_that_lost_its_lock_in_process_is_fenced_out_and_its_vector_never_becomes_coverage() {
    // Losing the lock mid-embed lets another run take over; the first run's late vector is discarded.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "in-process-fence");
    let first = FixtureEmbedder::new();
    let taker = FixtureEmbedder::new();
    let path = config.db_path.clone();
    let taker_config = config.clone();
    let taker_embedder = Arc::clone(&taker);
    let taken_over = Arc::new(Mutex::new(None));
    let taken_over_result = Arc::clone(&taken_over);
    let fired = AtomicBool::new(false);
    let armed = arm_after_first_commit();
    first.on_embed(move |_| {
        if armed.load(Ordering::SeqCst) && !fired.swap(true, Ordering::SeqCst) {
            assert_eq!(lose_held_claim_presence_for_test(&path), 1);
            let config = taker_config.clone();
            let embedder = Arc::clone(&taker_embedder);
            let result = std::thread::spawn(move || run(&config, context_id, &embedder))
                .join()
                .expect("the taker joins");
            *taken_over_result.lock().expect("taker result") = Some(result);
        }
    });
    let outcome =
        EventSearchProjectionRunner::catch_up_observed(run_request(&config, context_id, &first));
    taken_over
        .lock()
        .expect("taker result")
        .take()
        .expect("the taker ran")
        .expect("the taker finished the interrupted batch");
    assert!(
        outcome.result.is_err(),
        "the fenced-out run reports it lost ownership"
    );
    assert_eq!(
        outcome.failure_disposition,
        EventSearchFailureDisposition::Stale,
        "a fenced-out run never marks the taker's generation failed"
    );
    assert_eq!(
        taker.embedded().len(),
        1,
        "the taker embedded the interrupted batch once"
    );
    let store = open(&config);
    assert_eq!(current_passages(&store, context_id).len(), 2);
    assert_eq!(
        all_passage_count(&store, context_id),
        2,
        "the discarded vector never becomes duplicate coverage"
    );
    assert!(claim_rows(&store, context_id).is_empty());
    assert_eq!(search_status(&config, context_id).last_failure, None);
}

#[test]
fn a_filesystem_without_locks_refuses_the_claim_and_writes_nothing() {
    // A store that cannot take the holder lock refuses with a typed error and writes nothing.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "no-lock-filesystem");
    let before = published_projection(&config, context_id);

    const EOPNOTSUPP: i32 = 95;
    let embedder = FixtureEmbedder::new();
    fail_next_background_claim_presence_for_test(EOPNOTSUPP);
    let refusal = run(&config, context_id, &embedder)
        .expect_err("a standalone run cannot claim without a holder lock");
    assert!(
        matches!(&refusal, CgError::BackgroundClaimUnavailable { path, .. }
            if path.contains(config.db_path.file_name().unwrap().to_str().unwrap())),
        "the refusal names the store that cannot lock: {refusal:?}"
    );
    assert!(embedder.embedded().is_empty(), "nothing was embedded");
    assert_eq!(
        published_projection(&config, context_id),
        before,
        "a refused claim writes no generation, attempt, cursor, passage or claim"
    );

    fail_next_background_claim_presence_for_test(EOPNOTSUPP);
    let store = open(&config);
    let refusal = store
        .catch_up_event_search(held_store_request(context_id, &embedder))
        .expect_err("a held-Store door cannot claim without a holder lock");
    assert!(
        matches!(refusal, CgError::BackgroundClaimUnavailable { .. }),
        "{refusal:?}"
    );
    drop(store);
    assert_eq!(published_projection(&config, context_id), before);

    run(&config, context_id, &embedder).expect("a lockable store claims normally");
    assert_eq!(current_passages(&open(&config), context_id).len(), 2);
}

#[test]
fn status_reports_interrupted_not_running_after_a_crash() {
    // Status says running while a worker lives, interrupted once it is killed, idle once resumed.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "status-after-crash");
    let mut child = ClaimChild::spawn("hold-after-claim", &config, context_id);
    child.expect_mark("HOLDING");
    let live = search_status(&config, context_id);
    assert_eq!(live.state, EventHistorySearchState::Running);
    assert_eq!(live.work, BackgroundWorkState::Running);

    // SAFETY: the pid is this test's own live child; SIGKILL leaves no chance to release anything.
    assert_eq!(unsafe { libc::kill(child.pid(), libc::SIGKILL) }, 0);
    let (status, _) = child.finish();
    assert!(!status.success(), "the worker was killed");

    let interrupted = search_status(&config, context_id);
    assert_eq!(
        interrupted.state,
        EventHistorySearchState::Interrupted,
        "work nobody is doing is never reported as running"
    );
    assert_eq!(interrupted.work, BackgroundWorkState::Interrupted);
    assert_eq!(
        interrupted.last_failure, None,
        "a crash is not a recorded failure"
    );
    assert!(interrupted.pending_events > 0);

    let assessed = open(&config)
        .event_search_work(
            ContextSelector::one(context_id),
            &context_graph::EventSearchWorkSpec {
                profile: profile(),
                embedding_space_id: SPACE.into(),
                encoding_profile_id: "background-claim-recovery-v1".into(),
            },
        )
        .expect("assess pending work");
    assert_eq!(assessed[0].work, BackgroundWorkState::Interrupted);
    assert!(assessed[0].work_required);

    run(&config, context_id, &FixtureEmbedder::new()).expect("the next run resumes at once");
    let ready = search_status(&config, context_id);
    assert_eq!(ready.state, EventHistorySearchState::Ready);
    assert_eq!(ready.work, BackgroundWorkState::Idle);
}

#[test]
fn a_worker_that_panics_mid_batch_is_taken_over_at_once() {
    // A worker that unwinds out of a batch leaves no claim anyone has to wait out.
    let root = tempfile::tempdir().expect("tempdir");
    let (config, context_id) = two_stream_store(root.path(), "panicked-worker");
    let dying = FixtureEmbedder::new();
    dying.on_prepare(|_| panic!("fixture worker unwinds mid-batch"));
    let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(&config, context_id, &dying)
    }));
    assert!(died.is_err(), "the fixture worker died mid-batch");
    assert_ne!(
        search_status(&config, context_id).work,
        BackgroundWorkState::Running,
        "a worker that unwound is not reported as running"
    );

    let embedder = FixtureEmbedder::new();
    let receipt = run(&config, context_id, &embedder).expect("the next run claims at once");
    assert_eq!(receipt.embedded_event_count, 4);
    assert_eq!(embedder.embedded().len(), 2);
    assert!(claim_rows_at(&config, context_id).is_empty());
}
