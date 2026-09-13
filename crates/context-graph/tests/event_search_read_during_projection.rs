//! Reads are never refused as busy while a projection worker holds the Store (assertion 67a).
//!
//! The worker here is a plain writable Store, exactly as the detached projection worker and an
//! import open it: no administrative handler. The owner channel lives in an explicit runtime
//! directory, as the CLI places it. Reads run from inside the worker's final head-switch
//! transaction through the internal finalization pause.

#![cfg(context_graph_internal_test_support)]

use std::cell::RefCell;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::test_support::pause_next_event_search_finalization_for_test;
use context_graph::{
    AppendEventBatch, ConsumerReader, ContextId, ContextSelector, CreateContext, DistanceMetric,
    Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace,
    EmbeddingSpaceBinding, EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole,
    EventSearchBuildRequest, EventSearchFilter, EventSearchProfile, EvidenceKind, EvidenceProducer,
    Result, Store, StoreConfig, event_history_reader_declaration,
};
use serde_json::json;

const CONNECTOR: &str = "read-during-projection";
const SPACE: &str = "read-during-projection-768";

/// The owner channel's directory for this whole test process, set before any Store opens.
fn owner_runtime_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = tempfile::tempdir().expect("owner runtime dir").keep();
        // The owner plane accepts only an owner-only runtime root.
        std::fs::set_permissions(&dir, std::os::unix::fs::PermissionsExt::from_mode(0o700))
            .expect("owner-only runtime dir");
        // SAFETY: set once, before any Store or reader in this process reads the environment.
        unsafe { std::env::set_var("CONTEXTDB_OWNER_READ_RUNTIME_DIR", &dir) };
        dir
    })
}

struct FixtureEmbedder;

impl FixtureEmbedder {
    fn registration() -> EmbedderRegistration {
        EmbedderRegistration {
            space: FixtureEmbedder.embedding_space(),
            embedder: Arc::new(FixtureEmbedder),
        }
    }

    fn output() -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "deterministic-768".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![1.0; 768],
        }
    }
}

impl Embedder for FixtureEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search uses prepared inputs")
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
            model_name: "deterministic-768".into(),
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

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(1_536)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        seal.document_builder()
            .token_count(text.len())
            .byte_boundaries((0..text.len()).map(|index| (index, index + 1)).collect())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        seal.query_builder()
            .token_count(text.len())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }

    fn embed_prepared_document_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }

    fn embed_prepared_query_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(Self::output())
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "capture-v1".into(),
        anchor_declaration_version: "anchors-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(key: &str, ordinal: u64, text: &str) -> EventDraft {
    EventDraft {
        session_ref: "read-session".into(),
        source_event_key: key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "captured-author".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(text.into()),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

/// The worker's writable open: a plain Store with no administrative handler.
fn open_worker(path: &Path) -> Store {
    owner_runtime_dir();
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: path.to_path_buf(),
            extra_embedding_spaces: vec![FixtureEmbedder.embedding_space()],
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        vec![FixtureEmbedder::registration()],
    )
    .expect("open the worker's writable Store")
}

fn open_reader(path: &Path) -> ConsumerReader {
    owner_runtime_dir();
    let mut reader = ConsumerReader::open(path, event_history_reader_declaration())
        .expect("the read-only reader opens");
    reader
        .attach_event_search_embedder_registrations(vec![FixtureEmbedder::registration()])
        .expect("attach the query embedder");
    reader
}

fn append(store: &Store, context_id: ContextId, key: &str, ordinal: u64, text: &str) {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "read-stream",
            vec![draft(key, ordinal, text)],
        ))
        .expect("append one event");
}

fn catch_up(store: &Store, context_id: ContextId) {
    store
        .catch_up_event_search(EventSearchBuildRequest {
            context_id,
            embedder: FixtureEmbedder::registration(),
            profile: EventSearchProfile::default(),
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero bytes"),
            lease_duration_ms: NonZeroU64::new(60_000).expect("nonzero lease"),
        })
        .expect("the worker's catch-up completes");
}

/// A Context with one projected event and a second event the worker has yet to project.
fn ready_store_with_pending_event(path: &Path) -> (Store, ContextId) {
    let store = open_worker(path);
    let context_id = store
        .create_context(CreateContext {
            name: "read during projection".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    append(&store, context_id, "first", 1, "incident rollback plan");
    catch_up(&store, context_id);
    append(&store, context_id, "second", 2, "rollback finished");
    (store, context_id)
}

/// Status and find through `reader`, as the CLI's `status` and `find` run them.
fn read_both(reader: &ConsumerReader, context_id: ContextId) -> std::result::Result<usize, String> {
    reader
        .event_history_status(ContextSelector::one(context_id))
        .map_err(|error| format!("status refused: {error:?}"))?;
    let found = reader
        .find_event_history(
            ContextSelector::one(context_id),
            "incident rollback",
            EventSearchFilter::default(),
        )
        .map_err(|error| format!("find refused: {error:?}"))?;
    Ok(found.hits.len())
}

#[test]
fn a_reader_opened_while_the_worker_is_finalizing_answers_status_and_find() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("reader-inside-finalization.db");
    let (worker, context_id) = ready_store_with_pending_event(&path);

    let observed = Rc::new(RefCell::new(None));
    let inside = observed.clone();
    let reader_path = path.clone();
    pause_next_event_search_finalization_for_test(move |_: &Store| {
        let result = ConsumerReader::open(&reader_path, event_history_reader_declaration())
            .map_err(|error| format!("reader open refused: {error:?}"))
            .and_then(|mut reader| {
                reader
                    .attach_event_search_embedder_registrations(vec![
                        FixtureEmbedder::registration(),
                    ])
                    .map_err(|error| format!("attach refused: {error:?}"))?;
                read_both(&reader, context_id)
            });
        *inside.borrow_mut() = Some(result);
    });
    catch_up(&worker, context_id);

    let result = observed
        .borrow_mut()
        .take()
        .expect("the worker reached its finalization window");
    let hits = result.expect("status and find are answered while the worker holds the Store");
    assert!(
        hits > 0,
        "find answers from the ready head during finalization"
    );
}

#[test]
fn a_reader_opened_before_the_worker_takes_the_store_keeps_answering() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("reader-before-worker.db");
    let (worker, context_id) = ready_store_with_pending_event(&path);
    drop(worker);

    let reader = Rc::new(open_reader(&path));
    assert!(
        read_both(&reader, context_id).expect("the reader answers before any worker") > 0,
        "the ready head is searchable"
    );

    let worker = open_worker(&path);
    let observed = Rc::new(RefCell::new(None));
    let inside = observed.clone();
    let held = reader.clone();
    pause_next_event_search_finalization_for_test(move |_: &Store| {
        *inside.borrow_mut() = Some(read_both(&held, context_id));
    });
    catch_up(&worker, context_id);
    observed
        .borrow_mut()
        .take()
        .expect("the worker reached its finalization window")
        .expect("the earlier reader answers while the worker is finalizing");
    drop(worker);

    read_both(&reader, context_id).expect("the earlier reader answers after the worker lets go");
}

const REOPEN_CHILD_ENV: &str = "CG_READ_DURING_PROJECTION_REOPEN_CHILD";
const REOPEN_TEST: &str = "readers_are_never_refused_while_another_process_reopens_the_store";

/// How long the child keeps reopening the Store. Bounded so the reads it overlaps stay a handful
/// and the child's own memory stays small.
const REOPEN_WINDOW: std::time::Duration = std::time::Duration::from_secs(3);

/// How many read attempts the parent makes, each through one reader that is dropped before the
/// next attempt opens another.
const READ_ATTEMPTS: usize = 50;

/// The child half of the cross-process test: open and close a plain writable Store on the named
/// path in a tight loop, as the detached worker does between claim, render/embed and commit, until
/// the parent has finished its reads or the reopen window closes.
fn run_reopening_child(spec: &str) {
    let mut parts = spec.split('\n');
    let (Some(path), Some(started), Some(stop)) = (parts.next(), parts.next(), parts.next()) else {
        panic!("child spec is path, start marker and stop marker");
    };
    let path = Path::new(path);
    let stop = Path::new(stop);
    let cap = std::time::Instant::now() + REOPEN_WINDOW;
    let mut opened = 0_u64;
    while !stop.exists() && std::time::Instant::now() < cap {
        // A writable open may be refused while a reader is hydrating; the worker retries that.
        if let Ok(store) = Store::open_with_embedder_registrations(
            StoreConfig {
                db_path: path.to_path_buf(),
                extra_embedding_spaces: vec![FixtureEmbedder.embedding_space()],
                default_text_embedder: None,
                ..StoreConfig::default()
            },
            vec![FixtureEmbedder::registration()],
        ) {
            opened += 1;
            if opened == 1 {
                std::fs::write(started, b"started").expect("write the start marker");
            }
            drop(store);
        }
    }
    println!("child opened the writable Store {opened} times");
}

#[test]
fn readers_are_never_refused_while_another_process_reopens_the_store() {
    // The child inherits the parent's owner runtime directory, as the detached worker shares the
    // CLI's: its owner must publish its channel exactly where the parent's reader looks.
    if let Ok(spec) = std::env::var(REOPEN_CHILD_ENV) {
        run_reopening_child(&spec);
        return;
    }
    owner_runtime_dir();
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("reader-across-processes.db");
    let (worker, context_id) = ready_store_with_pending_event(&path);
    drop(worker);
    let started = root.path().join("child-started");
    let stop = root.path().join("parent-finished");

    let child = process::Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", REOPEN_TEST, "--nocapture", "--test-threads", "1"])
        .env(
            REOPEN_CHILD_ENV,
            format!(
                "{}\n{}\n{}",
                path.display(),
                started.display(),
                stop.display()
            ),
        )
        .stdout(process::Stdio::piped())
        .stderr(process::Stdio::piped())
        .spawn()
        .expect("spawn the reopening child");
    while !started.exists() {
        std::thread::yield_now();
    }

    let mut refusals = Vec::new();
    for attempt in 0..READ_ATTEMPTS {
        let answer = ConsumerReader::open(&path, event_history_reader_declaration())
            .map_err(|error| format!("reader open refused: {error:?}"))
            .and_then(|mut reader| {
                reader
                    .attach_event_search_embedder_registrations(vec![
                        FixtureEmbedder::registration(),
                    ])
                    .map_err(|error| format!("attach refused: {error:?}"))?;
                // The reader is dropped at the end of this attempt, before the next one opens.
                read_both(&reader, context_id)
            });
        if let Err(refusal) = answer {
            refusals.push(format!("attempt {attempt}: {refusal}"));
        }
    }
    std::fs::write(&stop, b"finished").expect("write the stop marker");
    let output = child.wait_with_output().expect("the child finishes");
    let child_report = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let child_opens = child_report
        .lines()
        .find_map(|line| {
            line.split("child opened the writable Store ")
                .nth(1)
                .and_then(|rest| rest.strip_suffix(" times"))
                .and_then(|count| count.parse::<u64>().ok())
        })
        .unwrap_or(0);
    assert!(
        output.status.success(),
        "the child ran cleanly: {child_report}"
    );
    assert!(
        refusals.is_empty(),
        "{} of {READ_ATTEMPTS} reads were refused while another process reopened the Store \
         {child_opens} times: {:#?}",
        refusals.len(),
        refusals
    );
    assert!(
        child_opens >= 5,
        "the child kept reopening the Store throughout the reads: {child_opens} opens; {child_report}"
    );
}
