//! Reads stay available while a projection worker holds the Store.
//!
//! The worker here is a plain writable Store, exactly as the detached projection worker and an
//! import open it: no administrative handler. The owner channel lives in an explicit runtime
//! directory, as the CLI places it. Reads run from inside the worker's final head-switch
//! transaction through the internal finalization pause.

#![cfg(context_graph_internal_test_support)]

use std::cell::RefCell;
use std::collections::HashMap;
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
use contextdb_core::Value;
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
            embedding_workers: NonZeroUsize::MIN,
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

#[test]
fn active_reader_snapshot_remains_consistent_across_currency_maintenance() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("reader-across-currency-maintenance.db");
    let (worker, context_id) = ready_store_with_pending_event(&path);
    let reader = open_reader(&path);
    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    let sql = "SELECT id, is_searchable FROM event_search_passages WHERE context_id = $context_id ORDER BY passage_ordinal";
    let mut cursor = reader
        .open_cursor(sql, &params)
        .expect("open reader statement before passage update");
    let first = cursor.first_page().expect("first snapshot page").clone();
    assert_eq!(first.rows.len(), 1, "one initial projected passage");
    assert_eq!(first.rows[0][1], Value::Bool(true));
    let passage_id = first.rows[0][0].clone();

    // A ConsumerReader alone does not pin one snapshot across separate statements. Keep the
    // statement cursor open and pin its pre-update engine snapshot for a second explicit read.
    let db = worker.sync_database();
    let snapshot = db.snapshot();
    let pin = db.pin_snapshot(snapshot);
    db.execute_at_snapshot(sql, &params, snapshot)
        .expect("read old snapshot before update");
    db.execute(
        "UPDATE event_search_passages SET is_searchable = false WHERE id = $id",
        &HashMap::from([("id".into(), passage_id.clone())]),
    )
    .expect("supersede a passage version while the reader is open");
    let held = db
        .run_maintenance_cycle()
        .expect("maintain while the old reader snapshot is pinned");
    eprintln!("reader-held currency maintenance: {:?}", held.currency);
    assert!(
        held.currency.versions_deferred_for_readers > 0,
        "the old snapshot's passage version is protected: {held:?}"
    );
    assert_eq!(cursor.first_page().expect("held reader page"), &first);
    let old = db
        .execute_at_snapshot(sql, &params, snapshot)
        .expect("old snapshot remains readable after maintenance");
    assert_eq!(old.rows[0][1], Value::Bool(true));
    let current = reader.execute(sql, &params).expect("read current passage");
    assert_eq!(current.rows[0][1], Value::Bool(false));

    cursor.close().expect("close held cursor");
    drop(pin);
    let released = db
        .run_maintenance_cycle()
        .expect("reclaim after the old reader snapshot releases");
    eprintln!(
        "reader-released currency maintenance: {:?}",
        released.currency
    );
    assert!(released.currency.pruned_versions > 0, "{released:?}");
    assert!(
        released
            .currency
            .compacted_tables
            .iter()
            .any(|table| table == "event_search_passages"),
        "the protected passage version was reclaimed: {released:?}"
    );
    let current = reader.execute(sql, &params).expect("read after reclaim");
    assert_eq!(current.rows[0][1], Value::Bool(false));
}

#[test]
fn consumer_cursor_fetches_unread_passage_versions_across_currency_maintenance() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("cursor-across-currency-maintenance.db");
    let worker = open_worker(&path);
    let context_id = worker
        .create_context(CreateContext {
            name: "paged passage reader".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    let drafts = (0..220)
        .map(|ordinal| {
            draft(
                &format!("passage-{ordinal}"),
                ordinal + 1,
                &format!("passage {ordinal}: {}", "x".repeat(230)),
            )
        })
        .collect();
    worker
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "paged-stream",
            drafts,
        ))
        .expect("append enough events for several cursor pages");
    catch_up(&worker, context_id);

    let db = worker.sync_database();
    // Projection may leave older versions of unrelated rows. Clear them before opening the
    // cursor so the deferred count below belongs to the passage this proof supersedes.
    db.run_maintenance_cycle()
        .expect("clear projection versions before the cursor opens");
    let reader = open_reader(&path);
    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    let sql = "SELECT id, is_searchable FROM event_search_passages WHERE context_id = $context_id ORDER BY passage_ordinal";
    let before = reader
        .execute(sql, &params)
        .expect("read passage identities before update");
    assert!(
        before.rows.len() > 200,
        "need several cursor pages: {before:?}"
    );
    let target = before.rows[150][0].clone();

    let mut cursor = reader
        .open_cursor(sql, &params)
        .expect("open consumer cursor before passage update");
    let first = cursor.first_page().expect("first cursor page").clone();
    assert!(first.has_more, "unread passage pages must remain");
    assert!(
        first.rows.len() <= 100,
        "the changed row must still be unread"
    );
    assert!(first.rows.iter().all(|row| row[0] != target));

    db.execute(
        "UPDATE event_search_passages SET is_searchable = false WHERE id = $id",
        &HashMap::from([("id".into(), target.clone())]),
    )
    .expect("supersede a passage the cursor has not fetched");
    worker
        .maintain_event_search_vectors_for_test()
        .expect("run passage vector maintenance with cursor open");
    let held = db
        .run_maintenance_cycle()
        .expect("maintain while only the consumer cursor holds the old read");
    assert!(
        held.currency.versions_deferred_for_readers > 0,
        "the cursor must defer reclaim of its unread passage version: {held:?}"
    );

    let mut pages = 1;
    let mut seen = first.rows;
    let mut has_more = first.has_more;
    while has_more {
        let page = cursor
            .fetch(None)
            .expect("continue the same cursor without retry");
        pages += 1;
        has_more = page.has_more;
        seen.extend(page.rows);
    }
    assert!(pages >= 3, "the cursor must fetch several pages");
    assert_eq!(seen.len(), before.rows.len());
    assert_eq!(
        seen, before.rows,
        "the cursor must return its pre-update snapshot"
    );
    assert_eq!(
        seen.iter()
            .find(|row| row[0] == target)
            .expect("unread target")[1],
        Value::Bool(true),
    );

    cursor.close().expect("release the consumer cursor");
    let released = db
        .run_maintenance_cycle()
        .expect("reclaim after the consumer cursor closes");
    assert!(released.currency.pruned_versions > 0, "{released:?}");
    assert!(released.currency.reclaimed_bytes > 0, "{released:?}");
    assert!(
        released
            .currency
            .compacted_tables
            .iter()
            .any(|table| table == "event_search_passages"),
        "the passage version must be reclaimed: {released:?}"
    );
    let current = reader
        .execute(sql, &params)
        .expect("fresh read after reclaim");
    assert_eq!(
        current
            .rows
            .iter()
            .find(|row| row[0] == target)
            .expect("current target")[1],
        Value::Bool(false),
    );
}

const REOPEN_CHILD_ENV: &str = "CG_READ_DURING_PROJECTION_REOPEN_CHILD";
const REOPEN_TEST: &str = "readers_are_never_refused_while_another_process_reopens_the_store";

/// How many read attempts the parent makes, each through one reader that is dropped before the
/// next attempt opens another.
const READ_ATTEMPTS: usize = 50;

/// The child half of the cross-process test: open and close a plain writable Store on the named
/// path in a tight loop, as the detached worker does between claim, render/embed and commit, until
/// the parent explicitly reports completion of its reads.
fn run_reopening_child(spec: &str) {
    use std::io::{Read, Write};
    let (path, socket) = spec
        .split_once('\n')
        .expect("child spec is store path and control socket");
    let mut control = std::os::unix::net::UnixStream::connect(socket).expect("parent control");
    control
        .set_nonblocking(true)
        .expect("nonblocking stop check between opens");
    let mut opened = 0_u64;
    loop {
        let mut stop = [0];
        match control.read(&mut stop) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("parent control: {error}"),
        }
        if let Ok(store) = Store::open_with_embedder_registrations(
            StoreConfig {
                db_path: Path::new(path).to_path_buf(),
                extra_embedding_spaces: vec![FixtureEmbedder.embedding_space()],
                default_text_embedder: None,
                ..StoreConfig::default()
            },
            vec![FixtureEmbedder::registration()],
        ) {
            opened += 1;
            drop(store);
            if opened == 5 {
                control
                    .write_all(&[1])
                    .expect("announce five completed reopens");
            }
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
    use std::io::{Read, Write};
    let socket_dir = tempfile::tempdir_in("/tmp").expect("short control socket directory");
    let socket = socket_dir.path().join("control.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).expect("control listener");

    let child = process::Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", REOPEN_TEST, "--nocapture", "--test-threads", "1"])
        .env(
            REOPEN_CHILD_ENV,
            format!("{}\n{}", path.display(), socket.display()),
        )
        .stdout(process::Stdio::piped())
        .stderr(process::Stdio::piped())
        .spawn()
        .expect("spawn the reopening child");
    let (mut control, _) = listener.accept().expect("reopening child connected");
    let mut ready = [0];
    control
        .read_exact(&mut ready)
        .expect("child completed five reopens");
    assert_eq!(ready, [1]);

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
    control
        .write_all(&[1])
        .expect("stop the reopening child after all reads");
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
