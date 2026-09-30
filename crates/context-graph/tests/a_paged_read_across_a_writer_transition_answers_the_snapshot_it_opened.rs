//! A read that started before a writer handed the store over finishes the read it started.
//!
//! ## The promise
//!
//! A downstream runtime pages through its memory while the machine around it changes: the process
//! that was writing exits, another opens and commits. The read is owed one answer, and it is the
//! answer the read OPENED with — every row that was committed when the cursor was taken, each of
//! them exactly once, and nothing that arrived afterwards. A page that silently picked up newer
//! rows would hand a consumer a list with duplicates and gaps and no way to notice; a read that
//! never returns would hang the runtime behind it.
//!
//! ## The two paths, because they are different code
//!
//! - A CURSOR's later pages (`ConsumerCursor::fetch`, `consumer_read.rs`) go straight to the
//!   engine cursor and propagate what it answers. The cursor's snapshot is the thing at stake.
//! - A one-shot STATEMENT (`ConsumerReader::execute` through `on_session`, `consumer_read.rs`)
//!   answers an owner-in-transition condition by selecting the route again and asking the same
//!   question there. Nothing was answered yet when it does that, so what a caller is owed is an
//!   answer, and this file pins which one arrives.
//!
//! Both are driven across a REAL transition: a writer process holds the store, exits, and a second
//! writer process opens and commits rows that would change the answer. Nothing sleeps: the harness
//! advances only when a child has printed that it has done its part.

#![cfg(unix)]

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use context_graph::{
    AppendEventBatch, ConsumerDeclaration, ConsumerSchema, ConsumerTable, ContextId, CreateContext,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, Store, StoreConfig,
    schema::TABLES,
};
use contextdb_core::Value;
use serde_json::json;

/// Which role a re-invocation of this test binary is playing. Absent in the parent.
const CHILD_ROLE_ENV: &str = "CG_WRITER_TRANSITION_CHILD_ROLE";
/// The store every child works on.
const CHILD_STORE_ENV: &str = "CG_WRITER_TRANSITION_STORE_PATH";
/// The Context every child appends into.
const CHILD_CONTEXT_ENV: &str = "CG_WRITER_TRANSITION_CONTEXT_ID";

const HOLDING: &str = "CG_WRITER_TRANSITION_HOLDING";
const RELEASED: &str = "CG_WRITER_TRANSITION_RELEASED";
const COMMITTED: &str = "CG_WRITER_TRANSITION_COMMITTED";
const RELEASE: &str = "release";

const CONNECTOR: &str = "writer-transition";
const SOURCE: &str = "writer-transition-fixture";
/// Enough rows that the read the proof opens cannot be answered in one page under the shipped
/// row ceiling.
const SEEDED_EVENTS: usize = 900;
/// What the second writer commits after the first has gone: rows that would change the answer.
const ARRIVING_EVENTS: usize = 120;

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn adapter() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-v1".into(),
        anchor_declaration_version: "fixture-v1".into(),
        anchor_schema: json!({"type":"object", "additionalProperties":true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

/// `count` captured events under `session`, appended in batches so the fixture is a handful of
/// commits rather than one per row.
fn append_events(store: &Store, context_id: ContextId, session: &str, first: usize, count: usize) {
    for batch_start in (0..count).step_by(150) {
        let batch_end = (batch_start + 150).min(count);
        let drafts = (batch_start..batch_end)
            .map(|index| {
                let ordinal = first + index;
                EventDraft {
                    session_ref: session.to_string(),
                    source_event_key: format!("{session}-{ordinal}"),
                    source_position: json!({ "ordinal": ordinal }),
                    occurred_at: None,
                    author: "human".into(),
                    kind: "message".into(),
                    render_role: EventRenderRole::HumanMessage,
                    interaction_ref: None,
                    text: Some(format!("row {ordinal}")),
                    payload: json!({}),
                    anchors: json!({}),
                    source_remainder: json!({}),
                    coverage: EventCoverage::Complete,
                }
            })
            .collect::<Vec<_>>();
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                adapter(),
                SOURCE,
                drafts,
            ))
            .expect("append captured events");
    }
}

fn child_store_path() -> PathBuf {
    PathBuf::from(std::env::var_os(CHILD_STORE_ENV).expect("a child is given the store"))
}

fn child_context_id() -> ContextId {
    ContextId::from(
        uuid::Uuid::parse_str(
            &std::env::var(CHILD_CONTEXT_ENV).expect("a child is given the Context"),
        )
        .expect("a child's Context id is a uuid"),
    )
}

/// The writer that holds the store the way a running import holds it, until it is released.
#[test]
fn a_writer_holder_child() {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("writer-holder") {
        return;
    }
    let store = Store::open(store_config(child_store_path())).expect("the writer child takes it");
    println!("{HOLDING}");
    std::io::stdout().flush().expect("flush the holding marker");
    let mut command = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut command)
        .expect("read the release command");
    assert_eq!(command.trim(), RELEASE);
    drop(store);
    println!("{RELEASED}");
    std::io::stdout()
        .flush()
        .expect("flush the released marker");
}

/// The writer that opens after the first has gone and commits rows that would change the answer.
#[test]
fn an_arriving_writer_child() {
    if std::env::var(CHILD_ROLE_ENV).ok().as_deref() != Some("arriving-writer") {
        return;
    }
    let store = Store::open(store_config(child_store_path())).expect("the second writer opens");
    append_events(
        &store,
        child_context_id(),
        "arrived-after-the-transition",
        SEEDED_EVENTS,
        ARRIVING_EVENTS,
    );
    drop(store);
    println!("{COMMITTED}");
    std::io::stdout()
        .flush()
        .expect("flush the committed marker");
}

struct WriterChild {
    child: Child,
    stdout: BufReader<ChildStdout>,
    finished: bool,
}

impl WriterChild {
    fn spawn(role: &str, test_name: &str, store_path: &Path, context_id: ContextId) -> Self {
        let mut child =
            Command::new(std::env::current_exe().expect("the current integration-test binary"))
                .arg("--exact")
                .arg(test_name)
                .arg("--nocapture")
                .env(CHILD_ROLE_ENV, role)
                .env(CHILD_STORE_ENV, store_path)
                .env(CHILD_CONTEXT_ENV, context_id.as_uuid().to_string())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap_or_else(|error| panic!("start the {role} child: {error}"));
        let stdout = child.stdout.take().expect("the child's stdout");
        Self {
            child,
            stdout: BufReader::new(stdout),
            finished: false,
        }
    }

    /// Advance only when the child has printed `marker`, never on elapsed time.
    fn wait_for(&mut self, marker: &str) {
        let mut line = String::new();
        while self
            .stdout
            .read_line(&mut line)
            .expect("read the child's markers")
            != 0
        {
            if line.contains(marker) {
                return;
            }
            line.clear();
        }
        panic!("the child exited before it reached `{marker}`");
    }

    fn release_and_wait(mut self) {
        let stdin = self.child.stdin.as_mut().expect("the child's release pipe");
        writeln!(stdin, "{RELEASE}").expect("release the holding child");
        stdin.flush().expect("flush the release");
        self.wait_for(RELEASED);
        let status = self.child.wait().expect("wait for the holding child");
        self.finished = true;
        assert!(
            status.success(),
            "a released writer exits cleanly: {status}"
        );
    }

    fn wait_until_done(mut self) {
        self.wait_for(COMMITTED);
        let status = self.child.wait().expect("wait for the arriving writer");
        self.finished = true;
        assert!(
            status.success(),
            "the arriving writer exits cleanly: {status}"
        );
    }
}

impl Drop for WriterChild {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn declaration() -> ConsumerDeclaration {
    let tables = ["contexts", "events"]
        .iter()
        .map(|name| {
            let spec = TABLES
                .iter()
                .find(|spec| spec.name == *name)
                .unwrap_or_else(|| panic!("context-graph publishes a table spec named {name}"));
            ConsumerTable {
                name: spec.name.to_string(),
                ddl: spec.ddl.to_string(),
            }
        })
        .collect();
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "writer_transition_reader".to_string(),
            tables,
        },
        scope_labels: BTreeSet::new(),
    }
}

struct Fixture {
    _root: tempfile::TempDir,
    db_path: PathBuf,
    context_id: ContextId,
}

fn fixture() -> Fixture {
    let root = tempfile::tempdir().expect("tempdir");
    let db_path = root.path().join("writer-transition.db");
    let store = Store::open(store_config(db_path.clone())).expect("open the fixture store");
    let context_id = store
        .create_context(CreateContext {
            name: "writer-transition".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    append_events(&store, context_id, "seeded", 0, SEEDED_EVENTS);
    drop(store);
    Fixture {
        _root: root,
        db_path,
        context_id,
    }
}

/// The statement both proofs read: every seeded event, in a stable order, from a table the reader
/// declared.
const PAGED_READ: &str = "SELECT id, seq FROM events WHERE context_id = $context_id ORDER BY seq";

fn params(context_id: ContextId) -> HashMap<String, Value> {
    HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))])
}

fn seq_of(row: &[Value], columns: &[String]) -> u64 {
    let index = columns
        .iter()
        .position(|column| column == "seq")
        .expect("the read names seq");
    match &row[index] {
        Value::Int64(value) => u64::try_from(*value).expect("a sequence number is not negative"),
        other => panic!("seq came back as {other:?}"),
    }
}

#[test]
fn a_cursor_opened_before_the_handover_finishes_the_read_it_opened() {
    let fixture = fixture();

    let mut holder = WriterChild::spawn(
        "writer-holder",
        "a_writer_holder_child",
        &fixture.db_path,
        fixture.context_id,
    );
    holder.wait_for(HOLDING);

    let reader = context_graph::ConsumerReader::open(&fixture.db_path, declaration())
        .expect("a read-only consumer opens beside the writer that is holding the store");
    let mut cursor = reader
        .open_cursor(PAGED_READ, &params(fixture.context_id))
        .expect("open a cursor on the paged read");
    let first = cursor.first_page().expect("the first page").clone();
    assert!(
        first.has_more,
        "this proof needs a read that cannot be answered in one page; the first page carried {} \
         rows and reported no more",
        first.rows.len()
    );
    let mut seen = Vec::new();
    for row in &first.rows {
        seen.push(seq_of(row, &first.columns));
    }

    // The writer that was holding the store goes, and a different process opens and commits rows
    // that would change this read's answer.
    holder.release_and_wait();
    WriterChild::spawn(
        "arriving-writer",
        "an_arriving_writer_child",
        &fixture.db_path,
        fixture.context_id,
    )
    .wait_until_done();

    // The rest of the read the cursor opened.
    let mut fetches = Vec::new();
    let mut has_more = first.has_more;
    while has_more {
        match cursor.fetch(None) {
            Ok(page) => {
                has_more = page.has_more;
                for row in &page.rows {
                    seen.push(seq_of(row, &page.columns));
                }
                fetches.push(Ok(page.rows.len()));
            }
            Err(error) => {
                fetches.push(Err(format!("{error:?}")));
                break;
            }
        }
    }
    let _ = cursor.close();

    let refusals = fetches
        .iter()
        .filter_map(|fetch| fetch.as_ref().err())
        .collect::<Vec<_>>();
    assert!(
        refusals.is_empty(),
        "a cursor opened before the handover must finish the read it opened; it refused instead: \
         {refusals:#?}. Pages read before the refusal: {}, rows seen: {}",
        fetches.len(),
        seen.len()
    );

    let unique = seen.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(
        unique.len(),
        seen.len(),
        "the read answered some row twice: {} rows carrying {} distinct sequence numbers",
        seen.len(),
        unique.len()
    );
    assert_eq!(
        seen.len(),
        SEEDED_EVENTS,
        "the read answers exactly the rows its snapshot held: neither an omission nor one of the \
         {ARRIVING_EVENTS} rows the second writer committed after the cursor was opened"
    );
    assert_eq!(
        unique.iter().copied().max(),
        Some(SEEDED_EVENTS as u64),
        "the highest sequence number answered is the last row the snapshot held"
    );
}

#[test]
fn a_statement_crossing_the_handover_answers_the_route_that_took_over() {
    let fixture = fixture();

    let mut holder = WriterChild::spawn(
        "writer-holder",
        "a_writer_holder_child",
        &fixture.db_path,
        fixture.context_id,
    );
    holder.wait_for(HOLDING);

    let reader = context_graph::ConsumerReader::open(&fixture.db_path, declaration())
        .expect("a read-only consumer opens beside the writer that is holding the store");
    // The session this reader took was served by the writer that is now going away; a second
    // writer then opens and commits rows that change what the statement below would answer.
    holder.release_and_wait();
    WriterChild::spawn(
        "arriving-writer",
        "an_arriving_writer_child",
        &fixture.db_path,
        fixture.context_id,
    )
    .wait_until_done();

    let answered = reader
        .execute(
            "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
            &params(fixture.context_id),
        )
        .expect("a statement crossing a writer handover is answered, never left hanging");
    let row = answered.rows.first().expect("one aggregate row");
    let index = answered
        .columns
        .iter()
        .position(|column| column == "n")
        .expect("the count names n");
    let counted = match &row[index] {
        Value::Int64(value) => u64::try_from(*value).expect("a count is not negative"),
        other => panic!("the count came back as {other:?}"),
    };
    // This is the contract, stated so nobody has to guess it: a one-shot statement whose route
    // says only that the owner is gone is asked again on the route that took over, so it answers
    // from THAT snapshot -- the rows the second writer committed included. Nothing had been
    // answered when the route changed, so no answer is being revised; but two statements on one
    // reader can land on two snapshots, and a consumer that needs several reads of ONE snapshot
    // has to hold a cursor rather than repeat `execute`.
    assert_eq!(
        counted,
        (SEEDED_EVENTS + ARRIVING_EVENTS) as u64,
        "a statement re-asked across a writer handover answers the route that took over, whole: \
         it should see the {SEEDED_EVENTS} rows the store already held plus the \
         {ARRIVING_EVENTS} the second writer committed"
    );
}
