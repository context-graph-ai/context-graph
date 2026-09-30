#![cfg(unix)]
//! What a reader is served is decided by the table that is REALLY in the store, never by the
//! CREATE TABLE text the caller handed in.
//!
//! ## The promise
//!
//! A consumer that opens the read-only surface as the `edge` role is entitled to the edge rows and
//! to nothing else. The declaration it supplies says which tables it owns and which labels it may
//! see; the STORE says what those tables actually are. When the two disagree, the open is refused
//! by type, naming the table — never opened on the caller's word and then served rows the caller
//! was never entitled to.
//!
//! ## Why the disagreement is dangerous rather than untidy
//!
//! Read narrowing is the engine's own read-scoping vocabulary: only a split `SCOPE_LABEL_READ`
//! set narrows what a reader is served. A simple `SCOPE_LABEL ('edge', 'server')` column is
//! WRITE-scoping and narrows nothing at all. So a declaration that describes a real, persisted
//! simple-scoped table as split — or as carrying no scope column at all — and supplies the `edge`
//! label is asking for a narrowing that cannot happen. Accepted, the reader is handed a session
//! that applies no read gate whatever, and the very first `SELECT` over that table returns the
//! hub's rows alongside the edge's. That is silent widening: nothing fails, nothing is logged, and
//! a consumer that asked to see one role's rows is quietly shown both.
//!
//! The declaration need not be malicious to do this. A caller can supply a DDL constant for the
//! wrong table or copy a declaration from a sibling table. Either way, the declaration fails to
//! describe the persisted table it names.
//!
//! ## What the writable door already does
//!
//! The consumer's WRITABLE doors refuse exactly this drift: a declaration that no longer describes
//! the persisted table comes back `CgError::ConsumerTableDrifted`, naming the table and the
//! differences (`store::consumer_tables::verify_existing_table`). The reading door is the one
//! consumer surface that does not check, so the same forged declaration that is refused for
//! writing is accepted for reading.
//!
//! ## What is proved here
//!
//! Both forged shapes are refused, on BOTH read routes, before a single row can be served; the
//! truthful declaration over the same store still opens and answers, on both routes. The refusal
//! is a typed consumer-declaration refusal naming the table, so a consumer can act on it — either
//! the drift refusal the writable door gives, or the read-narrowing refusal the reading surface
//! already gives for a declaration it cannot honour. Which of the two is the surface's own choice;
//! that it is one of them, names the table, and arrives INSTEAD of an answer is the contract.
//!
//! Every assertion below is about a typed value or a row. Nothing waits on elapsed time and
//! nothing sleeps.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use context_graph::owner_control::{ControlHandler, request_owner};
use context_graph::{
    CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, ScopeLabel, Store,
    StoreConfig,
};
use contextdb_core::Value;

/// The table as it is REALLY created in the store: a simple, WRITE-scoping `SCOPE_LABEL` column.
/// The engine's row gate reads only split READ sets, so this column narrows nothing a reader is
/// served.
const PERSISTED_SIGNALS_DDL: &str = "CREATE TABLE vigil_signals (id TEXT PRIMARY KEY, body TEXT, scope_label TEXT SCOPE_LABEL ('edge', 'server'))";
/// The first forgery: the same table described as read-scoped, which is the shape a declaration
/// claiming the `edge` label would need in order to be honoured. The store does not have it.
const FORGED_SPLIT_SIGNALS_DDL: &str = "CREATE TABLE vigil_signals (id TEXT PRIMARY KEY, body TEXT, scope_label TEXT SCOPE_LABEL_READ ('edge') WRITE ('server'))";
/// The second forgery: the same table described with no scope column at all. A declaration this
/// shape carries nothing for the reading surface to object to, so it is the quietest way to arrive
/// at an unnarrowed session while claiming a role.
const FORGED_UNSCOPED_SIGNALS_DDL: &str =
    "CREATE TABLE vigil_signals (id TEXT PRIMARY KEY, body TEXT, scope_label TEXT)";
/// A genuinely read-scoped table, created exactly as declared: the positive control's other half,
/// where claiming the `edge` label IS honourable and the engine really does narrow.
const EVENTS_DDL: &str = "CREATE TABLE vigil_events (id TEXT PRIMARY KEY, body TEXT, scope_label TEXT SCOPE_LABEL_READ ('edge', 'server') WRITE ('edge', 'server'))";

/// The two rows every fixture store holds: one for each role, so an unnarrowed answer is visibly
/// different from a narrowed one.
const SEEDED_ROWS: [(&str, &str, &str); 2] = [
    ("edge-1", "raised at the edge", "edge"),
    ("server-1", "raised at the hub", "server"),
];

fn no_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// A store open that asks for no embedder: this file is about declarations and rows, and a model
/// download would be a second thing to go wrong.
fn store_config(db_path: &Path) -> StoreConfig {
    StoreConfig {
        db_path: db_path.to_path_buf(),
        default_text_embedder: None,
        ..Default::default()
    }
}

fn one_table_schema(name: &str, ddl: &str) -> ConsumerSchema {
    ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: name.to_string(),
            ddl: ddl.to_string(),
        }],
    }
}

/// A declaration over one table, opened as the roles named in `labels`.
fn declaration(name: &str, ddl: &str, labels: &[&str]) -> ConsumerDeclaration {
    ConsumerDeclaration {
        schema: one_table_schema(name, ddl),
        scope_labels: labels.iter().map(|label| ScopeLabel::new(*label)).collect(),
    }
}

fn seed(store: &Store, table: &str) {
    for (id, body, scope) in SEEDED_ROWS {
        store
            .sync_database()
            .execute(
                &format!("INSERT INTO {table} (id, body, scope_label) VALUES ($id, $body, $scope)"),
                &HashMap::from([
                    ("id".to_string(), Value::Text(id.to_string())),
                    ("body".to_string(), Value::Text(body.to_string())),
                    ("scope".to_string(), Value::Text(scope.to_string())),
                ]),
            )
            .expect("an unrestricted handle owns every row its declaration accepts");
    }
}

/// A real store holding the table exactly as `ddl` creates it, with one row for each role — then
/// closed and let go, so a reader takes the committed file rather than an owner.
fn seeded_store(directory: &Path, file: &str, table: &str, ddl: &str) -> PathBuf {
    let path = directory.join(file);
    let store = Store::open_with_consumer_schema(
        store_config(&path),
        Vec::new(),
        one_table_schema(table, ddl),
    )
    .expect("the consumer's first run creates the store and the table it declared");
    seed(&store, table);
    drop(store);
    path
}

/// The store with the write-scoped table every forgery below lies about.
fn persisted_simple_scoped_store(directory: &Path) -> PathBuf {
    seeded_store(
        directory,
        "write-scoped.db",
        "vigil_signals",
        PERSISTED_SIGNALS_DDL,
    )
}

/// The store whose table really is read-scoped, where the truthful narrowing declaration works.
fn persisted_read_scoped_store(directory: &Path) -> PathBuf {
    seeded_store(directory, "read-scoped.db", "vigil_events", EVENTS_DDL)
}

/// A live owner holding the store and answering on its channel, so a reader opened while this is
/// alive reaches the store over the OWNER route rather than by reading the committed file.
///
/// The ping is not decoration: it establishes that the channel is up and answering before any
/// reader dials it, which is what makes "this reader took the owner route" a fact rather than an
/// assumption about route selection.
fn live_owner(path: &Path) -> Store {
    let handler: ControlHandler = Arc::new(|frame: String| format!("handled:{}", frame.trim()));
    let store = Store::open_with_control_handler(store_config(path), handler)
        .expect("a writable open becomes the owner serving this store");
    let answer = request_owner(path, "ping\n")
        .expect("the owner answers on its channel, so a reader opened now takes the owner route");
    assert_eq!(
        answer, "handled:ping",
        "the owner really is the process serving this store's channel",
    );
    store
}

fn text_column(rows: &[Vec<Value>]) -> Vec<String> {
    rows.iter()
        .map(|row| match &row[0] {
            Value::Text(text) => text.clone(),
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect()
}

fn ids_in(reader: &ConsumerReader, table: &str) -> Vec<String> {
    let answered = reader
        .execute(&format!("SELECT id FROM {table} ORDER BY id"), &no_params())
        .expect("the declared table answers the reader that declared it");
    text_column(&answered.rows)
}

/// What an open on a forged declaration actually did — refused, or opened and served rows.
///
/// The defect is not "an open that should have failed": it is a reader being served rows it
/// declared no entitlement to. So a leak is RECORDED with the rows it produced rather than thrown
/// away, which is what lets one comparison say both that the two read routes agree and what they
/// agreed on.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// The open was refused by a typed consumer-declaration refusal, naming this table.
    Refused {
        variant: &'static str,
        table: String,
    },
    /// The open answered, and the edge-role reader was served these rows.
    Served(Vec<String>),
}

/// Open a store on a forged declaration and say what happened.
///
/// Two typed refusals are legitimate answers to a declaration that does not describe the persisted
/// table: the drift refusal the writable door already gives, and the reading surface's own refusal
/// of a narrowing it cannot honour. Which one is the surface's choice. Anything else — an untyped
/// failure, or an answer — is not.
fn open_forgery(path: &Path, forged: ConsumerDeclaration, context: &str) -> Outcome {
    match ConsumerReader::open(path, forged) {
        Ok(reader) => {
            let served = ids_in(&reader, "vigil_signals");
            let _ = reader.close();
            Outcome::Served(served)
        }
        Err(CgError::ConsumerTableDrifted { table, .. }) => Outcome::Refused {
            variant: "ConsumerTableDrifted",
            table,
        },
        Err(CgError::ConsumerTableRefused { subject, .. }) => Outcome::Refused {
            variant: "ConsumerTableRefused",
            table: subject,
        },
        Err(other) => panic!(
            "{context}: a declaration that does not describe the persisted table must be refused \
             by a typed consumer-declaration refusal naming the table, got: {other:?}"
        ),
    }
}

/// The one thing every forgery below must produce: a typed refusal naming the forged table,
/// instead of an answer.
fn assert_refused_naming_the_table(outcome: &Outcome, route: &str, shape: &str) {
    match outcome {
        Outcome::Refused { table, .. } => assert_eq!(
            "vigil_signals", table,
            "{route}, {shape}: the refusal must name the table whose declaration does not \
             describe it",
        ),
        Outcome::Served(served) => panic!(
            "{route}, {shape}: the forged declaration opened instead of being refused, and the \
             edge-role reader was served {served:?}. The store's real column is the write-scoping \
             SCOPE_LABEL form, so no read gate was applied and the hub's row reached a reader that \
             declared only the edge role."
        ),
    }
}

/// Every forged shape this file drives, with the words that name it in a failure.
fn forged_shapes() -> [(&'static str, ConsumerDeclaration); 2] {
    [
        (
            "a persisted write-scoped table forged as read-scoped",
            declaration("vigil_signals", FORGED_SPLIT_SIGNALS_DDL, &["edge"]),
        ),
        (
            "a persisted write-scoped table forged as carrying no scope column",
            declaration("vigil_signals", FORGED_UNSCOPED_SIGNALS_DDL, &["edge"]),
        ),
    ]
}

// The headline forgery. The store's `vigil_signals` really carries the write-scoping simple
// `SCOPE_LABEL` column; the declaration describes it as `SCOPE_LABEL_READ ('edge') WRITE
// ('server')` and opens as `edge`. Read the declaration alone and it is impeccable — it asks for a
// narrowing over what it claims is a read-scoped column. Read the STORE and the narrowing is
// impossible, so the open must be refused before the reader can ask for anything.
#[test]
fn a_write_scoped_table_forged_as_read_scoped_is_refused_before_any_row_is_served() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let store = persisted_simple_scoped_store(directory.path());

    let shape = "a persisted write-scoped table forged as read-scoped";
    let outcome = open_forgery(
        &store,
        declaration("vigil_signals", FORGED_SPLIT_SIGNALS_DDL, &["edge"]),
        shape,
    );

    assert_refused_naming_the_table(&outcome, "the direct route", shape);
}

// The quiet forgery, and the one a stale deployment reaches by accident: the declaration carries no
// scope column at all, so there is nothing in the text for a surface reading only the text to
// object to — while the `edge` label it opens with promises a narrowing the persisted column
// cannot deliver.
#[test]
fn a_write_scoped_table_forged_as_unscoped_is_refused_before_any_row_is_served() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let store = persisted_simple_scoped_store(directory.path());

    let shape = "a persisted write-scoped table forged as carrying no scope column";
    let outcome = open_forgery(
        &store,
        declaration("vigil_signals", FORGED_UNSCOPED_SIGNALS_DDL, &["edge"]),
        shape,
    );

    assert_refused_naming_the_table(&outcome, "the direct route", shape);
}

// The positive control, and the reason this is a verification rather than a ban. A declaration that
// DOES describe the persisted table opens and answers, in both of the shapes a consumer brings: the
// write-scoped table declared truthfully with no labels is served unnarrowed, exactly as it always
// has been, and the read-scoped table declared truthfully as `edge` is narrowed by the engine to
// the edge row.
#[test]
fn a_truthful_declaration_over_the_same_store_still_opens_and_answers() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let write_scoped = persisted_simple_scoped_store(directory.path());
    let read_scoped = persisted_read_scoped_store(directory.path());

    let unnarrowed = ConsumerReader::open(
        &write_scoped,
        declaration("vigil_signals", PERSISTED_SIGNALS_DDL, &[]),
    )
    .expect("a declaration that describes the persisted table opens");
    assert_eq!(
        vec!["edge-1".to_string(), "server-1".to_string()],
        ids_in(&unnarrowed, "vigil_signals"),
        "a truthful declaration claiming no label narrows nothing and is served every row, as it \
         always has been",
    );
    unnarrowed.close().expect("close the reader");

    let narrowed = ConsumerReader::open(
        &read_scoped,
        declaration("vigil_events", EVENTS_DDL, &["edge"]),
    )
    .expect("a declaration that describes the persisted read-scoped table opens");
    assert_eq!(
        vec!["edge-1".to_string()],
        ids_in(&narrowed, "vigil_events"),
        "a truthful narrowing declaration is honoured by the engine's row gate: the edge row and \
         nothing else",
    );
    narrowed.close().expect("close the reader");
}

// Route parity for the refusal. A reader reaches a store two ways — by reading the committed file
// itself, and by asking the process that owns it — and a check that runs on only one of them is a
// check a consumer cannot rely on: the same forged declaration would be refused or honoured
// depending on whether somebody happened to have the store open. Both routes must give the SAME
// typed refusal naming the SAME table, for both forged shapes.
#[test]
fn a_forged_declaration_is_refused_identically_on_the_direct_and_the_live_owner_routes() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let store = persisted_simple_scoped_store(directory.path());

    // Both routes are driven to completion before anything is asserted, so a failure reports what
    // EACH route did rather than stopping at the first one and leaving the other unmeasured.
    let direct: Vec<Outcome> = forged_shapes()
        .into_iter()
        .map(|(shape, forged)| open_forgery(&store, forged, shape))
        .collect();

    let owner = live_owner(&store);
    let over_the_owner: Vec<Outcome> = forged_shapes()
        .into_iter()
        .map(|(shape, forged)| open_forgery(&store, forged, shape))
        .collect();
    drop(owner);

    for ((shape, _), (direct_outcome, owner_outcome)) in forged_shapes()
        .into_iter()
        .zip(direct.iter().zip(&over_the_owner))
    {
        assert_eq!(
            direct_outcome, owner_outcome,
            "{shape}: a consumer must get the same answer whether it read the committed file \
             itself or asked the process that owns the store",
        );
        // Agreeing is only half the contract: two routes that agree on serving the hub's row to an
        // edge-role reader agree on the wrong answer, and a failure has to name what each of them
        // actually served rather than leaving one of the two to be inferred.
        if let (Outcome::Served(direct_rows), Outcome::Served(owner_rows)) =
            (direct_outcome, owner_outcome)
        {
            panic!(
                "{shape}: the forged declaration opened on BOTH read routes instead of being \
                 refused — reading the committed file served {direct_rows:?} and asking the live \
                 owner served {owner_rows:?}, to a reader that declared only the edge role."
            );
        }
        assert_refused_naming_the_table(direct_outcome, "the direct route", shape);
        assert_refused_naming_the_table(owner_outcome, "the live-owner route", shape);
    }
}

// Route parity for the positive control. The verification must not cost a consumer its answer on
// either route: the truthful declaration opens over the owner exactly as it does over the file, and
// the rows it is served are the same rows.
#[test]
fn a_truthful_declaration_answers_the_same_rows_on_the_direct_and_the_live_owner_routes() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let write_scoped = persisted_simple_scoped_store(directory.path());
    let read_scoped = persisted_read_scoped_store(directory.path());

    let read_directly = {
        let unnarrowed = ConsumerReader::open(
            &write_scoped,
            declaration("vigil_signals", PERSISTED_SIGNALS_DDL, &[]),
        )
        .expect("the truthful unnarrowed declaration opens on the direct route");
        let rows = ids_in(&unnarrowed, "vigil_signals");
        unnarrowed.close().expect("close the reader");

        let narrowed = ConsumerReader::open(
            &read_scoped,
            declaration("vigil_events", EVENTS_DDL, &["edge"]),
        )
        .expect("the truthful narrowing declaration opens on the direct route");
        let narrowed_rows = ids_in(&narrowed, "vigil_events");
        narrowed.close().expect("close the reader");
        (rows, narrowed_rows)
    };

    let write_scoped_owner = live_owner(&write_scoped);
    let unnarrowed = ConsumerReader::open(
        &write_scoped,
        declaration("vigil_signals", PERSISTED_SIGNALS_DDL, &[]),
    )
    .expect("the truthful unnarrowed declaration opens over the live owner");
    let owner_rows = ids_in(&unnarrowed, "vigil_signals");
    unnarrowed.close().expect("close the reader");
    drop(write_scoped_owner);

    let read_scoped_owner = live_owner(&read_scoped);
    let narrowed = ConsumerReader::open(
        &read_scoped,
        declaration("vigil_events", EVENTS_DDL, &["edge"]),
    )
    .expect("the truthful narrowing declaration opens over the live owner");
    let owner_narrowed_rows = ids_in(&narrowed, "vigil_events");
    narrowed.close().expect("close the reader");
    drop(read_scoped_owner);

    assert_eq!(
        read_directly.0, owner_rows,
        "the truthful unnarrowed declaration must be served the same rows over the owner as over \
         the file",
    );
    assert_eq!(
        read_directly.1, owner_narrowed_rows,
        "the truthful narrowing declaration must be served the same rows over the owner as over \
         the file",
    );
    assert_eq!(
        vec!["edge-1".to_string()],
        owner_narrowed_rows,
        "the engine's row gate narrows on the owner route too: the edge row and nothing else",
    );
}
