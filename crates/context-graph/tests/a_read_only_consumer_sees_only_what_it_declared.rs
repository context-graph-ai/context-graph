#![cfg(unix)]
//! The read-only reading surface a downstream consumer falls back to, and the promise it makes.
//!
//! ## What a consumer is entitled to
//!
//! A downstream runtime that cannot take a store — because another process owns it, or because it
//! must not risk changing it — is still entitled to READ the rows it owns. It declares the tables
//! and the scope labels it is allowed to see, opens the store ONCE, and gets back either a reader
//! or a typed refusal it can act on. Nothing about that open may bring a store into existence:
//! a typo, an unmounted volume, or a link whose target is gone must come back as a refusal, never
//! as an empty store that reports success and swallows everything written into it afterwards.
//!
//! ## What the surface must never expose
//!
//! The declaration is the whole of what the reader can see. A query naming context-graph's own
//! tables, the engine's own tables, or any table the consumer did not declare is refused by name —
//! and so is the store's table inventory, which must come back intersected with the declaration
//! rather than listing every table in the store.
//!
//! Scope labels are part of the declaration too, and the ENGINE enforces them. The reader carries
//! its declared labels into the read session, and a row outside them is INVISIBLE rather than
//! refused: an edge reader asking for the hub's row gets an empty answer, not an error. That is
//! the only shape of the promise that holds, because a surface that keeps its rows by REFUSING can
//! only refuse the query shapes it recognizes — `scope = 'edge' OR 1 = 1`, `NOT (scope = 'edge')`,
//! an IN-list, a LIKE, a subquery each re-widen a query an analyzer believed it had narrowed, and
//! the hub's rows come back. Every such shape is proved here to answer in-scope rows and nothing
//! else.
//!
//! ## The refusal taxonomy
//!
//! MISSING and UNREADABLE are different answers on purpose. "It is not there" sends an operator to
//! restore a backup; a store that IS there and cannot be looked at needs a permission or a mount
//! instead. Every refusal leaves the store, its companion and the directory holding them exactly
//! as they were found.
//!
//! Every assertion below is about a typed value, a row, or a pathname. Nothing waits on elapsed
//! time and nothing sleeps.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use context_graph::{
    CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, ScopeLabel, Store,
    StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::MetadataBody;

/// A consumer table whose rows carry the role they belong to, declared with the READ half the
/// engine's row gate acts on, so a reader opened as one role is served that role's rows and is
/// never shown another's.
const EVENTS_DDL: &str = "CREATE TABLE vigil_events (id TEXT PRIMARY KEY, body TEXT, scope_label TEXT SCOPE_LABEL_READ ('edge', 'server') WRITE ('edge', 'server'))";
/// A consumer table with no role on it at all: the plain case a reader journeys through. Its
/// vector column is what a table-keyed `SHOW` has to answer about — and has to stay silent about
/// for every table the consumer did not declare.
const NOTES_DDL: &str = "CREATE TABLE vigil_notes (id TEXT PRIMARY KEY, body TEXT, embedding VECTOR(4)) SYNC CONFLICT KEEP LATEST";
/// A consumer table whose scope column is the WRITE-scoping simple form. The engine's row gate acts
/// only on the READ set of a split declaration, so this column narrows nothing a reader is served —
/// which is why a declaration that claims read narrowing over it cannot be honoured.
const SIGNALS_DDL: &str = "CREATE TABLE vigil_signals (id TEXT PRIMARY KEY, body TEXT, scope_label TEXT SCOPE_LABEL ('edge', 'server'))";

/// The tables the consumer owns inside the store context-graph opens.
fn declared_schema() -> ConsumerSchema {
    ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![
            ConsumerTable {
                name: "vigil_events".to_string(),
                ddl: EVENTS_DDL.to_string(),
            },
            ConsumerTable {
                name: "vigil_notes".to_string(),
                ddl: NOTES_DDL.to_string(),
            },
        ],
    }
}

/// Everything an edge-role consumer is entitled to see: its own two tables, and only rows carrying
/// the edge label.
fn edge_declaration() -> ConsumerDeclaration {
    ConsumerDeclaration {
        schema: declared_schema(),
        scope_labels: BTreeSet::from([ScopeLabel::new("edge")]),
    }
}

/// A declaration naming exactly one of the consumer's two tables, for the proof that the surface
/// exposes what was DECLARED rather than what the namespace happens to hold.
fn notes_only_declaration() -> ConsumerDeclaration {
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "vigil".to_string(),
            tables: vec![ConsumerTable {
                name: "vigil_notes".to_string(),
                ddl: NOTES_DDL.to_string(),
            }],
        },
        scope_labels: BTreeSet::new(),
    }
}

fn no_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// One entry in a directory, with enough of it to notice any change. A link is recorded by what it
/// points at rather than by what is at the other end, so a directory holding a dangling link is
/// still comparable.
#[derive(Debug, PartialEq, Eq)]
enum EntryImage {
    File(Vec<u8>),
    Symlink(PathBuf),
    Directory,
}

/// Everything a directory holds, by name, down to the bytes. Two of these being equal is what "the
/// refusal left the directory untouched" means: not merely that no store appeared, but that no
/// companion, no lock and no residue of any kind did either.
fn directory_image(directory: &Path) -> BTreeMap<String, EntryImage> {
    std::fs::read_dir(directory)
        .expect("read the directory holding the store")
        .map(|entry| {
            let entry = entry.expect("read one directory entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = entry.file_type().expect("classify one directory entry");
            let image = if kind.is_symlink() {
                EntryImage::Symlink(
                    std::fs::read_link(entry.path()).expect("read one link's target"),
                )
            } else if kind.is_dir() {
                EntryImage::Directory
            } else {
                EntryImage::File(std::fs::read(entry.path()).expect("read one file's bytes"))
            };
            (name, image)
        })
        .collect()
}

/// A scratch directory named the way every open below resolves it, so a refusal that names a path
/// can be compared against one written here.
fn scratch_root(directory: &tempfile::TempDir) -> PathBuf {
    std::fs::canonicalize(directory.path()).expect("resolve the scratch directory")
}

fn insert_event(store: &Store, id: &str, body: &str, scope: &str) {
    store
        .sync_database()
        .execute(
            "INSERT INTO vigil_events (id, body, scope_label) VALUES ($id, $body, $scope)",
            &HashMap::from([
                ("id".to_string(), Value::Text(id.to_string())),
                ("body".to_string(), Value::Text(body.to_string())),
                ("scope".to_string(), Value::Text(scope.to_string())),
            ]),
        )
        .expect("an unrestricted handle owns every row its declaration accepts");
}

fn insert_note(store: &Store, id: &str, body: &str) {
    store
        .sync_database()
        .execute(
            "INSERT INTO vigil_notes (id, body) VALUES ($id, $body)",
            &HashMap::from([
                ("id".to_string(), Value::Text(id.to_string())),
                ("body".to_string(), Value::Text(body.to_string())),
            ]),
        )
        .expect("the consumer owns writes to its own table");
}

/// A real store, with context-graph's own tables and the consumer's, holding one row for each
/// role — then closed and let go, so a reader takes the committed file rather than an owner.
fn seeded_store(directory: &Path) -> PathBuf {
    let path = directory.join("consumer.db");
    let store = Store::open_with_consumer_schema(
        StoreConfig {
            db_path: path.clone(),
            ..Default::default()
        },
        Vec::new(),
        declared_schema(),
    )
    .expect("the consumer's first run creates the store and the tables it declared");
    insert_event(&store, "edge-1", "seen at the edge", "edge");
    insert_event(&store, "server-1", "seen at the hub", "server");
    insert_note(&store, "note-1", "a note the consumer owns");
    drop(store);
    path
}

/// Every table the store's vector indexes are keyed by, read through an unrestricted handle: what a
/// table-keyed `SHOW` hands back when no declaration is applied to it.
fn vector_index_tables_in_the_whole_store(path: &Path) -> BTreeSet<String> {
    let store = Store::open_with_consumer_schema(
        StoreConfig {
            db_path: path.to_path_buf(),
            ..Default::default()
        },
        Vec::new(),
        declared_schema(),
    )
    .expect("an unrestricted handle reopens the store the consumer already created");
    let answered = store
        .sync_database()
        .execute("SHOW VECTOR_INDEXES", &no_params())
        .expect("an unrestricted handle is answered about every vector index in the store");
    let tables = text_column(&answered.rows).into_iter().collect();
    drop(store);
    tables
}

/// The one table a write-scoping consumer declares, and a store holding a row for each of its
/// roles — then closed and let go, so a reader takes the committed file.
fn simple_scoped_schema() -> ConsumerSchema {
    ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_signals".to_string(),
            ddl: SIGNALS_DDL.to_string(),
        }],
    }
}

fn simple_scoped_store(directory: &Path) -> PathBuf {
    let path = directory.join("write-scoped.db");
    let store = Store::open_with_consumer_schema(
        StoreConfig {
            db_path: path.clone(),
            ..Default::default()
        },
        Vec::new(),
        simple_scoped_schema(),
    )
    .expect("the consumer's first run creates the store and the table it declared");
    for (id, body, scope) in [
        ("edge-1", "raised at the edge", "edge"),
        ("server-1", "raised at the hub", "server"),
    ] {
        store
            .sync_database()
            .execute(
                "INSERT INTO vigil_signals (id, body, scope_label) VALUES ($id, $body, $scope)",
                &HashMap::from([
                    ("id".to_string(), Value::Text(id.to_string())),
                    ("body".to_string(), Value::Text(body.to_string())),
                    ("scope".to_string(), Value::Text(scope.to_string())),
                ]),
            )
            .expect("an unrestricted handle owns every row its declaration accepts");
    }
    drop(store);
    path
}

/// An open that must refuse, with its refusal.
fn expect_refusal(result: context_graph::Result<ConsumerReader>, context: &str) -> CgError {
    match result {
        Ok(reader) => {
            let _ = reader.close();
            panic!("{context}: the read-only open answered instead of refusing");
        }
        Err(error) => error,
    }
}

fn text_column(rows: &[Vec<Value>]) -> Vec<String> {
    rows.iter()
        .map(|row| match &row[0] {
            Value::Text(text) => text.clone(),
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect()
}

// A store that is not there is a refusal that names it, and the open leaves the directory exactly
// as it found it: no store, no companion, no lock for the next caller to trip over.
#[test]
fn an_absent_store_is_refused_as_missing_and_leaves_the_directory_untouched() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let absent = root.join("never-created.db");
    let before = directory_image(&root);

    let refusal = expect_refusal(
        ConsumerReader::open(&absent, edge_declaration()),
        "absent store",
    );

    let CgError::StoreMissing { path } = &refusal else {
        panic!("a store that is not there must be typed missing, got: {refusal:?}");
    };
    assert_eq!(
        &absent, path,
        "the refusal must name the store it looked for"
    );
    assert!(
        !absent.exists(),
        "a read-only open must never bring a store into existence at {absent:?}",
    );
    assert_eq!(
        before,
        directory_image(&root),
        "the refusal must leave the directory byte-for-byte as it found it",
    );
}

// A link whose target is gone is the dangerous case: created through, it hands the consumer a
// brand-new empty deployment and reports success. It must refuse, and answer no first-start
// defaults.
#[test]
fn a_dangling_link_is_refused_and_never_answers_first_start_defaults() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let target = root.join("target-that-is-gone.db");
    let link = root.join("store-link.db");
    std::os::unix::fs::symlink(&target, &link)
        .expect("plant a link where a store the consumer expects used to be");
    let before = directory_image(&root);

    let refusal = expect_refusal(
        ConsumerReader::open(&link, edge_declaration()),
        "dangling link",
    );

    let named = match &refusal {
        CgError::StoreMissing { path } => path.clone(),
        other => panic!("a link whose target is gone must be a typed refusal, got: {other:?}"),
    };
    assert!(
        named == link || named == target,
        "the refusal must name the store it could not open; it named {named:?}",
    );
    assert!(
        !target.exists(),
        "a link whose target is gone must never be created through: nothing may appear at \
         {target:?}",
    );
    assert_eq!(
        before,
        directory_image(&root),
        "the refusal must leave the directory byte-for-byte as it found it",
    );
}

// A store that IS there and cannot be looked at is a different answer from one that is not there:
// calling it missing sends an operator to restore a backup when the fix is a permission.
#[test]
fn a_store_this_process_cannot_look_at_is_refused_as_unreadable_never_as_missing() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let vault = root.join("vault");
    std::fs::create_dir(&vault).expect("create the directory the store lives in");
    let store = seeded_store(&vault);
    let before = directory_image(&vault);

    std::fs::set_permissions(&vault, std::fs::Permissions::from_mode(0o000))
        .expect("seal the directory the store lives in");
    let sealed_answer = match ConsumerReader::open(&store, edge_declaration()) {
        Ok(reader) => {
            let _ = reader.close();
            None
        }
        Err(error) => Some(error),
    };
    std::fs::set_permissions(&vault, std::fs::Permissions::from_mode(0o700))
        .expect("unseal the directory so the rest of the journey can look at it");

    let refusal =
        sealed_answer.expect("a read-only open of a store this process cannot look at must refuse");
    assert!(
        !matches!(refusal, CgError::StoreMissing { .. }),
        "a store that cannot be looked at is UNREADABLE, not missing: {refusal:?}",
    );
    let CgError::StoreUnreadable { path, .. } = &refusal else {
        panic!("a metadata failure must be the typed unreadable refusal, got: {refusal:?}");
    };
    assert_eq!(
        &store, path,
        "the refusal must name the store it could not look at",
    );
    assert_eq!(
        before,
        directory_image(&vault),
        "the refusal must leave the directory byte-for-byte as it found it",
    );
}

// The reading surface is file-backed only. An in-memory request has no committed file to read and
// is refused by type, before any engine call, rather than quietly opening something empty.
#[test]
fn an_in_memory_request_is_refused_because_the_read_surface_is_file_backed() {
    let refusal = expect_refusal(
        ConsumerReader::open(":memory:", edge_declaration()),
        "in-memory request",
    );

    let CgError::InMemoryStoreHasNoReadSurface { requested } = &refusal else {
        panic!("an in-memory read request must be its own typed refusal, got: {refusal:?}");
    };
    assert_eq!(
        ":memory:", requested,
        "the refusal must repeat what the caller asked for",
    );
}

// The whole journey a consumer makes: open once, read its own rows, ask about its own table, page
// through a cursor, and close. Nothing here is a second transport and nothing polls.
#[test]
fn a_declared_reader_journeys_from_open_through_sql_metadata_and_cursor_to_close() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);

    let reader = ConsumerReader::open(&store, edge_declaration())
        .expect("a stopped store opens read-only for the consumer that declared its tables");

    let answered = reader
        .execute("SELECT id FROM vigil_notes ORDER BY id", &no_params())
        .expect("a declared table answers the reader");
    assert_eq!(
        vec!["note-1".to_string()],
        text_column(&answered.rows),
        "the reader reads back the row the consumer wrote",
    );

    let schema = reader
        .table_schema("vigil_notes")
        .expect("a declared table describes itself to the reader");
    let MetadataBody::Schema { schema } = &schema.body else {
        panic!("asking a declared table for its shape must answer with a schema: {schema:?}");
    };
    assert_eq!(
        "vigil_notes", schema.table,
        "the answer must be about the table that was asked for",
    );

    let mut cursor = reader
        .open_cursor("SELECT id FROM vigil_notes ORDER BY id", &no_params())
        .expect("a declared table opens as a cursor");
    assert_eq!(
        vec!["note-1".to_string()],
        text_column(
            &cursor
                .first_page()
                .expect("the cursor was just opened")
                .rows
        ),
        "the page produced at open carries the declared table's row",
    );
    assert!(
        !cursor
            .first_page()
            .expect("the cursor was just opened")
            .has_more,
        "one row is the whole answer, so nothing remains after the first page",
    );
    cursor.close().expect("the reader closes its own cursor");

    reader.close().expect("the reader closes its session");
}

// context-graph's own tables are not the consumer's to read. Naming one is refused by the
// declaration gate, whichever internal table it is.
#[test]
fn a_query_reaching_a_context_graph_internal_table_is_refused() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader =
        ConsumerReader::open(&store, edge_declaration()).expect("open the declared reader");

    for internal in ["cg_schema_version", "intentions", "entities"] {
        let refusal = reader
            .execute(&format!("SELECT id FROM {internal}"), &no_params())
            .expect_err("a consumer must not read context-graph's own tables");
        let CgError::UndeclaredTableRefused { table } = &refusal else {
            panic!("reading {internal} must be refused by name, got: {refusal:?}");
        };
        assert_eq!(
            internal, table,
            "the refusal must name the table that was reached for",
        );
    }

    reader.close().expect("close the reader");
}

// A table the consumer owns but did NOT declare is refused too: the declaration is the whole of
// what the reader can see, not the namespace it happens to sit in.
#[test]
fn a_query_reaching_a_table_the_consumer_did_not_declare_is_refused() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader = ConsumerReader::open(&store, notes_only_declaration())
        .expect("open a reader that declared one of its two tables");

    let refusal = reader
        .execute("SELECT id FROM vigil_events", &no_params())
        .expect_err("a table left out of the declaration is not readable through the surface");
    let CgError::UndeclaredTableRefused { table } = &refusal else {
        panic!("an undeclared table must be refused by name, got: {refusal:?}");
    };
    assert_eq!("vigil_events", table);

    reader
        .execute("SELECT id FROM vigil_notes", &no_params())
        .expect("the declared table still answers");

    reader.close().expect("close the reader");
}

// The exposes-nothing-beyond-the-declaration proof: the store holds context-graph's whole schema
// and both consumer tables, and the reader's inventory names only what its declaration named.
#[test]
fn the_readers_inventory_exposes_nothing_beyond_its_declaration() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);

    let reader = ConsumerReader::open(&store, notes_only_declaration())
        .expect("open a reader that declared one table");
    let mut inventory = reader
        .table_inventory()
        .expect("the reader lists what it may see");
    inventory.sort();
    assert_eq!(
        vec!["vigil_notes".to_string()],
        inventory,
        "the inventory must be the declaration, never the store's whole table list",
    );

    let refusal = reader
        .table_schema("cg_schema_version")
        .expect_err("an undeclared table describes itself to nobody through this surface");
    let CgError::UndeclaredTableRefused { table } = &refusal else {
        panic!("asking about an undeclared table must be refused by name, got: {refusal:?}");
    };
    assert_eq!("cg_schema_version", table);

    reader.close().expect("close the reader");
}

// Scope labels are part of the declaration, and the ENGINE enforces them. An edge reader reads the
// edge row; the hub's row is not an error the reader has to catch, it is simply not there — whether
// the SQL names the hub's label outright, hands it in as a parameter, or restricts nothing at all.
// Filtering rather than refusing is the whole point: the same table answers the reader with the row
// it may see in the very same statement that hides the one it may not.
#[test]
fn an_out_of_scope_row_is_invisible_to_the_reader_rather_than_refused() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader =
        ConsumerReader::open(&store, edge_declaration()).expect("open the edge-role reader");

    let answered = reader
        .execute(
            "SELECT id FROM vigil_events WHERE scope_label = 'edge' ORDER BY id",
            &no_params(),
        )
        .expect("the role the reader declared answers");
    assert_eq!(
        vec!["edge-1".to_string()],
        text_column(&answered.rows),
        "the edge reader reads the edge row and nothing else",
    );

    let named = reader
        .execute(
            "SELECT id FROM vigil_events WHERE scope_label = 'server' ORDER BY id",
            &no_params(),
        )
        .expect("a label the reader did not declare is invisible to it, never an error it catches");
    assert!(
        text_column(&named.rows).is_empty(),
        "naming the hub's label must answer with no rows, got: {:?}",
        text_column(&named.rows),
    );

    let parameterized = reader
        .execute(
            "SELECT id FROM vigil_events WHERE scope_label = $scope ORDER BY id",
            &HashMap::from([("scope".to_string(), Value::Text("server".to_string()))]),
        )
        .expect("a parameter is not a way around the declaration, and not a reason to refuse");
    assert!(
        text_column(&parameterized.rows).is_empty(),
        "the hub's label handed in as a parameter must answer with no rows, got: {:?}",
        text_column(&parameterized.rows),
    );

    let unrestricted = reader
        .execute("SELECT id FROM vigil_events ORDER BY id", &no_params())
        .expect(
            "a query restricting nothing is answered, narrowed by the engine to the declaration",
        );
    assert_eq!(
        vec!["edge-1".to_string()],
        text_column(&unrestricted.rows),
        "a query that restricts nothing must still see only the row its declaration allows",
    );

    reader.close().expect("close the reader");
}

// The surface is genuinely read-only: a statement that would change the store is refused before it
// reaches the engine, even on a table the consumer owns and declared.
#[test]
fn a_statement_that_would_change_the_store_is_refused_by_the_read_only_surface() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader =
        ConsumerReader::open(&store, edge_declaration()).expect("open the declared reader");

    let statement = "INSERT INTO vigil_notes (id, body) VALUES ('note-2', 'from a reader')";
    let refusal = reader
        .execute(statement, &no_params())
        .expect_err("a reading surface must refuse a write");
    let CgError::ReadOnlySurfaceRefusedWrite { statement: named } = &refusal else {
        panic!("a write on the read surface must be its own refusal, got: {refusal:?}");
    };
    assert_eq!(
        statement, named,
        "the refusal must repeat the statement it would not run",
    );

    reader.close().expect("close the reader");

    let after = ConsumerReader::open(&store, notes_only_declaration())
        .expect("reopen the store to check nothing landed");
    let rows = after
        .execute("SELECT id FROM vigil_notes ORDER BY id", &no_params())
        .expect("read the declared table back");
    assert_eq!(
        vec!["note-1".to_string()],
        text_column(&rows.rows),
        "a refused write must leave no row behind",
    );
    after.close().expect("close the reader");
}

// Two consumers reading the same stopped store at once both get answers: reading is not a lock
// somebody has to queue for.
#[test]
fn two_read_only_opens_of_the_same_stopped_store_both_answer() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);

    let first = ConsumerReader::open(&store, edge_declaration()).expect("the first reader opens");
    let second = ConsumerReader::open(&store, edge_declaration())
        .expect("a second reader opens the same stopped store while the first still holds it");

    for (which, reader) in [("first", &first), ("second", &second)] {
        let answered = reader
            .execute("SELECT id FROM vigil_notes ORDER BY id", &no_params())
            .unwrap_or_else(|error| panic!("the {which} reader must answer: {error:?}"));
        assert_eq!(
            vec!["note-1".to_string()],
            text_column(&answered.rows),
            "the {which} reader reads the declared row",
        );
    }

    first.close().expect("the first reader closes");
    second.close().expect("the second reader closes");
}

// The proof the superseded design could not make. Every one of these shapes re-widens a predicate
// an analyzer believed it had narrowed — a disjunction, a negation, an IN-list naming both roles, a
// LIKE that matches everything, a subquery over the same table. Each is ANSWERED rather than
// refused, and each answer holds only what the reader's declared labels admit, because the engine
// decides row by row and never reads the SQL as a promise.
#[test]
fn every_query_shape_that_re_widens_a_narrowed_predicate_answers_only_in_scope_rows() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader =
        ConsumerReader::open(&store, edge_declaration()).expect("open the edge-role reader");

    let only_the_edge_row = vec!["edge-1".to_string()];
    let nothing: Vec<String> = Vec::new();
    for (shape, sql, expected) in [
        (
            "a disjunction that re-widens the restriction",
            "SELECT id FROM vigil_events WHERE scope_label = 'edge' OR 1 = 1 ORDER BY id",
            &only_the_edge_row,
        ),
        (
            "a negation that inverts the restriction",
            "SELECT id FROM vigil_events WHERE NOT (scope_label = 'edge') ORDER BY id",
            &nothing,
        ),
        (
            "an IN-list naming both roles",
            "SELECT id FROM vigil_events WHERE scope_label IN ('edge', 'server') ORDER BY id",
            &only_the_edge_row,
        ),
        (
            "a LIKE that matches every row",
            "SELECT id FROM vigil_events WHERE body LIKE '%' ORDER BY id",
            &only_the_edge_row,
        ),
        (
            "a subquery over the same scope-labelled table",
            "SELECT id FROM vigil_events WHERE id IN (SELECT id FROM vigil_events) ORDER BY id",
            &only_the_edge_row,
        ),
    ] {
        let answered = reader.execute(sql, &no_params()).unwrap_or_else(|error| {
            panic!("{shape} must be answered under the engine's own narrowing, got: {error:?}")
        });
        assert_eq!(
            *expected,
            text_column(&answered.rows),
            "{shape} must hand back only the rows the reader's declaration admits",
        );
    }

    // A cursor is a second way out of the same session, and it is narrowed by the same gate.
    let mut cursor = reader
        .open_cursor("SELECT id FROM vigil_events ORDER BY id", &no_params())
        .expect("an unrestricted read of the scope-labelled table opens as a cursor");
    assert_eq!(
        only_the_edge_row,
        text_column(
            &cursor
                .first_page()
                .expect("the cursor was just opened")
                .rows
        ),
        "the page produced at open must carry only the row the declaration admits",
    );
    cursor.close().expect("the reader closes its own cursor");

    reader.close().expect("close the reader");
}

// A `SHOW` keyed by table is data about tables, so it answers about the declared ones and stays
// silent about every other — exactly as the reader's table inventory already does. Reaching the
// engine unchecked, it would name context-graph's own embedding tables to a consumer that declared
// neither.
#[test]
fn a_table_keyed_show_answers_only_about_the_tables_the_consumer_declared() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);

    let whole_store = vector_index_tables_in_the_whole_store(&store);
    assert!(
        whole_store.contains("vigil_notes"),
        "the declared table's vector index must be in the store to be answered about: {whole_store:?}",
    );
    let undeclared: BTreeSet<&String> = whole_store
        .iter()
        .filter(|table| table.as_str() != "vigil_notes")
        .collect();
    assert!(
        !undeclared.is_empty(),
        "the store must hold a vector index on a table this reader never declared, or the \
         intersection below proves nothing",
    );

    let reader = ConsumerReader::open(&store, notes_only_declaration())
        .expect("open a reader that declared one table");
    let answered = reader
        .execute("SHOW VECTOR_INDEXES", &no_params())
        .expect("a table-keyed SHOW answers the reader about its own tables");
    let named: BTreeSet<String> = text_column(&answered.rows).into_iter().collect();
    assert_eq!(
        BTreeSet::from(["vigil_notes".to_string()]),
        named,
        "a table-keyed SHOW must come back intersected with the declaration; it named {undeclared:?} \
         besides",
    );

    reader.close().expect("close the reader");
}

// A `SHOW` with no table of its own reports how the store itself is configured, which is not the
// consumer's data and is not withheld from it. Refusing these would be the over-correction: the
// reader would lose the store-level answers it is entitled to along with the table-keyed ones it
// is not. Only these two are genuinely store-level; `SHOW SYNC_CONFLICT_POLICY` is keyed by table
// and is proved above.
#[test]
fn a_store_level_show_with_no_table_of_its_own_answers_through_the_reader() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader =
        ConsumerReader::open(&store, edge_declaration()).expect("open the declared reader");

    for statement in ["SHOW MEMORY_LIMIT", "SHOW DISK_LIMIT"] {
        let answered = reader
            .execute(statement, &no_params())
            .unwrap_or_else(|error| {
                panic!("{statement} reports store-level configuration and must answer: {error:?}")
            });
        assert!(
            !answered.rows.is_empty(),
            "{statement} must answer with the store's own configuration rather than nothing",
        );
    }

    reader.close().expect("close the reader");
}

// The declaration-level refusals survive the move to engine-enforced scope. A table the consumer
// did not declare is refused by name however the statement reaches it — through a join, through a
// subquery, or bound behind a common table expression.
#[test]
fn an_undeclared_table_reached_through_a_join_or_a_subquery_is_still_refused() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader = ConsumerReader::open(&store, notes_only_declaration())
        .expect("open a reader that declared one of its two tables");

    for statement in [
        "SELECT vigil_notes.id FROM vigil_notes INNER JOIN vigil_events ON vigil_notes.id = \
         vigil_events.id",
        "SELECT id FROM vigil_notes WHERE id IN (SELECT id FROM vigil_events)",
        "WITH borrowed AS (SELECT id FROM vigil_events) SELECT id FROM borrowed",
    ] {
        let refusal = reader
            .execute(statement, &no_params())
            .expect_err("an undeclared table is not readable through the surface by any route");
        let CgError::UndeclaredTableRefused { table } = &refusal else {
            panic!("`{statement}` must be refused by name, got: {refusal:?}");
        };
        assert_eq!(
            "vigil_events", table,
            "the refusal must name the table that was reached for",
        );
    }

    reader.close().expect("close the reader");
}

// `SHOW SYNC_CONFLICT_POLICY` is keyed by table: it renders a row for every table that declared a
// conflict policy, and context-graph's own schema declares dozens. So it is intersected with the
// declaration exactly as the vector-index answer is — the consumer sees its own table's arbitration
// and the store's default, and learns the name of no table it did not declare.
#[test]
fn the_sync_conflict_policy_show_answers_only_about_declared_tables_and_the_store_default() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);
    let reader = ConsumerReader::open(&store, notes_only_declaration())
        .expect("open a reader that declared one table");

    let answered = reader
        .execute("SHOW SYNC_CONFLICT_POLICY", &no_params())
        .expect("a reader is entitled to how the store arbitrates its own table");
    let named: BTreeSet<String> = text_column(&answered.rows).into_iter().collect();
    assert_eq!(
        BTreeSet::from([
            "keep_first".to_string(),
            "vigil_notes=keep_latest".to_string(),
        ]),
        named,
        "the answer must be the store's default plus the declared table's own arbitration, and \
         nothing keyed by a table the consumer never declared",
    );

    reader.close().expect("close the reader");
}

// The engine's row gate acts only on the READ set of a split scope declaration, so a consumer
// asking to be narrowed to a label over a write-scoping column is asking for something no layer can
// enforce. It is refused at open, by type, naming the table — never opened and then served every
// row as though the narrowing had been applied. A declaration that claims no narrowing over the
// same table is served unnarrowed, exactly as it is today.
#[test]
fn a_read_narrowing_declaration_over_a_write_scoped_column_is_refused_at_open() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = simple_scoped_store(&root);

    let refusal = expect_refusal(
        ConsumerReader::open(
            &store,
            ConsumerDeclaration {
                schema: simple_scoped_schema(),
                scope_labels: BTreeSet::from([ScopeLabel::new("edge")]),
            },
        ),
        "a read-narrowing declaration over a write-scoping column",
    );
    let CgError::ConsumerTableRefused { subject, reason } = &refusal else {
        panic!("an unenforceable read declaration must be refused by type, got: {refusal:?}");
    };
    assert_eq!(
        "vigil_signals", subject,
        "the refusal must name the table whose scope column cannot narrow a read",
    );
    assert!(
        reason.contains("SCOPE_LABEL_READ"),
        "the refusal must name the engine's own read-scoping vocabulary, so the consumer can see \
         what its declaration would have to say instead; it said: {reason}",
    );

    let reader = ConsumerReader::open(
        &store,
        ConsumerDeclaration {
            schema: simple_scoped_schema(),
            scope_labels: BTreeSet::new(),
        },
    )
    .expect("a declaration claiming no narrowing over the same table opens as it always has");
    let answered = reader
        .execute("SELECT id FROM vigil_signals ORDER BY id", &no_params())
        .expect("a consumer that asked for no narrowing reads its own table");
    assert_eq!(
        vec!["edge-1".to_string(), "server-1".to_string()],
        text_column(&answered.rows),
        "a declaration with no labels narrows nothing, so every row its table accepts answers",
    );

    reader.close().expect("close the reader");
}

// A declaration whose CREATE TABLE text this surface cannot read is refused at open, and that is
// true of every consumer — not only one that also asked to be narrowed to a scope label. A gate
// that reads each table's scope column only while labels are present would open on a guess for the
// label-free majority: it would admit a table it never parsed, then answer queries against it.
#[test]
fn a_declaration_the_surface_cannot_read_is_refused_even_with_no_scope_labels() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let store = seeded_store(&root);

    for (shape, ddl) in [
        ("text that is not SQL at all", "not a create table at all"),
        (
            "SQL this surface can parse that declares no table",
            "SELECT id FROM vigil_notes",
        ),
    ] {
        let refusal = expect_refusal(
            ConsumerReader::open(
                &store,
                ConsumerDeclaration {
                    schema: ConsumerSchema {
                        namespace: "vigil".to_string(),
                        tables: vec![ConsumerTable {
                            name: "vigil_notes".to_string(),
                            ddl: ddl.to_string(),
                        }],
                    },
                    scope_labels: BTreeSet::new(),
                },
            ),
            shape,
        );
        let CgError::ConsumerTableRefused { subject, .. } = &refusal else {
            panic!("{shape} must be refused by type at open, got: {refusal:?}");
        };
        assert_eq!(
            "vigil_notes", subject,
            "the refusal must name the table whose declaration could not be read",
        );
    }

    let reader = ConsumerReader::open(&store, notes_only_declaration())
        .expect("a label-free declaration this surface CAN read still opens");
    reader
        .execute("SELECT id FROM vigil_notes ORDER BY id", &no_params())
        .expect("and still answers over the table it declared");
    reader.close().expect("close the reader");
}
