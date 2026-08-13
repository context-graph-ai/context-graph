//! The public consumer-table seam: an embedding consumer that already embeds the
//! library declares the tables it owns inside the store context-graph opens, so it
//! needs neither a second database nor a direct engine dependency of its own.
//!
//! What these tests hold context-graph to: the declaration is namespaced and the
//! consumer owns everything under its prefix; context-graph refuses its own table
//! names; the schema-version gate reads the same version whether or not consumer
//! tables are present; and the `RETAIN` / `SYNC` / `SYNC CONFLICT` text on a
//! consumer declaration reaches contextdb exactly as written — including the
//! engine's own refusal of `SYNC SAFE` on a keyless table, which the seam must
//! leave reachable rather than pre-empt with a message of its own; and that a
//! consumer opening with a role — an edge, not the hub — is held to the
//! `SCOPE_LABEL` its own declaration wrote down.

use std::collections::{BTreeSet, HashMap};

use context_graph::{
    CG_SCHEMA_VERSION, CgError, ConsumerSchema, ConsumerTable, ScopeLabel, Store, StoreConfig,
};
use contextdb_core::{Error as EngineError, Value};
use contextdb_engine::{Database, natural_key_columns_for_meta};

const SETTINGS_DDL: &str = "CREATE TABLE vigil_settings (name TEXT PRIMARY KEY, requested TEXT, source TEXT) RETAIN 48 HOURS SYNC SAFE SYNC TWO WAY SYNC CONFLICT KEEP LATEST";

fn settings_schema() -> ConsumerSchema {
    ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_settings".to_string(),
            ddl: SETTINGS_DDL.to_string(),
        }],
    }
}

fn config(dir: &tempfile::TempDir) -> StoreConfig {
    StoreConfig {
        db_path: dir.path().join("consumer.db"),
        ..Default::default()
    }
}

fn open_with(config: StoreConfig, schema: ConsumerSchema) -> context_graph::Result<Store> {
    Store::open_with_consumer_schema(config, Vec::new(), schema)
}

fn no_params() -> HashMap<String, Value> {
    HashMap::new()
}

fn write_setting(store: &Store, name: &str, requested: &str) {
    let params = HashMap::from([
        ("name".to_string(), Value::Text(name.to_string())),
        ("requested".to_string(), Value::Text(requested.to_string())),
        ("source".to_string(), Value::Text("add-on".to_string())),
    ]);
    store
        .sync_database()
        .execute(
            "INSERT INTO vigil_settings (name, requested, source) VALUES ($name, $requested, $source)",
            &params,
        )
        .expect("a consumer owns writes to its own table");
}

fn read_settings(store: &Store) -> Vec<(String, String)> {
    let rows = store
        .sync_database()
        .execute(
            "SELECT name, requested FROM vigil_settings ORDER BY name",
            &no_params(),
        )
        .expect("read the consumer table back");
    rows.rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Text(name), Value::Text(requested)) => (name.clone(), requested.clone()),
            other => panic!("unexpected consumer row shape: {other:?}"),
        })
        .collect()
}

#[test]
fn consumer_declares_a_namespaced_keyed_table_at_store_open() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = open_with(config(&dir), settings_schema()).expect("declare the consumer table");

    let meta = store
        .sync_database()
        .table_meta("vigil_settings")
        .expect("the declared table exists in the store context-graph opened");
    assert_eq!(
        Some(vec!["name".to_string()]),
        natural_key_columns_for_meta(&meta),
        "the consumer declared a keyed table and the key must reach the engine"
    );

    write_setting(&store, "detector_sample_frames", "4");
    assert_eq!(
        vec![("detector_sample_frames".to_string(), "4".to_string())],
        read_settings(&store),
        "the consumer reads back what it wrote through the store it shares with context-graph"
    );
}

#[test]
fn declaring_the_same_consumer_schema_again_keeps_the_existing_rows() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = open_with(config(&dir), settings_schema()).expect("first declaration");
    write_setting(&store, "detector_confidence_threshold", "0.55");
    drop(store);

    let store = open_with(config(&dir), settings_schema()).expect("second declaration");
    assert_eq!(
        vec![(
            "detector_confidence_threshold".to_string(),
            "0.55".to_string()
        )],
        read_settings(&store),
        "re-declaring an existing consumer table must not recreate or clear it"
    );
}

#[test]
fn context_graph_refuses_its_own_table_names() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let reserved = ConsumerSchema {
        namespace: "cg".to_string(),
        tables: vec![ConsumerTable {
            name: "cg_schema_version".to_string(),
            ddl: "CREATE TABLE cg_schema_version (id INTEGER PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "cg".to_string(),
            reason: "reserved_namespace",
        },
        open_with(config(&dir), reserved)
            .expect_err("the context-graph namespace is not available to a consumer")
    );

    let dir = tempfile::tempdir().expect("consumer tempdir");
    let outside = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "decisions".to_string(),
            ddl: "CREATE TABLE decisions (id UUID PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "decisions".to_string(),
            reason: "outside_namespace",
        },
        open_with(config(&dir), outside)
            .expect_err("an ontology table name is refused before any DDL reaches the engine")
    );

    let dir = tempfile::tempdir().expect("consumer tempdir");
    let engine_owned = ConsumerSchema {
        namespace: "work".to_string(),
        tables: vec![ConsumerTable {
            name: "work_inputs".to_string(),
            ddl: "CREATE TABLE work_inputs (id UUID PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "work".to_string(),
            reason: "reserved_namespace",
        },
        open_with(config(&dir), engine_owned)
            .expect_err("the engine's work-ledger namespace is not available to a consumer either")
    );
}

#[test]
fn a_namespace_that_lands_inside_a_reserved_one_is_refused_too() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    // `cg_extra` is not `cg`, but everything it would own — `cg_extra_settings`
    // and the rest — sits inside context-graph's own `cg_` prefix, which is the
    // thing the reserved list is protecting.
    let inside_cg = ConsumerSchema {
        namespace: "cg_extra".to_string(),
        tables: vec![ConsumerTable {
            name: "cg_extra_settings".to_string(),
            ddl: "CREATE TABLE cg_extra_settings (name TEXT PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "cg_extra".to_string(),
            reason: "reserved_namespace",
        },
        open_with(config(&dir), inside_cg)
            .expect_err("a namespace inside context-graph's own prefix is not the consumer's")
    );
}

#[test]
fn a_consumer_namespace_still_cannot_reach_a_table_context_graph_owns() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    // `audit` is nobody's reserved namespace and `audit_log` sits squarely inside
    // it — but `audit_log` is context-graph's own table, and a name check that
    // stopped at the namespace would hand it over.
    let reaching = ConsumerSchema {
        namespace: "audit".to_string(),
        tables: vec![ConsumerTable {
            name: "audit_log".to_string(),
            ddl: "CREATE TABLE audit_log (id UUID PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "audit_log".to_string(),
            reason: "reserved_table_name",
        },
        open_with(config(&dir), reaching)
            .expect_err("a table context-graph owns stays context-graph's, whoever declares it")
    );
}

#[test]
fn a_consumer_cannot_claim_a_table_name_context_graph_creates_at_runtime() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    // context-graph does not declare one entity-reference table per embedding
    // dimension up front; it creates the one a declared space needs, when that
    // space is declared. `entity_reference_embeddings_1024` is the name it will
    // create for a 1024-dimension space, and the whole family is context-graph's
    // whether or not this store holds that dimension yet — a consumer that gets
    // there first would be handed the table context-graph goes on to expect.
    let runtime_family = ConsumerSchema {
        namespace: "entity".to_string(),
        tables: vec![ConsumerTable {
            name: "entity_reference_embeddings_1024".to_string(),
            ddl: "CREATE TABLE entity_reference_embeddings_1024 (id UUID PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "entity_reference_embeddings_1024".to_string(),
            reason: "reserved_table_name",
        },
        open_with(config(&dir), runtime_family)
            .expect_err("a name context-graph creates at runtime is context-graph's own")
    );
}

#[test]
fn a_declaration_whose_quoted_name_hides_an_escaped_quote_is_refused() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    // contextdb reads a doubled quote inside a quoted identifier as one literal
    // quote, so this declaration creates `vigil_settings"shadow` — not the
    // `vigil_settings` it appears to name, and not the table the consumer
    // declared.
    let shadowed = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_settings".to_string(),
            ddl: "CREATE TABLE \"vigil_settings\"\"shadow\" (name TEXT PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "vigil_settings".to_string(),
            reason: "declaration_names_a_different_table",
        },
        open_with(config(&dir), shadowed)
            .expect_err("a doubled quote is one quote in the name, not the end of it")
    );
}

#[test]
fn a_consumer_declaration_must_create_the_table_it_names() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let mismatched = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_settings".to_string(),
            ddl: "CREATE TABLE vigil_other (id TEXT PRIMARY KEY)".to_string(),
        }],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "vigil_settings".to_string(),
            reason: "declaration_names_a_different_table",
        },
        open_with(config(&dir), mismatched)
            .expect_err("a declaration cannot name one table and create another")
    );
}

#[test]
fn a_consumer_cannot_declare_the_same_table_twice() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let duplicated = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![
            ConsumerTable {
                name: "vigil_settings".to_string(),
                ddl: SETTINGS_DDL.to_string(),
            },
            ConsumerTable {
                name: "vigil_settings".to_string(),
                ddl: "CREATE TABLE vigil_settings (name TEXT PRIMARY KEY)".to_string(),
            },
        ],
    };
    assert_eq!(
        CgError::ConsumerTableRefused {
            subject: "vigil_settings".to_string(),
            reason: "duplicate_table_name",
        },
        open_with(config(&dir), duplicated)
            .expect_err("one table name cannot arrive twice with two declarations behind it")
    );
}

#[test]
fn a_lowercase_declaration_is_accepted_the_way_contextdb_accepts_it() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let lowercase = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_settings".to_string(),
            ddl: "create table vigil_settings (name text primary key, requested text, source text)"
                .to_string(),
        }],
    };
    let store = open_with(config(&dir), lowercase).expect("contextdb parses keywords either case");
    write_setting(&store, "detector_queue_capacity", "16");
    assert_eq!(
        vec![("detector_queue_capacity".to_string(), "16".to_string())],
        read_settings(&store),
        "a consumer that writes its keywords in lower case owns the same table"
    );
}

#[test]
fn the_schema_version_gate_reads_the_same_version_with_a_consumer_table_present() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = open_with(config(&dir), settings_schema()).expect("declare the consumer table");
    write_setting(&store, "motion_sensitivity", "7");
    drop(store);

    // context-graph's own open path — the one that runs the schema-version gate —
    // must accept a store carrying consumer-namespaced tables it knows nothing about.
    let store = Store::open(config(&dir)).expect("context-graph reopens a store it shares");
    assert_eq!(
        vec![("motion_sensitivity".to_string(), "7".to_string())],
        read_settings(&store),
        "reopening through context-graph's own path leaves consumer rows untouched"
    );
    drop(store);

    // And the version the gate reports is context-graph's own, not something the
    // consumer table shifted: an older expectation still names the shipped version.
    let stale = StoreConfig {
        expected_schema_version: CG_SCHEMA_VERSION - 1,
        ..config(&dir)
    };
    assert_eq!(
        CgError::LegacySchemaDetected {
            found_version: CG_SCHEMA_VERSION,
            expected_version: CG_SCHEMA_VERSION - 1,
        },
        Store::open(stale).expect_err(
            "the consumer table must not change which schema version context-graph reports"
        )
    );
}

#[test]
fn consumer_policy_declarations_reach_the_engine_unchanged() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = open_with(config(&dir), settings_schema()).expect("declare the consumer table");
    let db = store.sync_database();
    let meta = db
        .table_meta("vigil_settings")
        .expect("the declared table exists");

    assert_eq!(
        Some(48 * 60 * 60),
        meta.default_ttl_seconds,
        "RETAIN 48 HOURS is the consumer's declaration and the engine must hold it"
    );
    assert!(
        meta.sync_safe,
        "SYNC SAFE is the consumer's declaration and the engine must hold it"
    );
    assert_eq!(
        Some("SYNC TWO WAY"),
        meta.sync_direction.map(|direction| direction.sql()),
        "the consumer's sync direction must reach the engine unchanged"
    );
    assert_eq!(
        Some("SYNC CONFLICT KEEP LATEST"),
        meta.conflict_policy
            .and_then(|policy| policy.declared_clause()),
        "the consumer's conflict policy must reach the engine unchanged"
    );
}

#[test]
fn a_declaration_using_if_not_exists_is_accepted_the_way_contextdb_accepts_it() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let if_not_exists = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_settings".to_string(),
            ddl: "CREATE TABLE IF NOT EXISTS vigil_settings (name TEXT PRIMARY KEY, requested TEXT, source TEXT)"
                .to_string(),
        }],
    };
    let store = open_with(config(&dir), if_not_exists)
        .expect("contextdb accepts IF NOT EXISTS and the declared name still names vigil_settings");
    write_setting(&store, "detector_min_area", "120");
    assert_eq!(
        vec![("detector_min_area".to_string(), "120".to_string())],
        read_settings(&store),
        "a consumer that declares IF NOT EXISTS still owns a working table"
    );
}

#[test]
fn a_declaration_using_a_quoted_identifier_is_accepted_the_way_contextdb_accepts_it() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let quoted = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_settings".to_string(),
            ddl: "CREATE TABLE \"vigil_settings\" (name TEXT PRIMARY KEY, requested TEXT, source TEXT)"
                .to_string(),
        }],
    };
    let store = open_with(config(&dir), quoted).expect(
        "contextdb accepts a quoted identifier and the declared name still names vigil_settings",
    );
    write_setting(&store, "detector_max_area", "9000");
    assert_eq!(
        vec![("detector_max_area".to_string(), "9000".to_string())],
        read_settings(&store),
        "a consumer that quotes its own table name still owns a working table"
    );
}

/// The reason the engine gives for refusing `ddl`, taken from the engine itself on
/// a database of its own rather than restated here. The seam's promise is that a
/// consumer reads this string and not a word more, which only the engine can say.
fn engine_refusal_reason(ddl: &str) -> String {
    let dir = tempfile::tempdir().expect("engine tempdir");
    let db = Database::open(dir.path().join("engine.db")).expect("open a plain engine database");
    let err = db
        .execute(ddl, &no_params())
        .expect_err("the engine refuses this declaration on its own");
    match err {
        EngineError::SchemaInvalid { reason } => reason,
        other => {
            panic!("the engine must refuse this declaration as an invalid schema, got {other:?}")
        }
    }
}

#[test]
fn the_engine_refuses_a_consumer_declaration_that_reshapes_a_table_the_engine_owns() {
    let dir = tempfile::tempdir().expect("consumer tempdir");

    // `peer_directory` is the engine's own fabric bookkeeping table, and a
    // consumer namespace can reach its name — `peer` is nobody's reserved
    // namespace. context-graph adds no name list of its own here: the engine
    // owns that name and judges each declaration against the shape its own
    // installer declares, so a declaration that reshapes the table is told, in
    // the engine's words, that the table is not one of its own. A declaration
    // restating the canonical shape exactly is the engine's to accept, and this
    // covers no claim about it.
    let ddl = "CREATE TABLE peer_directory (node_id TEXT PRIMARY KEY, note TEXT)";
    let engine_bookkeeping = ConsumerSchema {
        namespace: "peer".to_string(),
        tables: vec![ConsumerTable {
            name: "peer_directory".to_string(),
            ddl: ddl.to_string(),
        }],
    };
    let err = open_with(config(&dir), engine_bookkeeping)
        .expect_err("a table the engine owns cannot be reshaped into a consumer's own table");
    let CgError::Engine(reason) = err else {
        panic!("the engine's own refusal must surface, got {err:?}");
    };
    // The refusal the consumer reads must BE the engine's own message, whole: a
    // reason that merely opened or ended with the engine's words would also be
    // satisfied by one the seam had wrapped in a sentence of its own.
    assert_eq!(
        engine_refusal_reason(ddl),
        reason,
        "the seam must pass the engine's refusal through verbatim, not a word added"
    );
}

#[test]
fn the_engine_still_refuses_sync_safe_on_a_keyless_consumer_table() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let ddl = "CREATE TABLE vigil_keyless (name TEXT, requested TEXT) RETAIN 1 HOURS SYNC SAFE";
    let keyless = ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_keyless".to_string(),
            ddl: ddl.to_string(),
        }],
    };
    let err = open_with(config(&dir), keyless).expect_err("keyless SYNC SAFE cannot be declared");
    let CgError::Engine(reason) = err else {
        panic!("the engine's own refusal must surface, got {err:?}");
    };
    assert_eq!(
        engine_refusal_reason(ddl),
        reason,
        "the seam must pass the engine's refusal through verbatim, not a word added"
    );
}

/// A consumer table whose rows carry the role that may write them: the hub owns
/// every row this declaration accepts, and an edge owns none of them.
const HUB_SETTINGS_DDL: &str = "CREATE TABLE vigil_hub_settings (name TEXT PRIMARY KEY, requested TEXT, scope_label TEXT SCOPE_LABEL ('server'))";

fn hub_settings_schema() -> ConsumerSchema {
    ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_hub_settings".to_string(),
            ddl: HUB_SETTINGS_DDL.to_string(),
        }],
    }
}

fn write_hub_setting(store: &Store, name: &str) -> Result<(), EngineError> {
    let params = HashMap::from([
        ("name".to_string(), Value::Text(name.to_string())),
        ("requested".to_string(), Value::Text("4".to_string())),
        ("scope".to_string(), Value::Text("server".to_string())),
    ]);
    store
        .sync_database()
        .execute(
            "INSERT INTO vigil_hub_settings (name, requested, scope_label) VALUES ($name, $requested, $scope)",
            &params,
        )
        .map(|_| ())
}

fn hub_setting_names(store: &Store) -> Vec<String> {
    let rows = store
        .sync_database()
        .execute(
            "SELECT name FROM vigil_hub_settings ORDER BY name",
            &no_params(),
        )
        .expect("read the consumer table back");
    rows.rows
        .iter()
        .map(|row| match &row[0] {
            Value::Text(name) => name.clone(),
            other => panic!("unexpected consumer row shape: {other:?}"),
        })
        .collect()
}

#[test]
fn an_edge_scoped_consumer_handle_is_refused_a_row_its_declaration_reserves_for_the_hub() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = Store::open_with_consumer_schema_scoped(
        config(&dir),
        Vec::new(),
        hub_settings_schema(),
        BTreeSet::from([ScopeLabel::new("edge")]),
    )
    .expect("an edge opens the same declaration the hub does");

    let err = write_hub_setting(&store, "detector_sample_frames")
        .expect_err("an edge must not write a row its own declaration reserves for the hub");
    match err {
        EngineError::ScopeLabelViolation { requested, allowed } => {
            assert_eq!(
                ScopeLabel::new("server"),
                requested,
                "the refusal must name the role the row asked to be written as"
            );
            assert_eq!(
                BTreeSet::from([ScopeLabel::new("edge")]),
                allowed,
                "the refusal must name the role the handle was opened with"
            );
        }
        other => panic!("the engine's own scope refusal must surface, got {other:?}"),
    }

    assert!(
        hub_setting_names(&store).is_empty(),
        "a refused write must leave no row behind"
    );
}

#[test]
fn an_unrestricted_consumer_handle_writes_the_row_the_declaration_reserves_for_the_hub() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = open_with(config(&dir), hub_settings_schema())
        .expect("an unrestricted handle opens the declaration");

    write_hub_setting(&store, "detector_sample_frames")
        .expect("an unrestricted handle owns the row");
    assert_eq!(
        vec!["detector_sample_frames".to_string()],
        hub_setting_names(&store),
        "an unrestricted handle reads back the row it wrote"
    );
}

#[test]
fn a_hub_scoped_consumer_handle_writes_the_row_its_declaration_reserves_for_the_hub() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = Store::open_with_consumer_schema_scoped(
        config(&dir),
        Vec::new(),
        hub_settings_schema(),
        BTreeSet::from([ScopeLabel::new("server")]),
    )
    .expect("the hub opens the same declaration the edge does");

    write_hub_setting(&store, "detector_sample_frames")
        .expect("the role the declaration reserves the row for owns the write");
    assert_eq!(
        vec!["detector_sample_frames".to_string()],
        hub_setting_names(&store),
        "the hub reads back the row it wrote through its own narrowed handle"
    );
}

/// The same table as `HUB_SETTINGS_DDL` as an earlier version of a consumer might
/// have created it: same name, same columns, but without the `SCOPE_LABEL` the
/// hub/edge refusal rests on.
const HUB_SETTINGS_DDL_BEFORE_THE_LABEL: &str =
    "CREATE TABLE vigil_hub_settings (name TEXT PRIMARY KEY, requested TEXT, scope_label TEXT)";

fn schema_with(name: &str, ddl: &str) -> ConsumerSchema {
    ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: name.to_string(),
            ddl: ddl.to_string(),
        }],
    }
}

fn drift_refusal(
    dir: &tempfile::TempDir,
    created_with: &str,
    reopened_with: &str,
) -> (String, String) {
    let store = open_with(config(dir), schema_with("vigil_hub_settings", created_with))
        .expect("the earlier declaration creates the table");
    drop(store);

    let err = open_with(
        config(dir),
        schema_with("vigil_hub_settings", reopened_with),
    )
    .expect_err("a declaration that no longer describes the table in the store must be refused");
    match err {
        CgError::ConsumerTableDrifted { table, mismatch } => (table, mismatch),
        other => panic!("declaration drift must be refused by name, got {other:?}"),
    }
}

#[test]
fn a_declaration_that_adds_the_scope_label_the_refusal_rests_on_is_refused_on_the_table_that_lacks_it()
 {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let (table, mismatch) =
        drift_refusal(&dir, HUB_SETTINGS_DDL_BEFORE_THE_LABEL, HUB_SETTINGS_DDL);

    assert_eq!(
        "vigil_hub_settings", table,
        "the refusal must name the table whose declaration no longer matches"
    );
    assert!(
        mismatch.contains("scope_label"),
        "the refusal must name what differs, so the consumer knows the hub/edge refusal is not in force: {mismatch}"
    );
}

#[test]
fn a_declaration_that_changes_where_a_consumer_table_travels_is_refused_on_the_table_that_still_travels_the_old_way()
 {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let (table, mismatch) = drift_refusal(
        &dir,
        "CREATE TABLE vigil_hub_settings (name TEXT PRIMARY KEY, requested TEXT) SYNC TWO WAY SYNC CONFLICT KEEP LATEST",
        "CREATE TABLE vigil_hub_settings (name TEXT PRIMARY KEY, requested TEXT) SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST",
    );

    assert_eq!("vigil_hub_settings", table);
    assert!(
        mismatch.contains("sync direction") && mismatch.contains("conflict policy"),
        "the refusal must name both policies that no longer match: {mismatch}"
    );
}

#[test]
fn a_declaration_that_adds_a_column_is_refused_on_the_table_that_does_not_have_it() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let (_, mismatch) = drift_refusal(
        &dir,
        "CREATE TABLE vigil_hub_settings (name TEXT PRIMARY KEY, requested TEXT)",
        "CREATE TABLE vigil_hub_settings (name TEXT PRIMARY KEY, requested TEXT, author TEXT)",
    );

    assert!(
        mismatch.contains("column"),
        "the refusal must name the column surface that differs: {mismatch}"
    );
}

#[test]
fn the_declaration_that_created_the_table_still_opens_it() {
    let dir = tempfile::tempdir().expect("consumer tempdir");
    let store = open_with(config(&dir), hub_settings_schema()).expect("first declaration");
    write_hub_setting(&store, "detector_sample_frames").expect("the row the declaration allows");
    drop(store);

    let store = open_with(config(&dir), hub_settings_schema())
        .expect("an unchanged declaration must keep opening the table it created");
    assert_eq!(
        vec!["detector_sample_frames".to_string()],
        hub_setting_names(&store),
        "verifying the declaration must not disturb the rows already there"
    );
}
