//! Immutable intention history — schema 10 → 11 and the old-root migration discipline.
//!
//! At schema 11 intention-history pointers and predecessor links make every wording immutable.
//! Opening an old schema-10 root with the new binary loses no captured data: it is refused with a
//! typed legacy-schema error while the raw and memory it holds survive on disk, re-derivable under
//! the pre-release re-import discipline.

use std::collections::{BTreeMap, HashMap};

use context_graph::schema::{INDEXES, SUBSCRIPTION_EVENT_TYPE_DDL, TABLES};
use context_graph::{CgError, Store, StoreConfig};
use contextdb_core::{TxId, Value as DbValue};
use contextdb_engine::Database;
use uuid::Uuid;

use crate::fabric_common::FabricNode;

const SCHEMA_ELEVEN_ONLY_TABLES: [&str; 3] = [
    "intention_current_snapshots",
    "intention_snapshot_links",
    "decision_intention_citations",
];

fn install_genuine_schema_ten(db: &Database) {
    contextdb_engine::work_ledger::install_work_ledger_schema(db)
        .expect("install schema-10 work ledger");
    for table in TABLES {
        if SCHEMA_ELEVEN_ONLY_TABLES.contains(&table.name) {
            continue;
        }
        // Schema 10 had the snapshot rows, but had not yet made that table append-only.
        let ddl = if table.name == "intention_snapshots" {
            table.ddl.replace(") IMMUTABLE SYNC", ") SYNC")
        } else {
            table.ddl.to_string()
        };
        db.execute(&ddl, &HashMap::new())
            .unwrap_or_else(|error| panic!("install schema-10 table {}: {error:?}", table.name));
    }
    for index in INDEXES {
        db.execute(index.ddl, &HashMap::new())
            .unwrap_or_else(|error| panic!("install schema-10 index {}: {error:?}", index.name));
    }
    for ddl in SUBSCRIPTION_EVENT_TYPE_DDL {
        db.execute(ddl, &HashMap::new())
            .unwrap_or_else(|error| panic!("install schema-10 event type {ddl}: {error:?}"));
    }
}

fn seed_schema_ten_rows(db: &Database) {
    let context_id = Uuid::now_v7();
    let intention_id = Uuid::now_v7();
    let decision_id = Uuid::now_v7();
    let digest_id = Uuid::now_v7();
    db.execute(
        "INSERT INTO cg_schema_version (id, version, installed_at) VALUES ($id, $version, $installed_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Int64(1)),
            ("version".to_string(), DbValue::Int64(10)),
            ("installed_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("seed genuine schema-10 version row");
    db.execute(
        "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(context_id)),
            ("name".to_string(), DbValue::Text("schema-ten-context".to_string())),
            ("labels".to_string(), DbValue::Json(serde_json::json!(["migration"]))),
            ("properties".to_string(), DbValue::Json(serde_json::json!({"source": "schema-10"}))),
            ("created_at".to_string(), DbValue::TxId(TxId(2))),
        ]),
    )
    .expect("seed schema-10 context");
    db.execute(
        "INSERT INTO intentions (id, description, status, context_id, origin, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $properties, $tags, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(intention_id)),
            ("description".to_string(), DbValue::Text("survive the upgrade".to_string())),
            ("status".to_string(), DbValue::Text("active".to_string())),
            ("context_id".to_string(), DbValue::Uuid(context_id)),
            ("origin".to_string(), DbValue::Text("human".to_string())),
            ("properties".to_string(), DbValue::Json(serde_json::json!({}))),
            ("tags".to_string(), DbValue::Json(serde_json::json!(["upgrade"]))),
            ("created_at".to_string(), DbValue::TxId(TxId(3))),
        ]),
    )
    .expect("seed schema-10 intention");
    db.execute(
        "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(decision_id)),
            ("decision_type".to_string(), DbValue::Text("technical".to_string())),
            ("description".to_string(), DbValue::Text("keep the auth module at src/auth.py".to_string())),
            ("reasoning".to_string(), DbValue::Json(serde_json::json!([]))),
            ("confidence".to_string(), DbValue::Float64(0.9)),
            ("status".to_string(), DbValue::Text("active".to_string())),
            ("agent_id".to_string(), DbValue::Null),
            ("context_id".to_string(), DbValue::Uuid(context_id)),
            ("properties".to_string(), DbValue::Json(serde_json::json!({}))),
            ("tags".to_string(), DbValue::Json(serde_json::json!(["upgrade"]))),
            ("created_at".to_string(), DbValue::TxId(TxId(4))),
        ]),
    )
    .expect("seed schema-10 decision");
    db.execute(
        "INSERT INTO digests (id, source, session_ref, summary, kind, vector_text, dimension, context_id, properties, created_at) VALUES ($id, $source, $session_ref, $summary, $kind, $vector_text, $dimension, $context_id, $properties, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(digest_id)),
            ("source".to_string(), DbValue::Text("session".to_string())),
            ("session_ref".to_string(), DbValue::Text("schema-10/s1".to_string())),
            ("summary".to_string(), DbValue::Text("raw capture survives".to_string())),
            ("kind".to_string(), DbValue::Text("conversation".to_string())),
            ("vector_text".to_string(), DbValue::Vector(vec![0.0; 768])),
            ("dimension".to_string(), DbValue::Int64(768)),
            ("context_id".to_string(), DbValue::Uuid(context_id)),
            ("properties".to_string(), DbValue::Json(serde_json::json!({"schema": 10}))),
            ("created_at".to_string(), DbValue::TxId(TxId(5))),
        ]),
    )
    .expect("seed schema-10 raw digest");
}

fn table_row_bytes(db: &Database) -> BTreeMap<String, Vec<Vec<u8>>> {
    db.table_names()
        .into_iter()
        .map(|table| {
            let mut rows = db
                .scan(&table, db.snapshot())
                .unwrap_or_else(|error| panic!("scan {table}: {error:?}"))
                .into_iter()
                .map(|row| {
                    serde_json::to_vec(
                        &row.values
                            .into_iter()
                            .collect::<BTreeMap<String, DbValue>>(),
                    )
                    .expect("serialize row values deterministically")
                })
                .collect::<Vec<_>>();
            rows.sort();
            (table, rows)
        })
        .collect()
}

fn table_topology(db: &Database) -> Vec<String> {
    let mut tables = db.table_names();
    tables.sort();
    tables
}

#[test]
fn fresh_schema_eleven_root_carries_intention_history_tables() {
    let node = FabricNode::open("fresh_schema_eleven");
    let tables = node.store.sync_database().table_names();
    assert!(
        SCHEMA_ELEVEN_ONLY_TABLES
            .iter()
            .all(|expected| tables.iter().any(|actual| actual == expected)),
        "a fresh schema-11 root carries the immutable intention-history tables: {tables:?}"
    );
}

#[test]
fn old_schema_ten_root_is_refused_without_data_loss() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cg.db");

    let (before_topology, before_rows) = {
        let db = Database::open(&path).expect("create raw schema-10 database");
        install_genuine_schema_ten(&db);
        seed_schema_ten_rows(&db);
        let topology = table_topology(&db);
        for table in SCHEMA_ELEVEN_ONLY_TABLES {
            assert!(
                !topology.iter().any(|actual| actual == table),
                "the genuine schema-10 fixture must not pre-install schema-11 table {table}: {topology:?}"
            );
        }
        let rows = table_row_bytes(&db);
        (topology, rows)
    };

    let result = Store::open(StoreConfig {
        db_path: path.clone(),
        ..Default::default()
    });
    assert!(
        matches!(
            result,
            Err(CgError::LegacySchemaDetected {
                found_version: 10,
                expected_version: 11,
            })
        ),
        "opening a genuine schema-10 root under the schema-11 binary yields the typed legacy-schema error, not a crash or wipe: {result:?}"
    );

    let raw = Database::open(&path).expect("raw database open after refused schema-11 open");
    assert_eq!(
        table_topology(&raw),
        before_topology,
        "default schema-11 open must leave the schema-10 table topology exactly unchanged"
    );
    assert_eq!(
        table_row_bytes(&raw),
        before_rows,
        "default schema-11 open must leave every persisted schema-10 row byte-for-byte unchanged"
    );
}
