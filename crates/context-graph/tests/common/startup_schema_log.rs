//! Reading a store's whole schema log back, and asserting the startup schema is one commit.
//!
//! Shared by the two proofs that own this contract: the fresh-store one
//! (`tests/startup_schema_is_one_commit.rs`) and the two-node one
//! (`tests/a_second_node_receives_the_whole_startup_schema.rs`). They are separate binaries because
//! one opens a file and the other runs a real sync fixture, and a binary's wall-clock ceiling
//! should describe one of those, not their sum.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use context_graph::Store;
use context_graph::schema::{INDEXES, TABLES};
use contextdb_core::Lsn;
use contextdb_engine::sync_types::DdlChange;

/// Every schema commit a fresh store's open authors, including the engine's work-ledger schema.
pub const STORE_OPEN_SCHEMA_COMMITS: usize = 1;

/// Startup schema operations include vector columns, rank indexes, composite foreign keys,
/// and immutable table declarations.
pub const TABLES_WITH_VECTOR_RANK_OR_DAG_DECLARATIONS: &[&str] = &[
    "evidence",
    "entity_reference_embeddings_128",
    "entity_reference_embeddings_256",
    "entity_reference_embeddings_512",
    "entity_reference_embeddings_768",
    "digests",
    "edges",
];

/// `(name -> commit)` for every table and index in a store's whole schema log, plus how many
/// statements each commit carries and the order the statements were authored in.
pub struct SchemaLog {
    pub table_commit: BTreeMap<String, Lsn>,
    pub index_commit: BTreeMap<String, Lsn>,
    pub statements_per_commit: BTreeMap<Lsn, usize>,
    pub author_order: Vec<String>,
}

pub fn schema_log(store: &Store) -> SchemaLog {
    let changes = store.sync_database().changes_since(Lsn(0));
    assert_eq!(
        changes.ddl.len(),
        changes.ddl_lsn.len(),
        "every schema change carries its commit"
    );
    let mut log = SchemaLog {
        table_commit: BTreeMap::new(),
        index_commit: BTreeMap::new(),
        statements_per_commit: BTreeMap::new(),
        author_order: Vec::new(),
    };
    for (ddl, lsn) in changes.ddl.iter().zip(&changes.ddl_lsn) {
        *log.statements_per_commit.entry(*lsn).or_default() += 1;
        match ddl {
            DdlChange::CreateTable { name, .. } => {
                log.author_order.push(format!("table {name}"));
                log.table_commit.insert(name.clone(), *lsn);
            }
            DdlChange::CreateIndex { name, .. } => {
                log.author_order.push(format!("index {name}"));
                log.index_commit.insert(name.clone(), *lsn);
            }
            DdlChange::CreateEventType { name, .. } => {
                log.author_order.push(format!("event type {name}"));
            }
            other => log.author_order.push(format!("{other:?}")),
        }
    }
    log
}

/// The one commit every startup declaration is visible in. A node may author schema of its own
/// after the startup schema -- a sync client records its peer contacts in a table of its own -- so
/// this asserts what the startup schema does, not that the store never declares anything else.
pub fn assert_whole_startup_schema_is_one_commit(store: &Store, stage: &str) -> Lsn {
    let log = schema_log(store);
    let tables = TABLES
        .iter()
        .map(|table| table.name)
        .collect::<BTreeSet<_>>();
    let indexes = INDEXES
        .iter()
        .map(|index| index.name)
        .collect::<BTreeSet<_>>();
    let mut commits = BTreeSet::new();
    for name in &tables {
        let commit = log.table_commit.get(*name).unwrap_or_else(|| {
            panic!("{stage}: startup table {name} is missing from the schema log")
        });
        commits.insert(*commit);
    }
    for name in &indexes {
        let commit = log.index_commit.get(*name).unwrap_or_else(|| {
            panic!("{stage}: startup index {name} is missing from the schema log")
        });
        commits.insert(*commit);
    }
    assert_eq!(
        commits.len(),
        1,
        "{stage}: the whole startup schema is visible in one commit, got {commits:?} across {:?}",
        log.statements_per_commit
    );
    let commit = *commits.first().expect("one startup schema commit");
    for name in TABLES_WITH_VECTOR_RANK_OR_DAG_DECLARATIONS {
        assert_eq!(
            log.table_commit[*name], commit,
            "{stage}: {name} declares a vector column, a rank policy or DAG edges and still \
             becomes visible with the rest of the startup schema"
        );
    }
    commit
}
