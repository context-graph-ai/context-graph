//! A fresh Store authors its startup tables with as few schema commits as the engine allows.
//!
//! A peer replays each schema commit it receives over its whole schema, so a startup schema
//! authored one statement per commit made a new node's first push and pull cost the square of the
//! schema size. The engine's transactional schema staging carries only tables with no vector, rank
//! or DAG column (upstream gap 12), so those tables and every index still commit on their own. The
//! plain tables whose foreign-key parents are all plain share one commit; the plain tables that
//! reference a vector table share a second one, after their parents.

use std::collections::{BTreeMap, BTreeSet};

use context_graph::schema::{INDEXES, TABLES};
use context_graph::{Store, StoreConfig};
use contextdb_core::Lsn;
use contextdb_engine::sync_types::DdlChange;

/// The startup tables the engine cannot stage in a transaction: a vector column or DAG edges.
const UNSTAGEABLE_TABLES: &[&str] = &[
    "evidence",
    "entity_reference_embeddings_128",
    "entity_reference_embeddings_256",
    "entity_reference_embeddings_512",
    "entity_reference_embeddings_768",
    "digests",
    "edges",
];

/// The plain startup tables that reference a vector table, and so commit after it.
const PLAIN_TABLES_AFTER_VECTOR_PARENTS: &[&str] = &[
    "evidence_embedding_handles",
    "evidence_availability_log",
    "digest_embedding_handles",
];

/// Every schema commit a fresh store's open authors, cg's own and the engine's work-ledger schema.
const STORE_OPEN_SCHEMA_COMMITS: usize = 55;

#[test]
fn a_fresh_store_installs_its_plain_startup_tables_in_the_fewest_schema_commits() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("fresh.db"),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open a fresh store");
    let changes = store.sync_database().changes_since(Lsn(0));
    assert_eq!(
        changes.ddl.len(),
        changes.ddl_lsn.len(),
        "every schema change carries its commit"
    );

    let tables = TABLES
        .iter()
        .map(|table| table.name)
        .collect::<BTreeSet<_>>();
    let indexes = INDEXES
        .iter()
        .map(|index| index.name)
        .collect::<BTreeSet<_>>();
    let mut table_commit = BTreeMap::new();
    let mut index_commit = BTreeMap::new();
    let mut statements_per_commit = BTreeMap::<Lsn, usize>::new();
    for (ddl, lsn) in changes.ddl.iter().zip(&changes.ddl_lsn) {
        *statements_per_commit.entry(*lsn).or_default() += 1;
        match ddl {
            DdlChange::CreateTable { name, .. } if tables.contains(name.as_str()) => {
                table_commit.insert(name.clone(), *lsn);
            }
            DdlChange::CreateIndex { name, .. } if indexes.contains(name.as_str()) => {
                index_commit.insert(name.clone(), *lsn);
            }
            _ => {}
        }
    }
    assert_eq!(
        table_commit.len(),
        tables.len(),
        "every startup table is in the schema log"
    );
    assert_eq!(
        index_commit.len(),
        indexes.len(),
        "every startup index is in the schema log"
    );

    let commits_of = |names: &mut dyn Iterator<Item = &str>| {
        names
            .map(|name| table_commit[name])
            .collect::<BTreeSet<_>>()
    };
    let first_plain = commits_of(&mut tables.iter().copied().filter(|name| {
        !UNSTAGEABLE_TABLES.contains(name) && !PLAIN_TABLES_AFTER_VECTOR_PARENTS.contains(name)
    }));
    assert_eq!(
        first_plain.len(),
        1,
        "the plain startup tables with plain parents share one schema commit, got {} commits",
        first_plain.len()
    );
    let second_plain = commits_of(&mut PLAIN_TABLES_AFTER_VECTOR_PARENTS.iter().copied());
    assert_eq!(
        second_plain.len(),
        1,
        "the plain startup tables that reference a vector table share one schema commit, got {} \
         commits",
        second_plain.len()
    );
    assert_ne!(
        first_plain, second_plain,
        "the tables that reference a vector table commit after it, not with the first batch"
    );
    for name in UNSTAGEABLE_TABLES {
        assert_eq!(
            statements_per_commit[&table_commit[*name]], 1,
            "{name} cannot be staged in a transaction and commits on its own"
        );
    }
    for (name, lsn) in &index_commit {
        assert_eq!(
            statements_per_commit[lsn], 1,
            "index {name} commits on its own"
        );
    }
    assert_eq!(
        statements_per_commit.len(),
        STORE_OPEN_SCHEMA_COMMITS,
        "a fresh store's open authors exactly this many schema commits"
    );
}
