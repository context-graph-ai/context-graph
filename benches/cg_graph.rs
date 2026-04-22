//! Criterion benches for context-graph traversal at scale.
//!
//! The matching acceptance tests in `tests/acceptance/graph_*.rs` use the
//! smallest fixture that preserves each scenario's correctness shape. Scale
//! multipliers and wall-clock budgets live here — each bench re-asserts the
//! correctness shape at scale before timing, and a budget miss panics.

mod common;

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use common::fanout_fixture::seed_fanout_fixture;
use context_graph::{CreateContext, EntityId, RelationshipType, Store, TraverseFilter};
use criterion::{Criterion, criterion_group, criterion_main};
use tempfile::TempDir;

const FANOUT_NEIGHBORS: usize = 500;
const FANOUT_LEAVES_PER_NEIGHBOR: usize = 5;
const FANOUT_BUDGET: Duration = Duration::from_millis(100);

fn fixture() -> (TempDir, PathBuf, Store) {
    let tempdir = TempDir::new().expect("tempdir");
    let db_path = tempdir.path().join("context-graph.db");
    let store = Store::open(&db_path);
    (tempdir, db_path, store)
}

fn reopen(db_path: &Path) -> Store {
    Store::open(db_path)
}

fn context_input(name: &str) -> CreateContext {
    CreateContext {
        name: name.to_string(),
        labels: vec![],
        properties: BTreeMap::new(),
    }
}

fn bench_traverse_fanout_heavy_path_500(c: &mut Criterion) {
    let (_tempdir, db_path, store) = fixture();
    let ctx = store
        .create_context(context_input("fanout-bench"))
        .expect("context")
        .id;
    let ids = seed_fanout_fixture(&store, ctx, FANOUT_NEIGHBORS, FANOUT_LEAVES_PER_NEIGHBOR);
    drop(store);
    let store = reopen(&db_path);

    let result = store
        .traverse(
            ids.hub,
            TraverseFilter {
                depth: 2,
                rel: None,
            },
        )
        .expect("traverse");

    assert_eq!(result.nodes.len(), 3001);

    let mut expected_set: HashSet<EntityId> = HashSet::new();
    expected_set.insert(ids.hub);
    for n in &ids.neighbors {
        expected_set.insert(*n);
    }
    for row in &ids.leaves {
        for l in row {
            expected_set.insert(*l);
        }
    }
    let actual_set: HashSet<EntityId> = result.nodes.iter().map(|n| n.entity_id).collect();
    assert_eq!(actual_set, expected_set);

    let start_matches: Vec<_> = result
        .nodes
        .iter()
        .filter(|n| n.entity_id == ids.hub)
        .collect();
    assert_eq!(start_matches.len(), 1);
    assert_eq!(start_matches[0].hop, 0);
    assert_eq!(start_matches[0].relationship, None);

    let neighbor_set: HashSet<EntityId> = ids.neighbors.iter().copied().collect();
    for node in result
        .nodes
        .iter()
        .filter(|n| neighbor_set.contains(&n.entity_id))
    {
        assert_eq!(node.hop, 1);
        assert_eq!(node.relationship, Some(RelationshipType::DependsOn));
    }

    let leaf_set: HashSet<EntityId> = ids.leaves.iter().flatten().copied().collect();
    for node in result
        .nodes
        .iter()
        .filter(|n| leaf_set.contains(&n.entity_id))
    {
        assert_eq!(node.hop, 2);
        assert_eq!(node.relationship, Some(RelationshipType::RelatesTo));
    }

    c.bench_function("traverse_fanout_heavy_path_500", |b| {
        b.iter(|| {
            store
                .traverse(
                    ids.hub,
                    TraverseFilter {
                        depth: 2,
                        rel: None,
                    },
                )
                .expect("traverse")
        });
    });

    let started = Instant::now();
    let _ = store
        .traverse(
            ids.hub,
            TraverseFilter {
                depth: 2,
                rel: None,
            },
        )
        .expect("traverse");
    let elapsed = started.elapsed();
    assert!(
        elapsed <= FANOUT_BUDGET,
        "traverse_fanout_heavy_path_500 exceeded budget: {elapsed:?} > {FANOUT_BUDGET:?}"
    );
}

criterion_group!(benches, bench_traverse_fanout_heavy_path_500);
criterion_main!(benches);
