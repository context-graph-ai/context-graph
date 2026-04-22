mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::Utc;
use common::entity_fixture::seed_entity_fixture;
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateIntention, EntityId, EntityType, IntentionId,
    IntentionOrigin, IntentionStatus, ListDecisionFilter, ListEntityFilter, ReasoningStep, Store,
};
use criterion::{Criterion, criterion_group, criterion_main};
use tempfile::TempDir;

const DECISION_BUDGET: Duration = Duration::from_millis(100);
const ENTITY_BUDGET_P95: Duration = Duration::from_millis(100);
const ENTITY_SAMPLES: usize = 20;

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

fn create_intention_input(context_id: ContextId, description: &str) -> CreateIntention {
    CreateIntention {
        id: None,
        description: description.to_string(),
        status: IntentionStatus::Active,
        origin: IntentionOrigin::Human,
        context_id,
        properties: BTreeMap::new(),
        tags: vec![],
    }
}

fn reasoning(label: &str) -> Vec<ReasoningStep> {
    vec![ReasoningStep {
        operation: format!("op-{label}"),
        details: format!("details-{label}"),
        result: format!("result-{label}"),
        agent_id: Some("claude-code-instance-dev".to_string()),
        timestamp: Utc::now(),
    }]
}

fn create_decision_input(
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
) -> CreateDecision {
    CreateDecision {
        decision_type: "architecture".to_string(),
        description: description.to_string(),
        reasoning: reasoning(description),
        confidence: Some(0.9),
        intention_ids,
        based_on_entity_ids,
        tags: vec!["architecture".to_string()],
        properties: BTreeMap::new(),
        context_id,
    }
}

fn bench_decision_list_by_intention_10k(c: &mut Criterion) {
    let (_tempdir, db_path, store) = fixture();
    let ctx = store
        .create_context(context_input("client-acme"))
        .expect("context");
    let i1 = store
        .create_intention(create_intention_input(ctx.id, "pick a web framework"))
        .expect("I1");
    let i2 = store
        .create_intention(create_intention_input(ctx.id, "pick a message broker"))
        .expect("I2");
    let d1 = store
        .create_decision(create_decision_input(
            ctx.id,
            "use axum",
            vec![i1.id],
            vec![],
        ))
        .expect("D1");
    let d2 = store
        .create_decision(create_decision_input(
            ctx.id,
            "use actix",
            vec![i1.id],
            vec![],
        ))
        .expect("D2");
    let d3 = store
        .create_decision(create_decision_input(
            ctx.id,
            "use nats",
            vec![i2.id],
            vec![],
        ))
        .expect("D3");
    for i in 0..10_000 {
        store
            .create_decision(create_decision_input(
                ctx.id,
                &format!("unrelated-{i}"),
                vec![i2.id],
                vec![],
            ))
            .expect("seed unrelated decisions");
    }
    drop(store);
    let store = reopen(&db_path);

    let filter = || ListDecisionFilter {
        intention_id: Some(i1.id),
        ..ListDecisionFilter::default()
    };
    let warm = store.list_decisions(filter()).expect("list decisions");
    assert_eq!(warm, vec![d2.clone(), d1.clone()]);
    assert!(!warm.iter().any(|d| d.id == d3.id));
    assert_eq!(
        store.last_query_trace().expect("query trace"),
        "edge_type=SERVES"
    );

    c.bench_function("decision_list_by_intention_10k", |b| {
        b.iter(|| store.list_decisions(filter()).expect("list decisions"));
    });

    let started = Instant::now();
    let _ = store.list_decisions(filter()).expect("list decisions");
    let elapsed = started.elapsed();
    assert!(
        elapsed <= DECISION_BUDGET,
        "decision_list_by_intention_10k exceeded budget: {elapsed:?} > {DECISION_BUDGET:?}"
    );
}

fn bench_entity_list_filters_use_engine_indexes_5k(c: &mut Criterion) {
    let (_tempdir, db_path, store) = fixture();
    let ctx = store
        .create_context(context_input("platform"))
        .expect("context")
        .id;
    seed_entity_fixture(&store, ctx, 5_000);
    drop(store);
    let store = reopen(&db_path);

    let make_filter = |needle: &str| ListEntityFilter {
        name_contains: Some(needle.to_string()),
        entity_type: Some(EntityType::Service),
        include_archived: false,
        ..ListEntityFilter::default()
    };
    let first = store
        .list_entities(make_filter("pay"))
        .expect("list entities first");
    let second = store
        .list_entities(make_filter("pay"))
        .expect("list entities second");
    let third = store
        .list_entities(make_filter("PAY"))
        .expect("list entities uppercase");
    assert_eq!(first, second);
    assert_eq!(first, third);
    assert!(
        first
            .iter()
            .all(|e| e.entity_type == EntityType::Service && e.name.to_lowercase().contains("pay"))
    );
    assert!(
        first
            .windows(2)
            .all(|pair| pair[0].created_at > pair[1].created_at
                || (pair[0].created_at == pair[1].created_at && pair[0].id >= pair[1].id))
    );
    let trace = store.last_query_trace().expect("entity query trace");
    assert!(trace.contains("entity_type"));
    assert!(trace.contains("name_contains"));

    c.bench_function("entity_list_filters_use_engine_indexes_5k", |b| {
        b.iter(|| {
            store
                .list_entities(make_filter("PaY"))
                .expect("list entities")
        });
    });

    let mut samples = Vec::with_capacity(ENTITY_SAMPLES);
    for _ in 0..ENTITY_SAMPLES {
        let started = Instant::now();
        let _ = store
            .list_entities(make_filter("PaY"))
            .expect("list entities");
        samples.push(started.elapsed());
    }
    samples.sort();
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    assert!(
        p95 <= ENTITY_BUDGET_P95,
        "entity_list_filters_use_engine_indexes_5k p95 exceeded budget: {p95:?} > {ENTITY_BUDGET_P95:?}"
    );
}

criterion_group!(
    benches,
    bench_decision_list_by_intention_10k,
    bench_entity_list_filters_use_engine_indexes_5k
);
criterion_main!(benches);
