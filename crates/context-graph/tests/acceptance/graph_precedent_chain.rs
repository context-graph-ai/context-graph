use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateIntention, DecisionId, EntityId,
    IntentionId, IntentionOrigin, IntentionStatus, ReasoningStep, Store,
};
use tempfile::TempDir;

macro_rules! ok {
    ($expr:expr, $step:literal) => {{
        let result = $expr;
        assert!(result.is_ok(), "{}: expected Ok, got {:?}", $step, result);
        result.unwrap()
    }};
}

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
    precedent_ids: Vec<DecisionId>,
) -> CreateDecision {
    CreateDecision {
        decision_type: "vendor_choice".to_string(),
        description: description.to_string(),
        reasoning: reasoning(description),
        confidence: Some(0.9),
        intention_ids,
        based_on_entity_ids,
        tags: vec![],
        properties: BTreeMap::new(),
        context_id,
        precedent_ids,
    }
}

// Use case: A payment-platform reviewer at client-acme sees Decision D_b — "switch the payment provider to Stripe" — and asks what earlier Decision it cites as precedent. The engine follows the CITES edge and returns `[D_b, D_a]` after reopen.
#[test]
fn precedent_chain_walks_cites() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "maintain payment provider SLA")),
        "tx1: intention"
    );
    let d_a = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "use PayPal",
            vec![i.id],
            vec![],
            vec![],
        )),
        "tx2: create d_a"
    );
    let d_b = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "switch to Stripe",
            vec![i.id],
            vec![],
            vec![d_a.id],
        )),
        "tx3: create d_b cites d_a"
    );
    drop(store);
    let store = reopen(&db_path);
    let chain = ok!(store.precedent_chain(d_b.id, 10), "tx4: precedent_chain");
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].id, d_b.id);
    assert_eq!(chain[1].id, d_a.id);
}

// Use case: A consultant returning to a long-running engagement walks an 8-level reasoning chain. The engine returns all 8 Decisions in strict newest-first order — `[D8, D7, ..., D1]` — never a set or a prefix.
#[test]
fn precedent_chain_deep() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "evolve caching strategy")),
        "tx1: intention"
    );
    let mut decisions: Vec<context_graph::Decision> = Vec::with_capacity(8);
    for n in 1..=8u32 {
        let precedents = if n == 1 {
            vec![]
        } else {
            vec![decisions[(n - 2) as usize].id]
        };
        let d = ok!(
            store.create_decision(create_decision_input(
                ctx,
                &format!("D{n}"),
                vec![i.id],
                vec![],
                precedents,
            )),
            "txN: create Dn"
        );
        decisions.push(d);
    }
    drop(store);
    let store = reopen(&db_path);
    let chain = ok!(
        store.precedent_chain(decisions[7].id, 10),
        "tx9: precedent_chain"
    );
    let expected: Vec<DecisionId> = (0..8).rev().map(|i| decisions[i].id).collect();
    let actual: Vec<DecisionId> = chain.iter().map(|d| d.id).collect();
    assert_eq!(actual, expected);
}

// Use case: An agent at client-acme attempts to close a CITES cycle via the admin raw write path (fresh-id `create_decision` cannot close a cycle by construction). The engine refuses with `CgError::CyclicCitation` and the pre-existing chain remains acyclic after reopen.
#[test]
fn precedent_chain_cycle_rejected_at_create() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "pick a serialisation format")),
        "tx1: intention"
    );
    let d_a = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "use JSON",
            vec![i.id],
            vec![],
            vec![],
        )),
        "tx2: d_a"
    );
    let d_b = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "use Protobuf",
            vec![i.id],
            vec![],
            vec![d_a.id],
        )),
        "tx3: d_b cites d_a"
    );
    let d_c = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "use FlatBuffers",
            vec![i.id],
            vec![],
            vec![d_b.id],
        )),
        "tx4: d_c cites d_b"
    );
    let err = store.admin_raw_insert_cites(d_a.id, d_c.id).unwrap_err();
    assert!(
        matches!(err, CgError::CyclicCitation),
        "expected CyclicCitation, got {:?}",
        err
    );
    drop(store);
    let store = reopen(&db_path);
    let chain = ok!(store.precedent_chain(d_c.id, 10), "tx5: chain after reopen");
    let actual: Vec<DecisionId> = chain.iter().map(|d| d.id).collect();
    assert_eq!(actual, vec![d_c.id, d_b.id, d_a.id]);
}

// Use case: A consultant requests a precedent chain at depth 20 intending to pull an unbounded reasoning trail. The engine refuses with `CgError::InvalidDepth` — silent clamping would return a truncated chain that the consultant misreads as the full lineage.
#[test]
fn precedent_chain_depth_cap_rejects_over_ten() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "evolve caching strategy")),
        "tx1: intention"
    );
    let mut decisions: Vec<context_graph::Decision> = Vec::with_capacity(15);
    for n in 0..15u32 {
        let precedents = if n == 0 {
            vec![]
        } else {
            vec![decisions[(n - 1) as usize].id]
        };
        let d = ok!(
            store.create_decision(create_decision_input(
                ctx,
                &format!("D{n}"),
                vec![i.id],
                vec![],
                precedents,
            )),
            "txN: create Dn"
        );
        decisions.push(d);
    }
    drop(store);
    let store = reopen(&db_path);
    let err = store.precedent_chain(decisions[14].id, 20_u8).unwrap_err();
    assert!(
        matches!(
            err,
            CgError::InvalidDepth {
                requested: 20,
                cap: 10
            }
        ),
        "expected InvalidDepth {{ requested: 20, cap: 10 }}, got {:?}",
        err
    );
    let ok_chain = ok!(
        store.precedent_chain(decisions[14].id, 10_u8),
        "tx16: cap accepted"
    );
    assert!(!ok_chain.is_empty());
}
