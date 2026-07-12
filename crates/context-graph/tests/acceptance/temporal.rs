use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, EntityId,
    EntityType, IntentionOrigin, IntentionPatch, IntentionStatus, ReasoningStep, Store,
    StoreConfig,
};
use serde_json::{Value, json};
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
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    (tempdir, db_path, store)
}

fn reopen(db_path: &Path) -> Store {
    Store::open(StoreConfig {
        db_path: db_path.to_path_buf(),
        ..Default::default()
    })
    .expect("open")
}

fn map(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn context_input(name: &str) -> CreateContext {
    CreateContext {
        name: name.to_string(),
        labels: vec![],
        properties: BTreeMap::new(),
    }
}

fn intention_input(context_id: ContextId, description: &str) -> CreateIntention {
    CreateIntention {
        id: None,
        description: description.to_string(),
        status: IntentionStatus::Active,
        origin: IntentionOrigin::Human,
        context_id,
        properties: BTreeMap::new(),
        tags: vec![],
        blueprint_catalog_id: None,
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

fn entity_input(
    context_id: ContextId,
    entity_type: EntityType,
    name: &str,
    properties: BTreeMap<String, Value>,
) -> CreateEntity {
    CreateEntity {
        entity_type,
        name: name.to_string(),
        properties,
        tags: vec![],
        context_id,
    }
}

fn decision_input(
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<context_graph::IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
) -> CreateDecision {
    CreateDecision {
        decision_type: "temporal".to_string(),
        description: description.to_string(),
        reasoning: reasoning(description),
        confidence: Some(0.9),
        intention_ids,
        based_on_entity_ids,
        basis_fields: None,
        based_on_snapshots: vec![],
        tags: vec![],
        properties: BTreeMap::new(),
        context_id,
        precedent_ids: vec![],
        agent_id: None,
    }
}

// Use case: an audit/forensics use — a regulator reads a Q3 incident report and must see the Intention
// text in force the day the alert fired, not the edited current wording. Delivers the "what did we
// mean" temporal promise on the Intention snapshot timeline: history() returns oldest-first
// contiguous snapshots, intention_at() selects by [valid_from, valid_to) covering the query TxId.
// Regression guard on cg-crud's shipped intention_at + history(IntentionId) surface.
#[test]
fn intention_prior_language_queryable_at_any_txid() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(store.create_context(context_input("audit")), "context").id;
    let intention = ok!(
        store.create_intention(intention_input(ctx, "use MQTT for transport")),
        "v1 intention"
    );
    let decision = ok!(
        store.create_decision(decision_input(
            ctx,
            "pick transport",
            vec![intention.id],
            vec![]
        )),
        "D1 cites v1"
    );
    let tx_decision = decision.created_at;
    let _ = ok!(
        store.update_intention(
            intention.id,
            IntentionPatch {
                description: Some("use a transport that traverses NAT; NATS chosen".to_string()),
                status: None,
                properties: None,
                tags: None,
            },
        ),
        "v2 refinement"
    );
    drop(store);
    let store = reopen(&db_path);
    let snapshots = ok!(store.history(intention.id), "history");
    assert_eq!(
        snapshots.len(),
        2,
        "two snapshot rows after v0->v1 refinement"
    );
    assert!(
        snapshots[0].valid_from < snapshots[1].valid_from,
        "oldest-first ordering"
    );
    assert_eq!(
        snapshots[0].valid_to,
        Some(snapshots[1].valid_from),
        "contiguity: snapshots[0].valid_to == snapshots[1].valid_from"
    );
    assert_eq!(snapshots[0].description, "use MQTT for transport");
    assert_eq!(snapshots[1].valid_to, None, "open last snapshot");

    let covering = ok!(
        store.intention_at(intention.id, tx_decision),
        "intention_at"
    )
    .expect("covering");
    assert_eq!(
        covering.description, "use MQTT for transport",
        "v1 wording at D1.created_at, not the current v2 text"
    );
    assert!(covering.valid_from <= tx_decision, "valid_from <= query tx");
    assert!(
        covering.valid_to.expect("closed v1") > tx_decision,
        "valid_to > query tx (half-open interval covers tx)"
    );
    Ok(())
}

// Use case: an operator auditing a past Decision asks "what was the threshold at the TxId the
// Decision fired, not today's value after a later calibration?" Delivers "what was true" on
// Entity snapshot history across three updates; state_at returns a distinct snapshot for each
// historical TxId.
// Regression guard on cg-crud's shipped state_at surface.
#[test]
fn state_at_returns_historical_snapshot_for_each_moment() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(store.create_context(context_input("audit")), "context").id;
    let entity = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "threshold-rule",
            map(&[("region", json!("us-east-1")), ("tier", json!("standard"))])
        )),
        "tx1 create entity"
    );
    ok!(
        store.update_entity(
            entity.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("region", json!("us-west-2")),
                    ("tier", json!("standard"))
                ])),
                tags: None,
            },
        ),
        "tx2 update"
    );
    ok!(
        store.update_entity(
            entity.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("region", json!("us-west-2")),
                    ("tier", json!("premium"))
                ])),
                tags: None,
            },
        ),
        "tx3 update"
    );
    drop(store);
    let store = reopen(&db_path);
    let history = ok!(store.entity_history(entity.id), "history");
    assert_eq!(history.len(), 3);
    let tx1 = history[0].valid_from;
    let tx2 = history[1].valid_from;
    let tx3 = history[2].valid_from;

    let s1 = ok!(store.state_at(entity.id, tx1), "state_at tx1").expect("snapshot at tx1");
    assert_eq!(s1.properties["region"], json!("us-east-1"));
    assert_eq!(s1.properties["tier"], json!("standard"));
    assert_eq!(s1.valid_from, tx1);
    assert_eq!(s1.valid_to, Some(tx2));

    let s2 = ok!(store.state_at(entity.id, tx2), "state_at tx2").expect("snapshot at tx2");
    assert_eq!(s2.properties["region"], json!("us-west-2"));
    assert_eq!(s2.properties["tier"], json!("standard"));
    assert_eq!(s2.valid_from, tx2);
    assert_eq!(s2.valid_to, Some(tx3));

    let s3 = ok!(store.state_at(entity.id, tx3), "state_at tx3").expect("snapshot at tx3");
    assert_eq!(s3.properties["region"], json!("us-west-2"));
    assert_eq!(s3.properties["tier"], json!("premium"));
    assert_eq!(s3.valid_from, tx3);
    assert_eq!(s3.valid_to, None);
    Ok(())
}

// Use case: audit script treats earliest-Ok(Some) as creation time. state_at at a TxId strictly
// before an Entity existed returns Ok(None), not Err — the caller need not special-case pre-creation.
// Regression guard on cg-crud's shipped state_at return shape (Ok(None) for no covering row).
#[test]
fn state_at_before_entity_existed() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(store.create_context(context_input("audit")), "context").id;
    // Establish a real pre-creation TxId via an unrelated intention operation.
    let anchor = ok!(
        store.create_intention(intention_input(ctx, "anchor — pre-entity")),
        "anchor intention"
    );
    let pre_create_tx = anchor.created_at;
    let entity = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "late",
            BTreeMap::new()
        )),
        "entity created after anchor"
    );
    assert!(
        entity.created_at > pre_create_tx,
        "entity tx > anchor tx precondition"
    );
    drop(store);
    let store = reopen(&db_path);
    let result = store.state_at(entity.id, pre_create_tx);
    assert!(
        matches!(result, Ok(None)),
        "pre-creation state_at is Ok(None), got {:?}",
        result
    );
    drop(store);
    let store = reopen(&db_path);
    let result = store.state_at(entity.id, pre_create_tx);
    assert!(
        matches!(result, Ok(None)),
        "Ok(None) persists across reopen, got {:?}",
        result
    );
    Ok(())
}

// Use case: builder-as-substrate dashboard rendering "missing" cells — state_at on a never-stored
// EntityId returns Ok(None), same as pre-creation; the caller does not distinguish.
// Regression guard on cg-crud's shipped state_at.
#[test]
fn state_at_nonexistent_entity() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(store.create_context(context_input("dashboard")), "context").id;
    // A committed operation provides a valid TxId to query at.
    let anchor = ok!(
        store.create_intention(intention_input(ctx, "anchor — nonexistent probe")),
        "anchor"
    );
    let current_txid = anchor.created_at;
    let nonexistent = EntityId::new_v7();
    let result = store.state_at(nonexistent, current_txid);
    assert!(
        matches!(result, Ok(None)),
        "nonexistent entity state_at is Ok(None), got {:?}",
        result
    );
    drop(store);
    let store = reopen(&db_path);
    let result = store.state_at(nonexistent, current_txid);
    assert!(
        matches!(result, Ok(None)),
        "Ok(None) after reopen, got {:?}",
        result
    );
    Ok(())
}

// Use case: a library consumer renders an Entity's timeline in a dashboard — no phantom gaps,
// content matches what was committed at each TxId. entity_history returns contiguous oldest-first
// EntitySnapshots; last snapshot has valid_to==None; contents equal committed values.
// Contiguous snapshots (each valid_to equals the next valid_from) at the API surface.
// Regression guard on cg-crud's shipped entity_history.
#[test]
fn history_oldest_first_contiguous() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(store.create_context(context_input("timeline")), "context").id;
    let entity = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "team-roster",
            map(&[("size", json!("4")), ("lead", json!("alice"))])
        )),
        "tx1"
    );
    ok!(
        store.update_entity(
            entity.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[("size", json!("6")), ("lead", json!("alice"))])),
                tags: None,
            },
        ),
        "tx2"
    );
    ok!(
        store.update_entity(
            entity.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[("size", json!("6")), ("lead", json!("bob"))])),
                tags: None,
            },
        ),
        "tx3"
    );
    drop(store);
    let store = reopen(&db_path);
    let history = ok!(store.entity_history(entity.id), "history");
    assert_eq!(history.len(), 3, "three snapshots after two updates");
    let tx1 = history[0].valid_from;
    let tx2 = history[1].valid_from;
    let tx3 = history[2].valid_from;
    assert_eq!(history[0].valid_from, tx1);
    assert_eq!(history[0].valid_to, Some(tx2));
    assert_eq!(history[0].properties["size"], json!("4"));
    assert_eq!(history[0].properties["lead"], json!("alice"));
    assert_eq!(history[1].valid_from, tx2);
    assert_eq!(
        history[1].valid_from,
        history[0].valid_to.expect("closed [0]"),
        "contiguity: [1].valid_from == [0].valid_to"
    );
    assert_eq!(history[1].valid_to, Some(tx3));
    assert_eq!(history[1].properties["size"], json!("6"));
    assert_eq!(history[1].properties["lead"], json!("alice"));
    assert_eq!(history[2].valid_from, tx3);
    assert_eq!(history[2].valid_to, None);
    assert_eq!(history[2].properties["size"], json!("6"));
    assert_eq!(history[2].properties["lead"], json!("bob"));
    assert!(
        history[0].valid_from < history[1].valid_from
            && history[1].valid_from < history[2].valid_from,
        "strictly increasing valid_from"
    );
    drop(store);
    let store = reopen(&db_path);
    let history2 = ok!(store.entity_history(entity.id), "history after reopen");
    assert_eq!(history, history2, "byte-identical across reopen");
    Ok(())
}

// Use case: an operator auditing rule changes across a TxId window runs "everything that
// changed in this rule and its dependencies since t1" — a diff with include_related=true
// surfaces neighbour changes split into changed and unchanged lists. Target Entity is excluded
// from both lists. FieldChange entries pin field name, old_value, and new_value to committed
// values.
#[test]
fn diff_across_related_entities() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("audit-window")),
        "context"
    )
    .id;
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "alert-rule",
            map(&[("tier", json!("standard"))])
        )),
        "target E"
    );
    let n1 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Database,
            "alert-db",
            map(&[("engine", json!("postgres")), ("max_conn", json!("100"))])
        )),
        "N1 DATABASE"
    );
    let n2 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "alert-peer",
            map(&[("region", json!("eu-west-1"))])
        )),
        "N2 SERVICE"
    );
    ok!(
        store.add_relationship(e.id, n1.id, "DEPENDS_ON".to_string()),
        "E -> N1"
    );
    ok!(
        store.add_relationship(e.id, n2.id, "DEPENDS_ON".to_string()),
        "E -> N2"
    );
    let anchor = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t1",
            map(&[("k", json!("a"))])
        )),
        "anchor t1"
    );
    let t1 = anchor.created_at;
    ok!(
        store.update_entity(
            n1.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("engine", json!("postgres-16")),
                    ("max_conn", json!("200"))
                ])),
                tags: None,
            },
        ),
        "N1 update in-window"
    );
    // Close the window with a fresh create_entity anchor (matches the pattern used by
    // diff_no_changes / diff_self_only / diff_window_contains_create_and_archive /
    // diff_on_archived_target_still_returns_pre_archive_state). Do NOT use update_entity on
    // the existing anchor to derive t2 — Entity.created_at means CREATION TIME per the
    // ontology and concepts.md, so reading an update tx from an update_entity return value
    // would either be wrong or force update_entity() to double-bind that field (breaking the
    // public type's contract).
    let anchor2 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t2",
            map(&[("k", json!("b"))])
        )),
        "anchor t2"
    );
    let t2 = anchor2.created_at;
    drop(store);
    let store = reopen(&db_path);
    let diff = ok!(store.diff(e.id, t1, t2, true), "diff include_related=true");
    assert_eq!(diff.target_id, e.id);
    assert_eq!(diff.from, t1);
    assert_eq!(diff.to, t2);
    assert_eq!(diff.changes.len(), 1, "exactly one changed neighbour (N1)");
    let changed = &diff.changes[0];
    assert_eq!(changed.entity_id, n1.id);
    assert!(
        changed.field_changes.iter().any(|fc| fc.field == "engine"
            && fc.old_value == Some(json!("postgres"))
            && fc.new_value == Some(json!("postgres-16"))),
        "FieldChange engine pinned by name + old/new values, got: {:?}",
        changed.field_changes
    );
    assert!(
        changed.field_changes.iter().any(|fc| fc.field == "max_conn"
            && fc.old_value == Some(json!("100"))
            && fc.new_value == Some(json!("200"))),
        "FieldChange max_conn pinned by name + old/new values, got: {:?}",
        changed.field_changes
    );
    assert_eq!(
        diff.unchanged.len(),
        1,
        "exactly one unchanged neighbour (N2)"
    );
    assert_eq!(diff.unchanged[0].entity_id, n2.id);
    assert!(
        diff.changes.iter().all(|c| c.entity_id != e.id)
            && diff.unchanged.iter().all(|u| u.entity_id != e.id),
        "target E.id appears in neither changes nor unchanged"
    );
    drop(store);
    let store = reopen(&db_path);
    let diff2 = ok!(store.diff(e.id, t1, t2, true), "diff after reopen");
    assert_eq!(diff, diff2, "byte-identical across reopen");
    Ok(())
}

// Use case: an admin tool surfaces rule deltas across a window — empty diff when nothing
// changed, no phantom entries from key re-ordering or value-comparison jitter. Defeats the
// "non-zero width always returns something" shortcut AND the zero-width-window trivial-empty
// shortcut (t1 < t2 strictly, with a committed operation between them).
#[test]
fn diff_no_changes() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("audit-window")),
        "context"
    )
    .id;
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "stable-rule",
            map(&[("timeout_ms", json!("500"))])
        )),
        "E"
    );
    let anchor_a = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-a",
            map(&[("k", json!("a"))])
        )),
        "anchor_a for t1"
    );
    let t1 = anchor_a.created_at;
    let anchor_b = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-b",
            map(&[("k", json!("b"))])
        )),
        "anchor_b between t1 and t2"
    );
    let t2 = anchor_b.created_at;
    assert!(t1 < t2, "strict window width");
    drop(store);
    let store = reopen(&db_path);
    let diff = ok!(
        store.diff(e.id, t1, t2, false),
        "diff include_related=false"
    );
    assert!(
        diff.changes.is_empty(),
        "no phantom change entries, got: {:?}",
        diff.changes
    );
    assert!(
        diff.unchanged.is_empty(),
        "include_related=false yields empty unchanged, got: {:?}",
        diff.unchanged
    );
    assert_eq!(diff.target_id, e.id);
    assert_eq!(diff.from, t1);
    assert_eq!(diff.to, t2);
    Ok(())
}

// Use case: builder-as-substrate single-entity audit — a plan that inspects one rule must not
// surface neighbour noise. include_related=false scopes the result to the target's OWN field
// changes; neighbour changes are excluded even when the neighbour also changed in the window.
#[test]
fn diff_self_only() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("builder-audit")),
        "context"
    )
    .id;
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::File,
            "main.rs",
            map(&[("loc", json!("200")), ("checksum", json!("aabb"))])
        )),
        "E FILE"
    );
    let n1 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Module,
            "core-module",
            map(&[("version", json!("1.0"))])
        )),
        "N1 MODULE"
    );
    ok!(
        store.add_relationship(e.id, n1.id, "DEPENDS_ON".to_string()),
        "E -> N1"
    );
    let anchor = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t1",
            map(&[("k", json!("a"))])
        )),
        "anchor t1"
    );
    let t1 = anchor.created_at;
    ok!(
        store.update_entity(
            e.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[("loc", json!("215")), ("checksum", json!("ccdd"))])),
                tags: None,
            },
        ),
        "E own-field update in-window"
    );
    ok!(
        store.update_entity(
            n1.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[("version", json!("1.1"))])),
                tags: None,
            },
        ),
        "N1 update in-window"
    );
    let anchor2 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t2",
            map(&[("k", json!("b"))])
        )),
        "anchor t2"
    );
    let t2 = anchor2.created_at;
    drop(store);
    let store = reopen(&db_path);
    let diff = ok!(
        store.diff(e.id, t1, t2, false),
        "diff include_related=false"
    );
    assert_eq!(
        diff.changes.len(),
        1,
        "exactly one change entry — the target"
    );
    let target_change = &diff.changes[0];
    assert_eq!(target_change.entity_id, e.id);
    assert_eq!(target_change.entity_name, "main.rs");
    assert!(
        target_change
            .field_changes
            .iter()
            .any(|fc| fc.field == "loc"
                && fc.old_value == Some(json!("200"))
                && fc.new_value == Some(json!("215"))),
        "FieldChange loc pinned by name + old/new, got: {:?}",
        target_change.field_changes
    );
    assert!(
        target_change
            .field_changes
            .iter()
            .any(|fc| fc.field == "checksum"
                && fc.old_value == Some(json!("aabb"))
                && fc.new_value == Some(json!("ccdd"))),
        "FieldChange checksum pinned by name + old/new, got: {:?}",
        target_change.field_changes
    );
    assert!(
        diff.unchanged.is_empty(),
        "include_related=false: unchanged is empty"
    );
    assert!(
        diff.changes.iter().all(|c| c.entity_id != n1.id)
            && diff.unchanged.iter().all(|u| u.entity_id != n1.id),
        "N1 neighbour excluded by include_related=false"
    );
    Ok(())
}

// Use case: an audit tool runs diff(rule, t1, now) but the rule was archived between t1 and
// now. diff must return Ok with the pre-archive property delta, not Err(NotFound), not an empty
// diff, not the archived row's properties on both sides.
#[test]
fn diff_on_archived_target_still_returns_pre_archive_state() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("archive-window")),
        "context"
    )
    .id;
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "rate-limit-rule",
            map(&[("tier", json!("standard")), ("region", json!("us-east-1"))])
        )),
        "E"
    );
    let anchor = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t1",
            map(&[("k", json!("a"))])
        )),
        "anchor t1"
    );
    let t1 = anchor.created_at;
    ok!(
        store.update_entity(
            e.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("tier", json!("premium")),
                    ("region", json!("us-east-1"))
                ])),
                tags: None,
            },
        ),
        "E in-window property change"
    );
    let anchor2 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t2",
            map(&[("k", json!("b"))])
        )),
        "anchor t2"
    );
    let t2 = anchor2.created_at;
    ok!(store.archive_entity(e.id), "archive target AFTER window");
    drop(store);
    let store = reopen(&db_path);
    let diff = ok!(
        store.diff(e.id, t1, t2, false),
        "diff archived target returns Ok"
    );
    assert_eq!(diff.target_id, e.id);
    assert_eq!(diff.from, t1);
    assert_eq!(diff.to, t2);
    assert_eq!(
        diff.changes.len(),
        1,
        "one change entry (target's own pre-archive delta)"
    );
    let target_change = &diff.changes[0];
    assert_eq!(target_change.entity_id, e.id);
    assert!(
        target_change
            .field_changes
            .iter()
            .any(|fc| fc.field == "tier"
                && fc.old_value == Some(json!("standard"))
                && fc.new_value == Some(json!("premium"))),
        "FieldChange tier pinned: old=standard (pre-update) new=premium (in-window, pre-archive), got: {:?}",
        target_change.field_changes
    );
    assert!(diff.unchanged.is_empty());
    Ok(())
}

// Use case: an admin tool asking "everything that changed between t1 and t2" must include
// Entities ADDED and Entities REMOVED in the window, not only Entities edited in place. Defeats
// the existed-in-both-endpoints shortcut where mid-window create/archive events silently drop
// from the diff.
#[test]
fn diff_window_contains_create_and_archive() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(store.create_context(context_input("mid-window")), "context").id;
    let e_target = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "alert-target",
            map(&[("tier", json!("standard"))])
        )),
        "target"
    );
    let n_stable = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Database,
            "alert-db",
            map(&[("engine", json!("postgres"))])
        )),
        "N_stable"
    );
    ok!(
        store.add_relationship(e_target.id, n_stable.id, "DEPENDS_ON".to_string()),
        "target -> N_stable"
    );
    let n_removed = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Module,
            "alert-legacy-module",
            map(&[("version", json!("1.0"))])
        )),
        "N_removed"
    );
    ok!(
        store.add_relationship(e_target.id, n_removed.id, "DEPENDS_ON".to_string()),
        "target -> N_removed"
    );
    let anchor = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t1",
            map(&[("k", json!("a"))])
        )),
        "anchor t1"
    );
    let t1 = anchor.created_at;
    let n_added = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Module,
            "alert-new-module",
            map(&[("version", json!("3.0"))])
        )),
        "N_added mid-window"
    );
    ok!(
        store.add_relationship(e_target.id, n_added.id, "DEPENDS_ON".to_string()),
        "target -> N_added"
    );
    ok!(
        store.archive_entity(n_removed.id),
        "archive N_removed mid-window"
    );
    let anchor2 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-t2",
            map(&[("k", json!("b"))])
        )),
        "anchor t2"
    );
    let t2 = anchor2.created_at;
    drop(store);
    let store = reopen(&db_path);
    let diff = ok!(
        store.diff(e_target.id, t1, t2, true),
        "diff include_related=true"
    );
    assert!(
        diff.changes.iter().any(|e| e.entity_id == n_added.id),
        "N_added present in changes (mid-window create reported)"
    );
    assert!(
        diff.changes.iter().any(|e| e.entity_id == n_removed.id),
        "N_removed present in changes (mid-window archive reported)"
    );
    assert_eq!(
        diff.changes.len(),
        2,
        "exactly two changed neighbours (create + archive), got {}",
        diff.changes.len()
    );
    assert_eq!(
        diff.unchanged.len(),
        1,
        "one unchanged neighbour (N_stable)"
    );
    assert_eq!(diff.unchanged[0].entity_id, n_stable.id);
    assert!(
        diff.changes
            .iter()
            .chain(diff.unchanged.iter())
            .all(|e| e.entity_id != e_target.id),
        "target appears in neither list"
    );
    Ok(())
}

// Use case: an agent records a Decision and later replays it asking "why did we decide this
// then?" — the basis snapshot must be frozen AT Decision creation, not current post-migration
// state. Mutating a BASED_ON Entity after Decision creation surfaces as drift on replay with
// old_value equal to the basis snapshot, never to current. Defeats the
// read-current-on-both-sides silent-drift hole.
#[test]
fn replay_uses_creation_time_snapshot_not_current() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("decision-replay")),
        "context"
    )
    .id;
    let intention = ok!(
        store.create_intention(intention_input(
            ctx,
            "rule v1: alert on unrecognized signature"
        )),
        "I"
    );
    let e1 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "region-service",
            map(&[("region", json!("region-a"))])
        )),
        "E1"
    );
    let e2 = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Database,
            "events-db",
            map(&[("engine", json!("postgres"))])
        )),
        "E2"
    );
    let decision = ok!(
        store.create_decision(decision_input(
            ctx,
            "enable rule v1 with basis [E1, E2]",
            vec![intention.id],
            vec![e1.id, e2.id],
        )),
        "D basis [E1, E2]"
    );
    let _tx_decision = decision.created_at;
    drop(store);
    let store = reopen(&db_path);
    ok!(
        store.update_entity(
            e1.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[("region", json!("region-b"))])),
                tags: None,
            },
        ),
        "E1 region change after D"
    );
    // Derive the update TxId from the persisted snapshot chain (matches the pattern used by
    // state_at_returns_historical_snapshot_for_each_moment and history_oldest_first_contiguous,
    // which read tx values from entity_history()[i].valid_from). Do NOT read tx_e1_update from
    // the update_entity return value — Entity.created_at means CREATION TIME per the ontology
    // and concepts.md; reading an update tx from that field would either be wrong or would force
    // update_entity() to double-bind that field (breaking the public type's contract).
    let tx_e1_update = ok!(store.entity_history(e1.id), "e1 history")[1].valid_from;
    let replay = ok!(store.replay(decision.id), "replay");
    assert_eq!(replay.decision_id, decision.id);
    assert!(
        replay.inputs_changed,
        "E1 drift must flip inputs_changed=true, got {}",
        replay.inputs_changed
    );
    assert_eq!(
        replay.changed_input_count, 1,
        "exactly one changed input (E1)"
    );
    assert_eq!(
        replay.unchanged_input_count, 1,
        "exactly one unchanged input (E2)"
    );
    assert_eq!(replay.input_diffs.len(), 2);
    let e1_diff = replay
        .input_diffs
        .iter()
        .find(|d| d.entity_id == e1.id)
        .expect("E1 diff");
    assert!(e1_diff.changed, "E1 changed=true");
    assert_eq!(e1_diff.field_changes.len(), 1, "one FieldChange on E1");
    assert_eq!(e1_diff.field_changes[0].field, "region");
    assert_eq!(e1_diff.field_changes[0].old_value, Some(json!("region-a")));
    assert_eq!(e1_diff.field_changes[0].new_value, Some(json!("region-b")));
    assert_eq!(e1_diff.changed_at, Some(tx_e1_update));
    let e2_diff = replay
        .input_diffs
        .iter()
        .find(|d| d.entity_id == e2.id)
        .expect("E2 diff");
    assert!(!e2_diff.changed, "E2 changed=false");
    assert!(e2_diff.field_changes.is_empty());
    assert_eq!(e2_diff.changed_at, None);
    Ok(())
}

// Use case: a notification sink rendering the basis-diff — replay returns inputs_changed=false
// with no phantom changes when no basis Entity has been mutated since Decision creation. Guards
// against delivering "something changed" placeholders a reader silences without reading.
#[test]
fn replay_no_changes() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("no-change-replay")),
        "context"
    )
    .id;
    let intention = ok!(
        store.create_intention(intention_input(ctx, "guard stable inputs")),
        "I"
    );
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Module,
            "stable-module",
            map(&[("api_version", json!("2.0"))])
        )),
        "E"
    );
    let decision = ok!(
        store.create_decision(decision_input(
            ctx,
            "stable decision",
            vec![intention.id],
            vec![e.id],
        )),
        "D"
    );
    drop(store);
    let store = reopen(&db_path);
    let replay = ok!(store.replay(decision.id), "replay no changes");
    assert!(
        !replay.inputs_changed,
        "inputs_changed=false when no basis mutated"
    );
    assert_eq!(replay.changed_input_count, 0);
    assert_eq!(replay.unchanged_input_count, 1);
    assert_eq!(replay.input_diffs.len(), 1, "one input diff (E)");
    let diff = &replay.input_diffs[0];
    assert_eq!(diff.entity_id, e.id);
    assert!(!diff.changed, "E changed=false");
    assert!(diff.field_changes.is_empty());
    assert_eq!(diff.changed_at, None);
    Ok(())
}

// Use case: audit chain of custody on the Intention side — a CLI or MCP client showing a past
// Decision's basis must display the Intention text in force at D.created_at, even if the
// Intention was retired later. replay's cited_intention_snapshots carries the snapshot covering
// D.created_at. Lifecycle follows shipped intention_archive_blocked_by_active_decision
// invariant: archive Decision first, then archive Intention. Defeats pick-oldest, pick-current,
// and archived-intention-shortcut (Err/empty when current status is archived).
#[test]
fn replay_includes_cited_intention_wording_at_decision_time() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("audit-replay")),
        "context"
    )
    .id;
    let intention = ok!(
        store.create_intention(intention_input(
            ctx,
            "policy v0: alert on unrecognized signature, lenient matching"
        )),
        "v0"
    );
    // v1 refinement lands BEFORE D.
    ok!(
        store.update_intention(
            intention.id,
            IntentionPatch {
                description: Some(
                    "policy v1: alert on unrecognized signature, strict matching".to_string()
                ),
                status: None,
                properties: None,
                tags: None,
            },
        ),
        "v1 refinement"
    );
    // Anchor to establish tx_after_v1 < tx_decision.
    let anchor = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "anchor-after-v1",
            map(&[("k", json!("a"))])
        )),
        "anchor"
    );
    let tx_after_v1 = anchor.created_at;
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "signature-detector",
            map(&[("threshold", json!("0.70"))])
        )),
        "E"
    );
    let decision = ok!(
        store.create_decision(decision_input(
            ctx,
            "set detector threshold to 0.70",
            vec![intention.id],
            vec![e.id],
        )),
        "D cites v1"
    );
    let tx_decision = decision.created_at;
    assert!(
        tx_decision > tx_after_v1,
        "D.created_at > tx_after_v1 (D sees v1 wording)"
    );
    // v2 refinement lands AFTER D.
    ok!(
        store.update_intention(
            intention.id,
            IntentionPatch {
                description: Some(
                    "policy v2: alert on unrecognized signature, relaxed matching during window"
                        .to_string()
                ),
                status: None,
                properties: None,
                tags: None,
            },
        ),
        "v2 refinement"
    );
    // Production lifecycle: archive Decision first (retires its SERVES edge as active), then
    // archive Intention — matches shipped intention_archive_blocked_by_active_decision invariant.
    ok!(
        store.archive_decision(decision.id, "policy retirement underway".to_string()),
        "archive D"
    );
    ok!(
        store.archive_intention(intention.id, "policy retired".to_string()),
        "archive I"
    );
    drop(store);
    let store = reopen(&db_path);
    let replay = ok!(
        store.replay(decision.id),
        "replay archived Decision + archived Intention"
    );
    assert_eq!(
        replay.cited_intention_snapshots.len(),
        1,
        "one cited Intention snapshot"
    );
    let snap = &replay.cited_intention_snapshots[0];
    assert_eq!(snap.intention_id, intention.id);
    assert_eq!(
        snap.description, "policy v1: alert on unrecognized signature, strict matching",
        "v1 wording at D.created_at — NOT v0 oldest, NOT v2 current"
    );
    assert!(snap.valid_from <= tx_decision, "valid_from <= tx_decision");
    assert!(
        snap.valid_to.is_none() || snap.valid_to.expect("closed") > tx_decision,
        "half-open [valid_from, valid_to) covers tx_decision"
    );
    Ok(())
}

// Use case: compliance bundle for a Decision that cites both a matching-policy Intention and a
// retention-policy Intention must quote each policy in the wording in force at decision time.
// replay resolves ONE covering snapshot per cited Intention — defeats (a) only-first-cited and
// (b) pick-oldest across multi-Intention replays by seeding each Intention with multiple snapshots
// before AND (for I1) after D.
#[test]
fn replay_multi_cited_intention_mixed_refinement() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("multi-cited")),
        "context"
    )
    .id;
    let i1 = ok!(
        store.create_intention(intention_input(ctx, "matching policy draft")),
        "I1 v0"
    );
    ok!(
        store.update_intention(
            i1.id,
            IntentionPatch {
                description: Some("matching policy v1".to_string()),
                status: None,
                properties: None,
                tags: None,
            },
        ),
        "I1 v1 before D"
    );
    let i2 = ok!(
        store.create_intention(intention_input(ctx, "retention policy draft")),
        "I2 v0"
    );
    ok!(
        store.update_intention(
            i2.id,
            IntentionPatch {
                description: Some("retention policy v1".to_string()),
                status: None,
                properties: None,
                tags: None,
            },
        ),
        "I2 v1 before D"
    );
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "compliance-service",
            BTreeMap::new()
        )),
        "E"
    );
    let decision = ok!(
        store.create_decision(decision_input(
            ctx,
            "multi-Intention compliance decision",
            vec![i1.id, i2.id],
            vec![e.id],
        )),
        "D cites [I1, I2] at v1"
    );
    let tx_decision = decision.created_at;
    drop(store);
    let store = reopen(&db_path);
    ok!(
        store.update_intention(
            i1.id,
            IntentionPatch {
                description: Some("matching policy v2".to_string()),
                status: None,
                properties: None,
                tags: None,
            },
        ),
        "I1 v2 after D"
    );
    // I2 left untouched — its v1 snapshot remains open past tx_decision.
    let replay = ok!(store.replay(decision.id), "replay multi-cited");
    assert_eq!(
        replay.cited_intention_snapshots.len(),
        2,
        "one snapshot per cited Intention"
    );
    let snap_i1 = replay
        .cited_intention_snapshots
        .iter()
        .find(|s| s.intention_id == i1.id)
        .expect("I1 snapshot");
    assert_eq!(
        snap_i1.description, "matching policy v1",
        "I1 covering snapshot at tx_decision is v1, NOT v0 oldest, NOT v2 current"
    );
    let snap_i2 = replay
        .cited_intention_snapshots
        .iter()
        .find(|s| s.intention_id == i2.id)
        .expect("I2 snapshot");
    assert_eq!(
        snap_i2.description, "retention policy v1",
        "I2 covering snapshot at tx_decision is v1, NOT v0 oldest"
    );
    for snap in &replay.cited_intention_snapshots {
        assert!(
            snap.valid_from <= tx_decision,
            "valid_from <= tx_decision for {:?}",
            snap.intention_id
        );
        assert!(
            snap.valid_to.is_none() || snap.valid_to.expect("closed") > tx_decision,
            "half-open interval covers tx_decision for {:?}",
            snap.intention_id
        );
    }
    Ok(())
}

// Use case: a library consumer replays a past Decision whose basis Entity was updated and then
// archived during a migration — "why did we decide this then?" must NOT return entity-not-found.
// replay surfaces basis frozen pre-update vs current archived state; state_at + entity_history
// still resolve; archive is a status flip, not a deletion of history.
#[test]
fn archived_entity_still_replayable() -> Result<(), CgError> {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("archived-replay")),
        "context"
    )
    .id;
    let intention = ok!(
        store.create_intention(intention_input(
            ctx,
            "rule v1: alert based on detector basis"
        )),
        "I"
    );
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "detector-service",
            map(&[("region", json!("us-east-1")), ("tier", json!("standard"))])
        )),
        "E basis"
    );
    let decision = ok!(
        store.create_decision(decision_input(
            ctx,
            "enable rule v1 with detector basis",
            vec![intention.id],
            vec![e.id],
        )),
        "D"
    );
    let tx_decision = decision.created_at;
    ok!(
        store.update_entity(
            e.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("region", json!("us-east-1-migrated")),
                    ("tier", json!("premium"))
                ])),
                tags: None,
            },
        ),
        "E update — properties change after basis frozen"
    );
    ok!(store.archive_entity(e.id), "archive E (status flip)");
    drop(store);
    let store = reopen(&db_path);
    let s_basis = ok!(store.state_at(e.id, tx_decision), "state_at tx_decision")
        .expect("pre-archive snapshot at tx_decision");
    assert_eq!(
        s_basis.properties["region"],
        json!("us-east-1"),
        "pre-update, pre-archive region"
    );
    assert_eq!(
        s_basis.properties["tier"],
        json!("standard"),
        "pre-update, pre-archive tier"
    );
    let history = ok!(store.entity_history(e.id), "entity_history after archive");
    assert!(
        history.len() >= 2,
        "pre-update + post-update snapshots both persist (len={})",
        history.len()
    );
    let replay = ok!(store.replay(decision.id), "replay with archived basis");
    assert_eq!(replay.input_diffs.len(), 1);
    let diff = &replay.input_diffs[0];
    assert_eq!(diff.entity_id, e.id);
    assert!(
        diff.changed,
        "basis differs from current (archived post-update state)"
    );
    assert!(
        diff.field_changes.iter().any(|fc| fc.field == "region"
            && fc.old_value == Some(json!("us-east-1"))
            && fc.new_value == Some(json!("us-east-1-migrated"))),
        "region FieldChange: old=pre-update basis, new=post-update archived, got: {:?}",
        diff.field_changes
    );
    assert!(
        diff.field_changes.iter().any(|fc| fc.field == "tier"
            && fc.old_value == Some(json!("standard"))
            && fc.new_value == Some(json!("premium"))),
        "tier FieldChange: old=pre-update basis, new=post-update archived, got: {:?}",
        diff.field_changes
    );
    assert_eq!(
        replay.cited_intention_snapshots.len(),
        1,
        "cited Intention snapshot present"
    );
    Ok(())
}

// Use case: BYOC harness invariance — engineer records Decision via Claude Code and reads it via
// Codex CLI six weeks later; temporal reads return the same answer regardless of which agent
// harness made the read. Store handles on the same on-disk path (sequential OR concurrent) return
// structurally-equal results. agent_id round-trips on the Decision row as attribution metadata.
// Temporal call signatures take only TxId + entity/decision id — no reader-identity parameter
// (compile-level invariant).
#[test]
fn temporal_reads_are_agent_id_agnostic() -> Result<(), CgError> {
    let (tempdir, db_path, store) = fixture();
    let ctx = ok!(store.create_context(context_input("byoc")), "context").id;
    let e = ok!(
        store.create_entity(entity_input(
            ctx,
            EntityType::Service,
            "budget-service",
            map(&[("region", json!("us-east-1"))])
        )),
        "E"
    );
    let intention = ok!(
        store.create_intention(intention_input(ctx, "latency budget policy")),
        "I"
    );
    let d_claude = ok!(
        store.create_decision(CreateDecision {
            decision_type: "budget".to_string(),
            description: "180ms budget".to_string(),
            reasoning: reasoning("d_claude"),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![e.id],
            basis_fields: None,
            based_on_snapshots: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: Some("claude-code".to_string()),
        }),
        "D_claude"
    );
    ok!(
        store.update_entity(
            e.id,
            context_graph::EntityPatch {
                name: None,
                properties: Some(map(&[("region", json!("eu-west-1"))])),
                tags: None,
            },
        ),
        "E region update between decisions"
    );
    let d_codex = ok!(
        store.create_decision(CreateDecision {
            decision_type: "budget".to_string(),
            description: "250ms after region move".to_string(),
            reasoning: reasoning("d_codex"),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![e.id],
            basis_fields: None,
            based_on_snapshots: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: Some("codex-cli".to_string()),
        }),
        "D_codex"
    );
    drop(store);

    // Sequential pass: open, read, drop, reopen, read, compare.
    let store_a = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open a");
    let a_state_claude = ok!(
        store_a.state_at(e.id, d_claude.created_at),
        "state_at D_claude tx (a)"
    );
    let a_state_codex = ok!(
        store_a.state_at(e.id, d_codex.created_at),
        "state_at D_codex tx (a)"
    );
    let a_history = ok!(store_a.entity_history(e.id), "entity_history (a)");
    let a_replay_claude = ok!(store_a.replay(d_claude.id), "replay D_claude (a)");
    let a_replay_codex = ok!(store_a.replay(d_codex.id), "replay D_codex (a)");
    drop(store_a);

    let store_b = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open b");
    let b_state_claude = ok!(
        store_b.state_at(e.id, d_claude.created_at),
        "state_at D_claude tx (b)"
    );
    let b_state_codex = ok!(
        store_b.state_at(e.id, d_codex.created_at),
        "state_at D_codex tx (b)"
    );
    let b_history = ok!(store_b.entity_history(e.id), "entity_history (b)");
    let b_replay_claude = ok!(store_b.replay(d_claude.id), "replay D_claude (b)");
    let b_replay_codex = ok!(store_b.replay(d_codex.id), "replay D_codex (b)");

    assert_eq!(
        a_state_claude, b_state_claude,
        "state_at D_claude invariant"
    );
    assert_eq!(a_state_codex, b_state_codex, "state_at D_codex invariant");
    assert_eq!(a_history, b_history, "entity_history invariant");
    assert_eq!(
        a_replay_claude, b_replay_claude,
        "replay D_claude invariant"
    );
    assert_eq!(a_replay_codex, b_replay_codex, "replay D_codex invariant");

    // Concurrent pass: two Store handles on the same on-disk path, both held simultaneously.
    let store_c = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open c");
    let store_d = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open d");
    let pc_c = ok!(store_c.replay(d_claude.id), "replay D_claude (c)");
    let pc_d = ok!(store_d.replay(d_claude.id), "replay D_claude (d)");
    assert_eq!(pc_c, pc_d, "concurrent-handle replay invariant");
    drop(store_c);
    drop(store_d);

    // Attribution round-trip — agent_id metadata persisted on Decision rows.
    let fetched_claude =
        ok!(store_b.get_decision(d_claude.id), "get D_claude").expect("D_claude decision");
    let fetched_codex =
        ok!(store_b.get_decision(d_codex.id), "get D_codex").expect("D_codex decision");
    assert_eq!(
        fetched_claude.agent_id,
        Some("claude-code".to_string()),
        "claude-code agent_id round-trip"
    );
    assert_eq!(
        fetched_codex.agent_id,
        Some("codex-cli".to_string()),
        "codex-cli agent_id round-trip"
    );

    drop(tempdir);
    Ok(())
}
