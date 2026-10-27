use crate::capture_sync_common::{
    drain_ticketed_pull, drain_ticketed_push, ticketed_memory_fixture,
};
use chrono::{DateTime, Utc};
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, EntityId,
    EntitySnapshotId, EntityType, EvidenceId, EvidenceRef, IntentionId, IntentionOrigin,
    IntentionPatch, IntentionSnapshotId, IntentionStatus, ObservationId, RecordObservation, Store,
};
use contextdb_core::{Error as DbError, TxId, Value as DbValue};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use uuid::Uuid;

fn observed_at(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("fixed observation timestamp")
        .with_timezone(&Utc)
}

fn append_successor_snapshot(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    author: &str,
    path: &str,
    at: DateTime<Utc>,
) -> EntitySnapshotId {
    let recorded = store
        .record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id,
            context_id,
            observation_type: "source_change".to_string(),
            source: author.to_string(),
            observed_at: at,
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                source_ref: format!("s3:session/{author}"),
                ..Default::default()
            }],
            observed_properties: BTreeMap::from([(
                "path".to_string(),
                serde_json::json!("src/auth.rs"),
            )]),
            state_delta: BTreeMap::from([("path".to_string(), Some(serde_json::json!(path)))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("append independently identified successor snapshot");
    recorded
        .snapshot_id
        .expect("state change must create a successor snapshot")
}

fn snapshot_links(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
) -> BTreeSet<(Uuid, Uuid)> {
    store
        .sync_database()
        .scan("entity_snapshot_links", store.sync_database().snapshot())
        .expect("read persisted snapshot links")
        .into_iter()
        .filter(|row| {
            row.values.get("context_id") == Some(&DbValue::Uuid(context_id.as_uuid()))
                && row.values.get("entity_id") == Some(&DbValue::Uuid(entity_id.as_uuid()))
        })
        .map(|row| {
            let predecessor = match row.values.get("predecessor_snapshot_id") {
                Some(DbValue::Uuid(id)) => *id,
                value => panic!("snapshot link must retain its predecessor id, got {value:?}"),
            };
            let successor = match row.values.get("successor_snapshot_id") {
                Some(DbValue::Uuid(id)) => *id,
                value => panic!("snapshot link must retain its successor id, got {value:?}"),
            };
            (predecessor, successor)
        })
        .collect()
}

fn assert_snapshot_links(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    predecessor: EntitySnapshotId,
    successors: [EntitySnapshotId; 2],
) {
    assert_eq!(
        snapshot_links(store, context_id, entity_id),
        successors
            .into_iter()
            .map(|successor| (predecessor.as_uuid(), successor.as_uuid()))
            .collect(),
        "both independently identified successors retain their exact predecessor links"
    );
}

fn assert_duplicate_link_is_refused(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    predecessor: EntitySnapshotId,
    successor: EntitySnapshotId,
) {
    let duplicate = store.sync_database().execute(
        "INSERT INTO entity_snapshot_links (id, context_id, entity_id, predecessor_snapshot_id, successor_snapshot_id, created_at) VALUES ($id, $context_id, $entity_id, $predecessor_snapshot_id, $successor_snapshot_id, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(Uuid::now_v7())),
            ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
            ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
            (
                "predecessor_snapshot_id".to_string(),
                DbValue::Uuid(predecessor.as_uuid()),
            ),
            (
                "successor_snapshot_id".to_string(),
                DbValue::Uuid(successor.as_uuid()),
            ),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    );
    assert!(
        matches!(duplicate, Err(DbError::UniqueViolation { .. })),
        "an exact predecessor-to-successor link must fail with its unique identity error, got {duplicate:?}"
    );
}

fn assert_intention_fork_and_exact_replay(
    store: &Store,
    intention_id: IntentionId,
    decision_id: context_graph::DecisionId,
    predecessor: IntentionSnapshotId,
    branch_a: IntentionSnapshotId,
    branch_b: IntentionSnapshotId,
) {
    let history = store.history(intention_id).expect("read synced history");
    assert_eq!(
        history.len(),
        3,
        "one predecessor and two successors survive"
    );
    assert!(
        history.iter().all(|snapshot| snapshot.valid_to.is_none()),
        "immutable snapshots are never closed by rewriting valid_to"
    );
    let links = store
        .intention_snapshot_links(intention_id)
        .expect("read synced intention links");
    assert_eq!(
        links
            .iter()
            .map(|link| (link.predecessor_snapshot_id, link.successor_snapshot_id))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([(predecessor, branch_a), (predecessor, branch_b)]),
        "both public-API updates retain the shared predecessor"
    );

    let at = history
        .iter()
        .filter(|snapshot| snapshot.id == branch_a || snapshot.id == branch_b)
        .map(|snapshot| snapshot.valid_from)
        .max()
        .expect("both fork heads exist");
    let ambiguity = store.intention_at(intention_id, at);
    assert!(
        matches!(
            ambiguity,
            Err(CgError::AmbiguousIntentionHistory {
                intention_id: ambiguous_id,
                ref branch_head_ids,
                ..
            }) if ambiguous_id == intention_id
                && branch_head_ids == &{
                    let mut expected = vec![branch_a, branch_b];
                    expected.sort_by_key(|id| id.as_uuid());
                    expected
                }
        ),
        "a branchless temporal read must return the typed ambiguity with both heads, got {ambiguity:?}"
    );
    for branch in [branch_a, branch_b] {
        assert_eq!(
            store
                .intention_at_on_branch(intention_id, at, branch)
                .expect("read explicitly selected branch")
                .expect("selected branch has a covering snapshot")
                .id,
            branch,
            "an explicit branch read returns that exact immutable head"
        );
    }

    let replay = store
        .replay(decision_id)
        .expect("replay decision after intention fork convergence");
    assert_eq!(
        replay
            .cited_intention_snapshots
            .iter()
            .map(|snapshot| snapshot.id)
            .collect::<Vec<_>>(),
        vec![branch_a],
        "Decision replay returns the exact intention wording cited at creation, never the converged current pointer"
    );
}

fn assert_immutable_row_refuses_update_and_delete(store: &Store, table: &str) {
    let db = store.sync_database();
    let row = db
        .scan(table, db.snapshot())
        .unwrap_or_else(|error| panic!("scan {table}: {error:?}"))
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("{table} must contain a row for the immutability probe"));
    let id = match row.values.get("id") {
        Some(DbValue::Uuid(id)) => *id,
        value => panic!("{table}.id must be UUID, got {value:?}"),
    };
    let update = db.execute(
        &format!("UPDATE {table} SET id = $replacement_id WHERE id = $id"),
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(id)),
            ("replacement_id".to_string(), DbValue::Uuid(Uuid::now_v7())),
        ]),
    );
    assert!(
        matches!(update, Err(DbError::ImmutableTable(ref actual)) if actual == table),
        "raw UPDATE of {table} must fail at the immutable-table boundary, got {update:?}"
    );
    let delete = db.execute(
        &format!("DELETE FROM {table} WHERE id = $id"),
        &HashMap::from([("id".to_string(), DbValue::Uuid(id))]),
    );
    assert!(
        matches!(delete, Err(DbError::ImmutableTable(ref actual)) if actual == table),
        "raw DELETE of {table} must fail at the immutable-table boundary, got {delete:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn successor_snapshot_branches_survive_sync_restart_and_restore() {
    let fixture = ticketed_memory_fixture("successor_snapshot_branches").await;
    let edge_a = fixture.open_edge("snapshot-edge-a");
    let edge_b = fixture.open_edge("snapshot-edge-b");
    let context = edge_a
        .store
        .create_context(CreateContext {
            name: "snapshot provenance".to_string(),
            ..Default::default()
        })
        .expect("create snapshot context");
    let entity = edge_a
        .store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: "authentication module".to_string(),
            properties: BTreeMap::from([("path".to_string(), serde_json::json!("src/auth.rs"))]),
            context_id: context.id,
            ..Default::default()
        })
        .expect("create predecessor entity");
    let predecessor = edge_a
        .store
        .entity_history(entity.id)
        .expect("read predecessor snapshot")
        .into_iter()
        .next()
        .expect("entity creation creates its first snapshot")
        .id;

    drain_ticketed_push(&edge_a.client).await;
    drain_ticketed_pull(&edge_b.client).await;

    let successor_a = append_successor_snapshot(
        &edge_a.store,
        context.id,
        entity.id,
        "agent-a",
        "src/authentication.rs",
        observed_at("2026-01-15T10:00:00Z"),
    );
    let successor_b = append_successor_snapshot(
        &edge_b.store,
        context.id,
        entity.id,
        "agent-b",
        "src/identity.rs",
        observed_at("2026-01-15T10:05:00Z"),
    );
    assert_ne!(
        successor_a, successor_b,
        "the successors are independently identified"
    );

    drain_ticketed_push(&edge_a.client).await;
    drain_ticketed_push(&edge_b.client).await;
    drain_ticketed_pull(&edge_a.client).await;
    drain_ticketed_pull(&edge_b.client).await;

    for store in [fixture.hub_store(), &edge_a.store, &edge_b.store] {
        assert_snapshot_links(
            store,
            context.id,
            entity.id,
            predecessor,
            [successor_a, successor_b],
        );
    }
    let graph = fixture
        .hub_store()
        .entity_history_graph(entity.id)
        .expect("read immutable history graph");
    assert_eq!(
        graph.branch_heads.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([successor_a, successor_b]),
        "the transport/current pointer is not allowed to choose a semantic branch"
    );
    let at_end = TxId(u64::MAX);
    let ambiguous = fixture.hub_store().state_at(entity.id, at_end);
    assert!(
        matches!(
            ambiguous,
            Err(CgError::AmbiguousTemporalState {
                entity_id,
                at,
                ref branch_heads,
            }) if entity_id == entity.id
                && at == at_end
                && branch_heads.iter().copied().collect::<BTreeSet<_>>()
                    == BTreeSet::from([successor_a, successor_b])
        ),
        "a fork must require an explicit semantic branch, got {ambiguous:?}"
    );
    let branch_a = fixture
        .hub_store()
        .state_at_on_branch(entity.id, at_end, successor_a)
        .expect("resolve explicit branch a")
        .expect("branch a remains live");
    assert_eq!(
        branch_a.properties["path"],
        serde_json::json!("src/authentication.rs")
    );
    let branch_b = fixture
        .hub_store()
        .state_at_on_branch(entity.id, at_end, successor_b)
        .expect("resolve explicit branch b")
        .expect("branch b remains live");
    assert_eq!(
        branch_b.properties["path"],
        serde_json::json!("src/identity.rs")
    );
    assert_duplicate_link_is_refused(
        fixture.hub_store(),
        context.id,
        entity.id,
        predecessor,
        successor_a,
    );

    drop(edge_a);
    let restarted_a = fixture.open_edge("snapshot-edge-a");
    assert_snapshot_links(
        &restarted_a.store,
        context.id,
        entity.id,
        predecessor,
        [successor_a, successor_b],
    );

    let restored = fixture.open_edge("snapshot-wiped-restore");
    drain_ticketed_pull(&restored.client).await;
    assert_snapshot_links(
        &restored.store,
        context.id,
        entity.id,
        predecessor,
        [successor_a, successor_b],
    );
    fixture.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn intention_fork_and_exact_decision_citation_survive_ticketed_sync_restart_and_restore() {
    let fixture = ticketed_memory_fixture("intention_snapshot_branches").await;
    let edge_a = fixture.open_edge("intention-edge-a");
    let edge_b = fixture.open_edge("intention-edge-b");
    let context = edge_a
        .store
        .create_context(CreateContext {
            name: "intention provenance".to_string(),
            ..Default::default()
        })
        .expect("create intention context");
    let intention = edge_a
        .store
        .create_intention(CreateIntention {
            description: "protect the loading dock".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            ..Default::default()
        })
        .expect("create shared predecessor intention");
    let predecessor = edge_a
        .store
        .history(intention.id)
        .expect("read initial intention history")
        .into_iter()
        .next()
        .expect("intention creation writes its initial snapshot")
        .id;

    drain_ticketed_push(&edge_a.client).await;
    drain_ticketed_pull(&edge_b.client).await;

    let branch_a = edge_a
        .store
        .update_intention_on_branch(
            intention.id,
            predecessor,
            IntentionPatch {
                description: Some("protect the loading dock after hours".to_string()),
                ..Default::default()
            },
        )
        .expect("edge A appends its successor through the public API");
    assert_eq!(branch_a.description, "protect the loading dock after hours");
    let decision = edge_a
        .store
        .create_decision(CreateDecision {
            decision_type: "security".to_string(),
            description: "install a loading-dock sensor".to_string(),
            intention_ids: vec![intention.id],
            context_id: context.id,
            ..Default::default()
        })
        .expect("Decision cites edge A's exact successor");

    let branch_b = edge_b
        .store
        .update_intention_on_branch(
            intention.id,
            predecessor,
            IntentionPatch {
                description: Some("protect the loading dock whenever it is unattended".to_string()),
                ..Default::default()
            },
        )
        .expect("edge B appends its successor through the public API");
    assert_eq!(
        branch_b.description,
        "protect the loading dock whenever it is unattended"
    );
    assert_ne!(
        branch_a.id, branch_b.id,
        "the offline successors have independent immutable identities"
    );

    drain_ticketed_push(&edge_a.client).await;
    drain_ticketed_push(&edge_b.client).await;
    drain_ticketed_pull(&edge_a.client).await;
    drain_ticketed_pull(&edge_b.client).await;

    for store in [fixture.hub_store(), &edge_a.store, &edge_b.store] {
        assert_intention_fork_and_exact_replay(
            store,
            intention.id,
            decision.id,
            predecessor,
            branch_a.id,
            branch_b.id,
        );
    }
    for table in [
        "intention_snapshots",
        "intention_snapshot_links",
        "decision_intention_citations",
    ] {
        assert_immutable_row_refuses_update_and_delete(fixture.hub_store(), table);
    }

    drop(edge_a);
    let restarted_a = fixture.open_edge("intention-edge-a");
    assert_intention_fork_and_exact_replay(
        &restarted_a.store,
        intention.id,
        decision.id,
        predecessor,
        branch_a.id,
        branch_b.id,
    );

    let restored = fixture.open_edge("intention-wiped-restore");
    drain_ticketed_pull(&restored.client).await;
    assert_intention_fork_and_exact_replay(
        &restored.store,
        intention.id,
        decision.id,
        predecessor,
        branch_a.id,
        branch_b.id,
    );
    fixture.shutdown().await;
}
