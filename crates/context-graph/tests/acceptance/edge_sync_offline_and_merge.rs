use crate::capture_sync_common::{
    assert_decision_status_active, assert_real_decision_content, assert_real_evidence_content,
    assert_real_intention_content, assert_real_observation_content, assert_real_outcome_state,
    assert_sync_contract_has_real_table_policies, bidirectional_sync, drain_ticketed_pull,
    drain_ticketed_push, push_pull, real_sync_fixture, record_contended_outcomes_concurrently,
    seed_decision_graph, seed_two_edges_concurrently, self_echo_sync_fixture,
    ticketed_memory_fixture, touch_same_decision_graph_concurrently,
    unavailable_authenticated_client, within,
};
use chrono::{DateTime, Utc};
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DecisionStatus, EntityId, EntitySnapshotId, EntityType, EvidenceId, EvidenceKind, EvidenceRef,
    IntentionId, IntentionOrigin, IntentionStatus, ObservationId, Outcome, RecordObservation,
    Store,
};
use contextdb_core::Value as DbValue;
use std::collections::{BTreeMap, BTreeSet};

struct IndependentHistory {
    author: String,
    decision_id: DecisionId,
    decision_description: String,
    observation_id: ObservationId,
    evidence_id: EvidenceId,
    evidence_source: String,
    snapshot_id: EntitySnapshotId,
    revised_path: String,
    observed_at: DateTime<Utc>,
    outcome: Outcome,
}

struct IndependentHistoryInput<'a> {
    context_id: ContextId,
    intention_id: IntentionId,
    shared_entity_id: EntityId,
    author: &'a str,
    revised_path: &'a str,
    observed_at: DateTime<Utc>,
    outcome_success: bool,
}

fn record_independent_history(
    store: &Store,
    input: IndependentHistoryInput<'_>,
) -> IndependentHistory {
    let IndependentHistoryInput {
        context_id,
        intention_id,
        shared_entity_id,
        author,
        revised_path,
        observed_at,
        outcome_success,
    } = input;
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let evidence_source = format!("vigil-edge:session/{author}");
    let observed = store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id: shared_entity_id,
            context_id,
            observation_type: "agent_session_fact".to_string(),
            source: author.to_string(),
            observed_at,
            evidence: vec![EvidenceRef {
                id: evidence_id,
                source_ref: evidence_source.clone(),
                ..Default::default()
            }],
            observed_properties: BTreeMap::from([(
                "path".to_string(),
                serde_json::json!("src/auth.rs"),
            )]),
            state_delta: BTreeMap::from([(
                "path".to_string(),
                Some(serde_json::json!(revised_path)),
            )]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record independently identified observation and evidence");
    assert!(
        observed.invalidations.is_empty(),
        "capturing evidence before a decision exists must not invent a judgement"
    );
    assert!(
        observed.snapshot_id.is_some(),
        "the observation must append the evidence-backed snapshot"
    );
    let snapshot_id = observed
        .snapshot_id
        .expect("the independent observation creates its basis snapshot");
    let decision_description = format!("{author} chose {revised_path}");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "authentication_implementation".to_string(),
            description: decision_description.clone(),
            intention_ids: vec![intention_id],
            based_on_entity_ids: vec![shared_entity_id],
            context_id,
            agent_id: Some(author.to_string()),
            ..Default::default()
        })
        .expect("create independently identified decision");
    assert_eq!(
        decision.based_on_snapshots,
        vec![snapshot_id],
        "each agent's decision must name that agent's own evidence-backed basis"
    );
    let outcome_notes = format!("{author} independently evaluated its implementation");
    let outcome = store
        .record_outcome(decision.id, outcome_success, Some(outcome_notes.clone()))
        .expect("record the agent's independently identified outcome");
    assert_eq!(outcome.notes.as_deref(), Some(outcome_notes.as_str()));
    IndependentHistory {
        author: author.to_string(),
        decision_id: decision.id,
        decision_description,
        observation_id,
        evidence_id,
        evidence_source,
        snapshot_id,
        revised_path: revised_path.to_string(),
        observed_at,
        outcome,
    }
}

fn assert_independent_histories(
    store: &Store,
    context_id: ContextId,
    intention_id: IntentionId,
    shared_entity_id: EntityId,
    initial_snapshot_id: EntitySnapshotId,
    histories: &[&IndependentHistory],
) {
    let expected_decisions = histories
        .iter()
        .map(|history| history.decision_id)
        .collect::<BTreeSet<_>>();
    let actual_decisions = store
        .by_intention(intention_id, Default::default())
        .expect("read decisions serving the shared intention")
        .into_iter()
        .inspect(|decision| {
            assert_eq!(
                decision.status,
                DecisionStatus::Active,
                "sync order must not pick a current decision"
            )
        })
        .map(|decision| decision.id)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_decisions, expected_decisions,
        "both decisions in tension remain independently addressable"
    );

    for history in histories {
        let decision = store
            .get_decision(history.decision_id)
            .expect("read decision")
            .expect("decision survives");
        assert_eq!(decision.description, history.decision_description);
        assert_eq!(decision.intention_ids, vec![intention_id]);
        assert_eq!(decision.status, DecisionStatus::Active);
        assert_eq!(decision.agent_id.as_deref(), Some(history.author.as_str()));
        assert_eq!(
            decision.based_on_snapshots,
            vec![history.snapshot_id],
            "each independently identified decision retains its own basis"
        );
        let outcome = decision
            .outcome
            .expect("each independently identified decision retains its outcome");
        assert_eq!(outcome.id, history.outcome.id);
        assert_eq!(outcome.decision_id, history.decision_id);
        assert_eq!(outcome.success, history.outcome.success);
        assert_eq!(outcome.notes, history.outcome.notes);
        assert_real_evidence_content(
            store,
            history.evidence_id,
            history.observation_id,
            EvidenceKind::Text,
            &history.evidence_source,
        );
        let observation = store
            .get_observation(history.observation_id)
            .expect("observation survives");
        assert_eq!(observation.entity_id, shared_entity_id);
        assert_eq!(observation.source, history.author);
        assert_eq!(
            observation.observed_at, history.observed_at,
            "real-world observation time survives independently of hub arrival order"
        );
        assert_eq!(
            observation
                .evidence
                .iter()
                .map(|row| row.id)
                .collect::<Vec<_>>(),
            vec![history.evidence_id]
        );
    }

    let entity_history = store
        .entity_history(shared_entity_id)
        .expect("complete shared-basis history survives");
    let expected_snapshot_ids = std::iter::once(initial_snapshot_id)
        .chain(histories.iter().map(|history| history.snapshot_id))
        .collect::<BTreeSet<_>>();
    let actual_snapshot_ids = entity_history
        .iter()
        .map(|snapshot| snapshot.id)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_snapshot_ids, expected_snapshot_ids,
        "the common starting point and both independent evidence branches must survive"
    );
    assert_eq!(
        entity_history
            .iter()
            .find(|snapshot| snapshot.id == initial_snapshot_id)
            .and_then(|snapshot| snapshot.properties.get("path")),
        Some(&serde_json::json!("src/auth.rs"))
    );
    for history in histories {
        let snapshot = entity_history
            .iter()
            .find(|snapshot| snapshot.id == history.snapshot_id)
            .expect("agent's evidence-backed snapshot survives");
        assert_eq!(
            snapshot.properties.get("path"),
            Some(&serde_json::json!(history.revised_path))
        );
        assert_eq!(
            snapshot.evidenced_by,
            vec![history.observation_id],
            "the branch remains attributable to the exact agent observation"
        );
    }

    assert_exact_decision_tension(store, context_id, intention_id, histories);
    assert!(
        store
            .list_invalidations(context_id, None, None)
            .expect("read invalidations")
            .is_empty(),
        "transport convergence must not invent semantic invalidation"
    );
}

fn assert_exact_decision_tension(
    store: &Store,
    context_id: ContextId,
    intention_id: IntentionId,
    histories: &[&IndependentHistory],
) {
    let expected_pair = histories
        .iter()
        .map(|history| history.decision_id.as_uuid())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        expected_pair.len(),
        2,
        "the proof requires two independently identified decisions"
    );
    let db = store.sync_database();
    let rows = db
        .scan("conflicts", db.snapshot())
        .expect("scan explicit decision tensions");
    let matching = rows
        .iter()
        .filter(|row| row.values.get("context_id") == Some(&DbValue::Uuid(context_id.as_uuid())))
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "the two opposite outcomes on one shared basis produce one explicit tension"
    );
    let row = matching[0];
    let actual_pair = ["older_decision_id", "newer_decision_id"]
        .into_iter()
        .map(|column| match row.values.get(column) {
            Some(DbValue::Uuid(id)) => *id,
            other => panic!("conflict {column} must be an exact decision id, got {other:?}"),
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_pair, expected_pair,
        "arrival order must not substitute a different decision pair"
    );
    let threatened = match row.values.get("threatened_intention_ids") {
        Some(DbValue::Json(value)) => serde_json::from_value::<Vec<IntentionId>>(value.clone())
            .expect("decode exact threatened intention ids"),
        other => panic!("conflict intentions must be persisted JSON, got {other:?}"),
    };
    assert_eq!(threatened, vec![intention_id]);
    assert_eq!(
        row.values.get("intention_in_tension"),
        Some(&DbValue::Bool(true)),
        "opposite outcomes for decisions serving the same intention must expose tension"
    );
}

fn fixed_observed_at(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("fixed observation timestamp")
        .with_timezone(&Utc)
}

#[tokio::test(flavor = "multi_thread")]
async fn independent_memory_histories_survive_convergence_restart_and_restore() {
    let fixture = ticketed_memory_fixture("independent_memory_histories_survive_convergence").await;
    let edge_a = fixture.open_edge("edge-a");
    let edge_b = fixture.open_edge("edge-b");
    let context = edge_a
        .store
        .create_context(CreateContext {
            name: "shared-agent-memory".to_string(),
            ..Default::default()
        })
        .expect("create shared context");
    let intention = edge_a
        .store
        .create_intention(CreateIntention {
            description: "keep the authentication implementation correct".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            ..Default::default()
        })
        .expect("create shared intention");
    let entity = edge_a
        .store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: "shared authentication module".to_string(),
            properties: BTreeMap::from([("path".to_string(), serde_json::json!("src/auth.rs"))]),
            context_id: context.id,
            ..Default::default()
        })
        .expect("create the shared decision basis");
    let initial_snapshot_id = edge_a
        .store
        .entity_history(entity.id)
        .expect("read initial shared-basis snapshot")
        .into_iter()
        .next()
        .expect("entity creation must persist its initial snapshot")
        .id;
    drain_ticketed_push(&edge_a.client).await;
    drain_ticketed_pull(&edge_b.client).await;

    let history_a = record_independent_history(
        &edge_a.store,
        IndependentHistoryInput {
            context_id: context.id,
            intention_id: intention.id,
            shared_entity_id: entity.id,
            author: "agent-a",
            revised_path: "src/authentication.rs",
            observed_at: fixed_observed_at("2026-01-15T10:00:00Z"),
            outcome_success: true,
        },
    );
    let history_b = record_independent_history(
        &edge_b.store,
        IndependentHistoryInput {
            context_id: context.id,
            intention_id: intention.id,
            shared_entity_id: entity.id,
            author: "agent-b",
            revised_path: "src/identity.rs",
            observed_at: fixed_observed_at("2026-01-15T10:05:00Z"),
            outcome_success: false,
        },
    );
    assert_ne!(history_a.decision_id, history_b.decision_id);
    assert_ne!(history_a.observation_id, history_b.observation_id);
    assert_ne!(history_a.evidence_id, history_b.evidence_id);

    // Arrival is deliberately ordered. Both histories must survive; transport
    // position is not semantic time and cannot choose which decision is current.
    drain_ticketed_push(&edge_a.client).await;
    drain_ticketed_push(&edge_b.client).await;
    drain_ticketed_pull(&edge_a.client).await;
    drain_ticketed_pull(&edge_b.client).await;
    for store in [fixture.hub_store(), &edge_a.store, &edge_b.store] {
        assert_independent_histories(
            store,
            context.id,
            intention.id,
            entity.id,
            initial_snapshot_id,
            &[&history_a, &history_b],
        );
    }

    drop(edge_a);
    drop(edge_b);
    let restarted_a = fixture.open_edge("edge-a");
    let restarted_b = fixture.open_edge("edge-b");
    assert_independent_histories(
        &restarted_a.store,
        context.id,
        intention.id,
        entity.id,
        initial_snapshot_id,
        &[&history_a, &history_b],
    );
    assert_independent_histories(
        &restarted_b.store,
        context.id,
        intention.id,
        entity.id,
        initial_snapshot_id,
        &[&history_a, &history_b],
    );

    let restored = fixture.open_edge("wiped-edge-restore");
    drain_ticketed_pull(&restored.client).await;
    assert_independent_histories(
        &restored.store,
        context.id,
        intention.id,
        entity.id,
        initial_snapshot_id,
        &[&history_a, &history_b],
    );
    fixture.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn capture_succeeds_offline_and_syncs_on_reconnect() {
    let fixture = real_sync_fixture("capture_succeeds_offline_and_syncs_on_reconnect").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "offline");
    let down_client = unavailable_authenticated_client(
        fixture.edge1.store.sync_database(),
        &fixture.tenant_id,
        &fixture.remote_url,
        fixture.edge1.db_path(),
    )
    .await;
    let push_while_down = within(down_client.push()).await;

    assert!(
        fixture
            .edge1_client
            .has_pending_push_changes()
            .expect("pending check")
    );
    assert!(push_while_down.is_err());
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    assert_real_decision_content(
        &fixture.edge2.store,
        seeded.decision.id,
        &seeded.decision.description,
    );
    assert!(
        !fixture
            .edge1_client
            .has_pending_push_changes()
            .expect("pending drained")
    );
    assert_sync_contract_has_real_table_policies();
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_distinct_inserts_on_two_edges_all_survive_after_sync() {
    let fixture = real_sync_fixture("concurrent_distinct_inserts_all_survive").await;
    let (edge1_row, edge2_row) =
        seed_two_edges_concurrently(&fixture, "edge1 distinct", "edge2 distinct").await;
    bidirectional_sync(&fixture.edge1_client, &fixture.edge2_client).await;

    assert_real_decision_content(
        &fixture.edge1.store,
        edge1_row.decision.id,
        &edge1_row.decision.description,
    );
    assert_real_decision_content(
        &fixture.edge1.store,
        edge2_row.decision.id,
        &edge2_row.decision.description,
    );
    assert_real_decision_content(
        &fixture.edge2.store,
        edge2_row.decision.id,
        &edge2_row.decision.description,
    );
    assert_real_decision_content(
        &fixture.edge2.store,
        edge1_row.decision.id,
        &edge1_row.decision.description,
    );
    assert_sync_contract_has_real_table_policies();
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_mutable_status_row_converges_deterministically() {
    let fixture = real_sync_fixture("concurrent_mutable_status_row_converges").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "mutable status");
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    assert_real_decision_content(
        &fixture.edge2.store,
        seeded.decision.id,
        &seeded.decision.description,
    );
    let contended = record_contended_outcomes_concurrently(&fixture, &seeded).await;
    bidirectional_sync(&fixture.edge1_client, &fixture.edge2_client).await;

    assert_real_outcome_state(
        &fixture.edge2.store,
        seeded.decision.id,
        &contended.server_winner,
        1,
    );
    assert_real_outcome_state(
        &fixture.edge1.store,
        seeded.decision.id,
        &contended.server_winner,
        1,
    );
    assert_real_decision_content(
        &fixture.edge1.store,
        contended.surrounding_edge2.decision.id,
        &contended.surrounding_edge2.decision.description,
    );
    assert_real_decision_content(
        &fixture.edge2.store,
        contended.surrounding_edge1.decision.id,
        &contended.surrounding_edge1.decision.description,
    );
    assert_sync_contract_has_real_table_policies();
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_intention_and_decision_basis_both_survive_convergence() {
    let fixture = real_sync_fixture("concurrent_intention_and_decision_basis").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "basis survival");
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    assert_real_intention_content(
        &fixture.edge2.store,
        seeded.intention.id,
        &seeded.intention.description,
    );
    assert_real_decision_content(
        &fixture.edge2.store,
        seeded.decision.id,
        &seeded.decision.description,
    );
    let touches = touch_same_decision_graph_concurrently(&fixture, &seeded).await;
    bidirectional_sync(&fixture.edge1_client, &fixture.edge2_client).await;

    assert_real_intention_content(
        &fixture.edge2.store,
        seeded.intention.id,
        &seeded.intention.description,
    );
    assert_real_intention_content(
        &fixture.edge1.store,
        seeded.intention.id,
        &seeded.intention.description,
    );
    for store in [&fixture.edge1.store, &fixture.edge2.store] {
        let decision = store
            .get_decision(seeded.decision.id)
            .expect("read synced decision")
            .expect("decision exists");
        assert!(
            decision
                .based_on_snapshots
                .contains(&seeded.basis_snapshot_id),
            "decision basis snapshot must survive convergence"
        );
    }
    assert_real_observation_content(
        &fixture.edge1.store,
        touches.edge2_observation_id,
        "edge-two-concurrent-touch",
        "path",
        serde_json::json!("src/auth_edge2.py"),
    );
    assert_real_observation_content(
        &fixture.edge2.store,
        touches.edge1_observation_id,
        "edge-one-concurrent-touch",
        "path",
        serde_json::json!("src/auth_edge1.py"),
    );
    assert_sync_contract_has_real_table_policies();
}

#[tokio::test(flavor = "multi_thread")]
async fn self_echo_pull_of_own_writes_is_idempotent() {
    let fixture = self_echo_sync_fixture("self_echo_pull_of_own_writes_is_idempotent").await;
    let seeded = seed_decision_graph(&fixture.node.store, "tenant-alpha", "self echo");
    let pushed = within(fixture.client.push()).await.expect("self push");
    assert!(
        pushed.applied_rows > 0,
        "the separate hub must accept the edge's graph before the echo pull: {pushed:?}"
    );
    assert!(
        pushed.conflicts.is_empty(),
        "the hub setup must not report a conflict: {pushed:?}"
    );
    assert_eq!(
        pushed.skipped_rows, 0,
        "the hub setup must not refuse a row by policy: {pushed:?}"
    );
    let pulled = within(fixture.client.pull_default())
        .await
        .expect("self pull");
    assert_eq!(
        pulled.applied_rows, 0,
        "an unchanged self-echo applies no row: {pulled:?}"
    );
    assert!(
        pulled.conflicts.is_empty(),
        "an unchanged self-echo is not a conflict: {pulled:?}"
    );
    assert_eq!(
        pulled.skipped_rows, 0,
        "an unchanged self-echo is not a policy refusal: {pulled:?}"
    );

    assert_decision_status_active(&fixture.node.store, seeded.decision.id);
    let invalidations = fixture
        .node
        .store
        .list_invalidations(seeded.context_id, Some(seeded.decision.id), None)
        .expect("read invalidations after self echo");
    assert_eq!(
        invalidations.len(),
        0,
        "self echo must not add invalidations"
    );
    assert_sync_contract_has_real_table_policies();
}
