use crate::capture_sync_common::{
    assert_decision_status_active, assert_real_decision_content, assert_real_intention_content,
    assert_real_observation_content, assert_real_outcome_state,
    assert_sync_contract_has_real_table_policies, bidirectional_sync, push_pull, real_sync_fixture,
    record_contended_outcomes_concurrently, seed_decision_graph, seed_two_edges_concurrently,
    self_echo_sync_fixture, touch_same_decision_graph_concurrently,
};

#[tokio::test(flavor = "multi_thread")]
async fn capture_succeeds_offline_and_syncs_on_reconnect() {
    let fixture = real_sync_fixture("capture_succeeds_offline_and_syncs_on_reconnect").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "offline");
    let down_client = contextdb_server::SyncClient::new(
        fixture.edge1.store.sync_database(),
        "nats://127.0.0.1:1",
        &fixture.tenant_id,
    );
    let push_while_down = down_client.push().await;

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
    fixture
        .edge1_client
        .push()
        .await
        .expect("edge1 outcome push");
    fixture
        .edge2_client
        .push()
        .await
        .expect("edge2 outcome push");
    fixture
        .edge1_client
        .pull(&crate::capture_sync_common::cg_conflict_policies())
        .await
        .expect("edge1 outcome pull");
    fixture
        .edge2_client
        .pull(&crate::capture_sync_common::cg_conflict_policies())
        .await
        .expect("edge2 outcome pull");

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
    fixture.client.push().await.expect("self push");
    fixture
        .client
        .pull(&crate::capture_sync_common::cg_conflict_policies())
        .await
        .expect("self pull");

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
