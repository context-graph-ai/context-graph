use crate::capture_sync_common::{
    assert_exact_extracted_from_edge, assert_no_raw_row_by_uuid, assert_real_decision_content,
    assert_real_digest_content, assert_real_entity_snapshot, assert_real_evidence_content,
    assert_real_intention_content, assert_real_observation_content, assert_real_outcome_state,
    assert_sync_contract_has_real_table_policies, bootstrap_sync_schema_without_rows,
    pre_client_sync_fixture, push_pull, real_sync_fixture, register_cg_sync_tables,
    seed_decision_graph,
};

#[tokio::test(flavor = "multi_thread")]
async fn decision_observation_and_extracted_from_edge_appear_on_second_edge_after_sync() {
    let fixture = real_sync_fixture("decision_observation_and_extracted_from_edge_appear").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "core convergence");
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    assert_real_decision_content(
        &fixture.edge2.store,
        seeded.decision.id,
        &seeded.decision.description,
    );
    assert_real_observation_content(
        &fixture.edge2.store,
        seeded.observation_id,
        "agent-session",
        "path",
        serde_json::json!("src/auth.py"),
    );
    assert_real_evidence_content(
        &fixture.edge2.store,
        seeded.evidence_id,
        seeded.observation_id,
        context_graph::EvidenceKind::Text,
        "vigil-edge:session/core convergence",
    );
    assert_real_entity_snapshot(
        &fixture.edge2.store,
        seeded.basis_snapshot_id,
        seeded.entity_id,
        &seeded.entity_name,
        &seeded.entity_path,
    );
    assert_real_digest_content(
        &fixture.edge2.store,
        seeded.digest_id,
        &seeded.digest_summary,
    );
    assert_exact_extracted_from_edge(
        &fixture.edge2.store,
        seeded.context_id,
        seeded.decision.id.as_uuid(),
        seeded.digest_id,
    );
    assert_sync_contract_has_real_table_policies();
}

#[tokio::test(flavor = "multi_thread")]
async fn evidence_and_outcome_rows_converge_by_id() {
    let fixture = real_sync_fixture("evidence_and_outcome_rows_converge_by_id").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "evidence outcome");
    let first_outcome = fixture
        .edge1
        .store
        .record_outcome(seeded.decision.id, false, Some("first outcome".to_string()))
        .expect("record first outcome");
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let replacing_outcome = fixture
        .edge1
        .store
        .record_outcome(
            seeded.decision.id,
            true,
            Some("replacement outcome".to_string()),
        )
        .expect("record replacement outcome");
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    assert_real_evidence_content(
        &fixture.edge2.store,
        seeded.evidence_id,
        seeded.observation_id,
        context_graph::EvidenceKind::Text,
        "vigil-edge:session/evidence outcome",
    );
    assert_real_outcome_state(
        &fixture.edge2.store,
        seeded.decision.id,
        &replacing_outcome,
        1,
    );
    assert_ne!(first_outcome.id, replacing_outcome.id);
    assert_sync_contract_has_real_table_policies();
}

#[tokio::test(flavor = "multi_thread")]
async fn existing_backlog_captured_before_first_sync_appears_on_second_edge() {
    let fixture =
        pre_client_sync_fixture("existing_backlog_captured_before_first_sync_appears").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "preconnect backlog");
    let (edge1_client, edge2_client) = fixture.connect();
    push_pull(&edge1_client, &edge2_client).await;

    assert_real_decision_content(
        &fixture.edge2.store,
        seeded.decision.id,
        &seeded.decision.description,
    );
    assert_real_intention_content(
        &fixture.edge2.store,
        seeded.intention.id,
        &seeded.intention.description,
    );
    assert_real_observation_content(
        &fixture.edge2.store,
        seeded.observation_id,
        "agent-session",
        "path",
        serde_json::json!("src/auth.py"),
    );
    assert_real_evidence_content(
        &fixture.edge2.store,
        seeded.evidence_id,
        seeded.observation_id,
        context_graph::EvidenceKind::Text,
        "vigil-edge:session/preconnect backlog",
    );
    assert_real_entity_snapshot(
        &fixture.edge2.store,
        seeded.basis_snapshot_id,
        seeded.entity_id,
        &seeded.entity_name,
        &seeded.entity_path,
    );
    assert_real_digest_content(
        &fixture.edge2.store,
        seeded.digest_id,
        &seeded.digest_summary,
    );
    assert_exact_extracted_from_edge(
        &fixture.edge2.store,
        seeded.context_id,
        seeded.decision.id.as_uuid(),
        seeded.digest_id,
    );
    assert_sync_contract_has_real_table_policies();
}

#[tokio::test(flavor = "multi_thread")]
async fn cross_context_isolation_survives_sync() {
    let fixture = real_sync_fixture("cross_context_isolation_survives_sync").await;
    bootstrap_sync_schema_without_rows(&fixture).await;
    let context_a = seed_decision_graph(&fixture.edge1.store, "tenant-alpha-a", "context A");
    let context_b = seed_decision_graph(&fixture.edge1.store, "tenant-alpha-b", "context B");
    let edge2_context_b = fixture.edge2.scoped_store([context_b.context_id]);
    let edge2_scoped_client = contextdb_server::SyncClient::new(
        edge2_context_b.sync_database(),
        &fixture.nats_url,
        &fixture.tenant_id,
    );
    register_cg_sync_tables(&edge2_scoped_client);
    fixture
        .edge1_client
        .push()
        .await
        .expect("push context rows to hub");
    let pull = edge2_scoped_client
        .pull(&crate::capture_sync_common::cg_conflict_policies())
        .await;
    assert!(
        pull.is_ok(),
        "context-B scoped data pull should succeed after admin DDL bootstrap: {pull:?}"
    );

    assert_real_decision_content(
        &edge2_context_b,
        context_b.decision.id,
        &context_b.decision.description,
    );
    assert_no_raw_row_by_uuid(
        &fixture.edge2.store,
        "decisions",
        context_a.decision.id.as_uuid(),
        "context A decision must be rejected at apply and absent from edge2 storage",
    );
    assert_eq!(
        edge2_context_b
            .get_decision(context_a.decision.id)
            .expect("read context A through context-B scoped handle"),
        None,
        "context A row must be rejected by the context-scoped apply guard"
    );
    assert_sync_contract_has_real_table_policies();
}
