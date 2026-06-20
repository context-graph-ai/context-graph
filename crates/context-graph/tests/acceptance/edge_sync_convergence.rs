use crate::capture_sync_common::{
    TestStore, assert_exact_extracted_from_edge, assert_no_raw_row_by_uuid,
    assert_real_decision_content, assert_real_digest_content, assert_real_entity_snapshot,
    assert_real_evidence_content, assert_real_intention_content, assert_real_observation_content,
    assert_real_outcome_state, assert_sync_contract_has_real_table_policies,
    bootstrap_sync_schema_without_rows, drain_pull, drain_push, pre_client_sync_fixture, push_pull,
    real_sync_fixture, register_cg_sync_tables, seed_decision_graph,
};
use context_graph::{DecisionStatus, InvalidationStatus, Store};
use contextdb_core::Value as DbValue;

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
async fn wiped_edge_pull_restores_full_committed_state_including_decision_status() {
    let fixture = real_sync_fixture(
        "wiped_edge_pull_restores_full_committed_state_including_decision_status",
    )
    .await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "wiped restore");
    let extracted_from_edge_id = exact_edge_id(
        &fixture.edge1.store,
        seeded.context_id,
        seeded.decision.id.as_uuid(),
        seeded.digest_id.as_uuid(),
        "EXTRACTED_FROM",
    );
    let based_on_edge_id = exact_edge_id(
        &fixture.edge1.store,
        seeded.context_id,
        seeded.decision.id.as_uuid(),
        seeded.entity_id.as_uuid(),
        "BASED_ON",
    );
    let invalidation = fixture
        .edge1
        .store
        .invalidate_decision(
            seeded.decision.id,
            seeded.observation_id,
            "local edge invalidated stale auth decision",
        )
        .expect("invalidate decision on source edge");
    let outcome = fixture
        .edge1
        .store
        .record_outcome(
            seeded.decision.id,
            false,
            Some("restored failed auth outcome".to_string()),
        )
        .expect("record outcome on source edge");

    drain_push(&fixture.edge1_client).await;
    assert_text_column_by_id(
        fixture.hub_store(),
        "decisions",
        seeded.decision.id.as_uuid(),
        "status",
        "invalidated",
    );
    assert_text_column_by_id(
        fixture.hub_store(),
        "invalidations",
        invalidation.invalidation_id.as_uuid(),
        "status",
        "pending",
    );

    let restored_edge = TestStore::new("wiped_restore_reopened_edge");
    let restored_client = contextdb_server::SyncClient::new(
        restored_edge.store.sync_database(),
        &fixture.nats_url,
        &fixture.tenant_id,
    );
    register_cg_sync_tables(&restored_client);
    assert_eq!(
        restored_edge
            .store
            .get_decision(seeded.decision.id)
            .expect("read pre-restore decision"),
        None,
        "replacement edge must begin empty before pulling from the hub"
    );

    drain_pull(&restored_client).await;

    assert_text_column_by_id(
        &restored_edge.store,
        "contexts",
        seeded.context_id.as_uuid(),
        "name",
        "tenant-alpha",
    );
    assert_text_column_by_id(
        &restored_edge.store,
        "entities",
        seeded.entity_id.as_uuid(),
        "name",
        &seeded.entity_name,
    );
    assert_real_intention_content(
        &restored_edge.store,
        seeded.intention.id,
        &seeded.intention.description,
    );
    assert_decision_status_from_committed_column(
        &restored_edge.store,
        seeded.decision.id,
        DecisionStatus::Invalidated,
    );
    assert_real_decision_content(
        &restored_edge.store,
        seeded.decision.id,
        &seeded.decision.description,
    );
    assert_real_observation_content(
        &restored_edge.store,
        seeded.observation_id,
        "agent-session",
        "path",
        serde_json::json!("src/auth.py"),
    );
    assert_real_evidence_content(
        &restored_edge.store,
        seeded.evidence_id,
        seeded.observation_id,
        context_graph::EvidenceKind::Text,
        "vigil-edge:session/wiped restore",
    );
    assert_real_entity_snapshot(
        &restored_edge.store,
        seeded.basis_snapshot_id,
        seeded.entity_id,
        &seeded.entity_name,
        &seeded.entity_path,
    );
    assert_real_digest_content(
        &restored_edge.store,
        seeded.digest_id,
        &seeded.digest_summary,
    );
    assert_real_outcome_state(&restored_edge.store, seeded.decision.id, &outcome, 1);
    assert_restored_invalidation(
        &restored_edge.store,
        seeded.context_id,
        seeded.decision.id,
        seeded.observation_id,
        invalidation.invalidation_id,
    );
    assert_exact_extracted_from_edge(
        &restored_edge.store,
        seeded.context_id,
        seeded.decision.id.as_uuid(),
        seeded.digest_id,
    );
    assert_exact_edge_by_id(
        &restored_edge.store,
        extracted_from_edge_id,
        seeded.context_id,
        seeded.decision.id.as_uuid(),
        seeded.digest_id.as_uuid(),
        "EXTRACTED_FROM",
    );
    assert_exact_edge_by_id(
        &restored_edge.store,
        based_on_edge_id,
        seeded.context_id,
        seeded.decision.id.as_uuid(),
        seeded.entity_id.as_uuid(),
        "BASED_ON",
    );
    assert_sync_contract_has_real_table_policies();
}

fn assert_decision_status_from_committed_column(
    store: &Store,
    decision_id: context_graph::DecisionId,
    expected: DecisionStatus,
) {
    assert_text_column_by_id(
        store,
        "decisions",
        decision_id.as_uuid(),
        "status",
        decision_status_str(expected),
    );
    let decision = store
        .get_decision(decision_id)
        .expect("read restored decision")
        .expect("restored decision exists");
    assert_eq!(decision.status, expected);
}

fn assert_restored_invalidation(
    store: &Store,
    context_id: context_graph::ContextId,
    decision_id: context_graph::DecisionId,
    observation_id: context_graph::ObservationId,
    invalidation_id: context_graph::InvalidationId,
) {
    let invalidations = store
        .list_invalidations(context_id, Some(decision_id), None)
        .expect("read restored invalidations");
    assert_eq!(
        invalidations.len(),
        1,
        "restored invalidation rows for decision {decision_id}: {invalidations:?}"
    );
    let invalidation = &invalidations[0];
    assert_eq!(invalidation.id, invalidation_id);
    assert_eq!(invalidation.status, InvalidationStatus::Pending);
    assert_eq!(invalidation.context_id, context_id);
    assert_eq!(invalidation.affected_decision_id, decision_id);
    assert_eq!(invalidation.trigger_observation_id, observation_id);
    assert_text_column_by_id(
        store,
        "invalidations",
        invalidation_id.as_uuid(),
        "status",
        "pending",
    );
}

fn assert_exact_edge_by_id(
    store: &Store,
    edge_id: uuid::Uuid,
    context_id: context_graph::ContextId,
    source_id: uuid::Uuid,
    target_id: uuid::Uuid,
    edge_type: &str,
) {
    let row = raw_row_by_id(store, "edges", edge_id);
    assert_eq!(row.get("source_id"), Some(&DbValue::Uuid(source_id)));
    assert_eq!(row.get("target_id"), Some(&DbValue::Uuid(target_id)));
    assert_eq!(
        row.get("context_id"),
        Some(&DbValue::Uuid(context_id.as_uuid()))
    );
    assert_eq!(
        row.get("edge_type"),
        Some(&DbValue::Text(edge_type.to_string()))
    );
}

fn exact_edge_id(
    store: &Store,
    context_id: context_graph::ContextId,
    source_id: uuid::Uuid,
    target_id: uuid::Uuid,
    edge_type: &str,
) -> uuid::Uuid {
    let rows = matching_edges(store, context_id, source_id, target_id, edge_type);
    assert_eq!(
        rows.len(),
        1,
        "expected exactly one source {edge_type} edge from {source_id} to {target_id} in context {context_id}; rows={rows:?}"
    );
    match rows[0].get("id") {
        Some(DbValue::Uuid(id)) => *id,
        other => panic!("source edge missing UUID id: {other:?}"),
    }
}

fn matching_edges(
    store: &Store,
    context_id: context_graph::ContextId,
    source_id: uuid::Uuid,
    target_id: uuid::Uuid,
    edge_type: &str,
) -> Vec<std::collections::HashMap<String, DbValue>> {
    let db = store.sync_database();
    db.scan("edges", db.snapshot())
        .expect("scan edges")
        .into_iter()
        .filter(|row| {
            matches!(
                (
                    row.values.get("source_id"),
                    row.values.get("target_id"),
                    row.values.get("context_id"),
                    row.values.get("edge_type"),
                ),
                (
                    Some(DbValue::Uuid(actual_source)),
                    Some(DbValue::Uuid(actual_target)),
                    Some(DbValue::Uuid(actual_context)),
                    Some(DbValue::Text(actual_type)),
                ) if *actual_source == source_id
                    && *actual_target == target_id
                    && *actual_context == context_id.as_uuid()
                    && actual_type == edge_type
            )
        })
        .map(|row| row.values)
        .collect()
}

fn assert_text_column_by_id(
    store: &Store,
    table: &str,
    id: uuid::Uuid,
    column: &str,
    expected: &str,
) {
    let row = raw_row_by_id(store, table, id);
    assert_eq!(
        row.get(column),
        Some(&DbValue::Text(expected.to_string())),
        "{table}.{column} for restored row {id}"
    );
}

fn raw_row_by_id(
    store: &Store,
    table: &str,
    id: uuid::Uuid,
) -> std::collections::HashMap<String, DbValue> {
    let db = store.sync_database();
    db.point_lookup(table, "id", &DbValue::Uuid(id), db.snapshot())
        .expect("point lookup restored row")
        .unwrap_or_else(|| panic!("restored row {table}.{id} must exist"))
        .values
}

fn decision_status_str(status: DecisionStatus) -> &'static str {
    match status {
        DecisionStatus::Active => "active",
        DecisionStatus::Invalidated => "invalidated",
        DecisionStatus::Superseded => "superseded",
        DecisionStatus::Archived => "archived",
    }
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
    drain_push(&fixture.edge1_client).await;
    drain_pull(&edge2_scoped_client).await;

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
