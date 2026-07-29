use crate::capture_sync_common::{
    TestStore, assert_exact_extracted_from_edge, assert_no_raw_row_by_uuid,
    assert_real_decision_content, assert_real_digest_content, assert_real_entity_snapshot,
    assert_real_evidence_content, assert_real_intention_content, assert_real_observation_content,
    assert_real_outcome_state, assert_sync_contract_has_real_table_policies,
    bootstrap_sync_schema_without_rows, drain_pull, drain_push, pre_client_sync_fixture, push_pull,
    real_sync_fixture, seed_decision_graph,
};
use context_graph::{DecisionId, DecisionStatus, InvalidationId, InvalidationStatus, Store, TxId};
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
    let invalidated = insert_committed_invalidated_decision_fixture(
        &fixture.edge1.store,
        seeded.context_id,
        seeded.intention.id,
        seeded.entity_id,
        seeded.observation_id,
    );
    let status_only = insert_committed_status_only_decision_fixture(
        &fixture.edge1.store,
        seeded.context_id,
        DecisionStatus::Invalidated,
        "status-only invalidated decision restore sentinel",
    );
    let active_control = insert_committed_status_only_decision_fixture(
        &fixture.edge1.store,
        seeded.context_id,
        DecisionStatus::Active,
        "status-only active decision restore control",
    );
    let outcome = fixture
        .edge1
        .store
        .record_outcome(
            invalidated.decision_id,
            false,
            Some("restored failed auth outcome".to_string()),
        )
        .expect("record outcome on source edge");

    drain_push(&fixture.edge1_client).await;
    assert_text_column_by_id(
        fixture.hub_store(),
        "decisions",
        invalidated.decision_id.as_uuid(),
        "status",
        "invalidated",
    );
    assert_text_column_by_id(
        fixture.hub_store(),
        "invalidations",
        invalidated.invalidation_id.as_uuid(),
        "status",
        "pending",
    );
    assert_text_column_by_id(
        fixture.hub_store(),
        "decisions",
        status_only.decision_id.as_uuid(),
        "status",
        "invalidated",
    );
    assert_text_column_by_id(
        fixture.hub_store(),
        "decisions",
        active_control.decision_id.as_uuid(),
        "status",
        "active",
    );

    let restored_edge = TestStore::new("wiped_restore_reopened_edge");
    let restored_client = fixture.new_edge_client(
        restored_edge.store.sync_database(),
        &restored_edge
            .db_path()
            .with_extension("fabric-identity.key"),
    );
    assert_eq!(
        restored_edge
            .store
            .get_decision(invalidated.decision_id)
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
        invalidated.decision_id,
        DecisionStatus::Invalidated,
    );
    assert_decision_status_from_committed_column(
        &restored_edge.store,
        status_only.decision_id,
        DecisionStatus::Invalidated,
    );
    assert_decision_status_from_committed_column(
        &restored_edge.store,
        active_control.decision_id,
        DecisionStatus::Active,
    );
    assert_real_decision_content(
        &restored_edge.store,
        invalidated.decision_id,
        &invalidated.description,
    );
    assert_real_decision_content(
        &restored_edge.store,
        status_only.decision_id,
        &status_only.description,
    );
    assert_real_decision_content(
        &restored_edge.store,
        active_control.decision_id,
        &active_control.description,
    );
    assert_no_invalidation_for_decision(
        &restored_edge.store,
        seeded.context_id,
        status_only.decision_id,
    );
    assert_no_invalidation_for_decision(
        &restored_edge.store,
        seeded.context_id,
        active_control.decision_id,
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
    assert_real_outcome_state(&restored_edge.store, invalidated.decision_id, &outcome, 1);
    assert_restored_invalidation(
        &restored_edge.store,
        seeded.context_id,
        invalidated.decision_id,
        seeded.observation_id,
        invalidated.invalidation_id,
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
        invalidated.based_on_edge_id,
        seeded.context_id,
        invalidated.decision_id.as_uuid(),
        seeded.entity_id.as_uuid(),
        "BASED_ON",
    );
    assert_exact_edge_by_id(
        &restored_edge.store,
        invalidated.serves_edge_id,
        seeded.context_id,
        invalidated.decision_id.as_uuid(),
        seeded.intention.id.as_uuid(),
        "SERVES",
    );
    assert_sync_contract_has_real_table_policies();
}

struct InvalidatedDecisionFixture {
    decision_id: DecisionId,
    invalidation_id: InvalidationId,
    serves_edge_id: uuid::Uuid,
    based_on_edge_id: uuid::Uuid,
    description: String,
}

struct StatusOnlyDecisionFixture {
    decision_id: DecisionId,
    description: String,
}

fn insert_committed_invalidated_decision_fixture(
    store: &Store,
    context_id: context_graph::ContextId,
    intention_id: context_graph::IntentionId,
    entity_id: context_graph::EntityId,
    observation_id: context_graph::ObservationId,
) -> InvalidatedDecisionFixture {
    let decision_id = DecisionId::new_v7();
    let invalidation_id = InvalidationId::new_v7();
    let serves_edge_id = uuid::Uuid::now_v7();
    let based_on_edge_id = uuid::Uuid::now_v7();
    let description = "auth module was invalidated before restore".to_string();
    let db = store.sync_database();
    db.execute(
        "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
        &std::collections::HashMap::from([
            ("id".to_string(), DbValue::Uuid(decision_id.as_uuid())),
            (
                "decision_type".to_string(),
                DbValue::Text("technical".to_string()),
            ),
            ("description".to_string(), DbValue::Text(description.clone())),
            ("reasoning".to_string(), DbValue::Json(serde_json::json!({}))),
            ("confidence".to_string(), DbValue::Float64(0.9)),
            (
                "status".to_string(),
                DbValue::Text("invalidated".to_string()),
            ),
            ("agent_id".to_string(), DbValue::Null),
            ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
            ("properties".to_string(), DbValue::Json(serde_json::json!({}))),
            ("tags".to_string(), DbValue::Json(serde_json::json!([]))),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert committed invalidated decision fixture");
    insert_edge_row(
        store,
        serves_edge_id,
        context_id,
        decision_id.as_uuid(),
        intention_id.as_uuid(),
        "SERVES",
    );
    insert_edge_row(
        store,
        based_on_edge_id,
        context_id,
        decision_id.as_uuid(),
        entity_id.as_uuid(),
        "BASED_ON",
    );
    let basis_diff = context_graph::BasisDiffResult {
        is_invalidated_candidate: true,
        divergences: Vec::new(),
        read_txid: TxId(1),
        current_snapshot_read_txid: TxId(1),
    };
    db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $reason, $basis_diff, $cascade_from, $severity, $threatened_intention_ids, $intention_now_un_served, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &std::collections::HashMap::from([
            ("id".to_string(), DbValue::Uuid(invalidation_id.as_uuid())),
            ("status".to_string(), DbValue::Text("pending".to_string())),
            ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
            (
                "affected_decision_id".to_string(),
                DbValue::Uuid(decision_id.as_uuid()),
            ),
            (
                "trigger_observation_id".to_string(),
                DbValue::Uuid(observation_id.as_uuid()),
            ),
            (
                "reason".to_string(),
                DbValue::Text("committed invalidated restore fixture".to_string()),
            ),
            (
                "basis_diff".to_string(),
                DbValue::Json(serde_json::to_value(basis_diff).expect("basis diff JSON")),
            ),
            ("cascade_from".to_string(), DbValue::Null),
            ("severity".to_string(), DbValue::Text("warning".to_string())),
            (
                "threatened_intention_ids".to_string(),
                DbValue::Json(
                    serde_json::to_value(vec![intention_id])
                        .expect("threatened intention ids JSON"),
                ),
            ),
            ("intention_now_un_served".to_string(), DbValue::Bool(true)),
            ("detected_at".to_string(), DbValue::TxId(TxId(1))),
            ("resolved_at".to_string(), DbValue::Null),
            ("resolution_action".to_string(), DbValue::Null),
            ("resolution_decision_id".to_string(), DbValue::Null),
        ]),
    )
    .expect("insert committed invalidation fixture");
    InvalidatedDecisionFixture {
        decision_id,
        invalidation_id,
        serves_edge_id,
        based_on_edge_id,
        description,
    }
}

fn insert_committed_status_only_decision_fixture(
    store: &Store,
    context_id: context_graph::ContextId,
    status: DecisionStatus,
    description: &str,
) -> StatusOnlyDecisionFixture {
    let decision_id = DecisionId::new_v7();
    let description = description.to_string();
    store
        .sync_database()
        .execute(
            "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
            &std::collections::HashMap::from([
                ("id".to_string(), DbValue::Uuid(decision_id.as_uuid())),
                (
                    "decision_type".to_string(),
                    DbValue::Text("technical".to_string()),
                ),
                ("description".to_string(), DbValue::Text(description.clone())),
                ("reasoning".to_string(), DbValue::Json(serde_json::json!({}))),
                ("confidence".to_string(), DbValue::Float64(0.5)),
                (
                    "status".to_string(),
                    DbValue::Text(decision_status_str(status).to_string()),
                ),
                ("agent_id".to_string(), DbValue::Null),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("properties".to_string(), DbValue::Json(serde_json::json!({}))),
                ("tags".to_string(), DbValue::Json(serde_json::json!([]))),
                ("created_at".to_string(), DbValue::TxId(TxId(1))),
            ]),
        )
        .expect("insert committed status-only invalidated decision fixture");
    StatusOnlyDecisionFixture {
        decision_id,
        description,
    }
}

fn insert_edge_row(
    store: &Store,
    edge_id: uuid::Uuid,
    context_id: context_graph::ContextId,
    source_id: uuid::Uuid,
    target_id: uuid::Uuid,
    edge_type: &str,
) {
    store
        .sync_database()
        .execute(
            "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
            &std::collections::HashMap::from([
                ("id".to_string(), DbValue::Uuid(edge_id)),
                ("source_id".to_string(), DbValue::Uuid(source_id)),
                ("target_id".to_string(), DbValue::Uuid(target_id)),
                ("edge_type".to_string(), DbValue::Text(edge_type.to_string())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("valid_from".to_string(), DbValue::TxId(TxId(1))),
                ("valid_to".to_string(), DbValue::Null),
                ("properties".to_string(), DbValue::Json(serde_json::json!({}))),
            ]),
        )
        .unwrap_or_else(|err| panic!("insert {edge_type} edge fixture: {err:?}"));
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

fn assert_no_invalidation_for_decision(
    store: &Store,
    context_id: context_graph::ContextId,
    decision_id: DecisionId,
) {
    let invalidations = store
        .list_invalidations(context_id, Some(decision_id), None)
        .expect("read invalidations for status-only sentinel");
    assert!(
        invalidations.is_empty(),
        "status-only decision {decision_id} must not be supported by an invalidation row: {invalidations:?}"
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
        "{table}.{column} for row {id}"
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
    let edge2_scoped_client = fixture.new_edge_client(
        edge2_context_b.sync_database(),
        &fixture
            .edge2
            .db_path()
            .with_extension("fabric-identity.key"),
    );
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
