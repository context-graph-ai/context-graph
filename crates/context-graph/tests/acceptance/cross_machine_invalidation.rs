use context_graph::ingest::{
    MemoryInvalidationNotice, changed_since_context_cursor, record_conversation_with_extractor,
    standing_impact,
};
use context_graph::{DecisionStatus, Invalidation, ObservationId, Store};

use crate::capture_sync_common::{
    SeededGraph, assert_real_observation_content, assert_sync_contract_has_real_table_policies,
    capture_request, drain_pull, drain_push, extractor, push_pull, real_sync_fixture,
    seed_decision_graph,
};

#[tokio::test(flavor = "multi_thread")]
async fn basis_change_captured_on_converged_edge_two_invalidates_edge_one_decision() {
    let fixture =
        real_sync_fixture("basis_change_captured_on_converged_edge_two_invalidates").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "edge one basis");
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let captured = record_conversation_with_extractor(
        &fixture.edge2.store,
        capture_request(
            "tenant-alpha",
            "Observation O_edge_two_trigger: E_auth_module moved -> src/authentication.py",
        ),
        &extractor(),
    )
    .expect("edge2 captures basis change against pulled edge1 decision");
    let captured_trigger_id = captured
        .stable_ids
        .get("O_edge_two_trigger")
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(ObservationId::from)
        .expect("edge2 capture report must expose the captured trigger observation id");
    assert_ne!(
        captured_trigger_id, seeded.observation_id,
        "trigger observation must be minted by edge2 capture, not reused from edge1"
    );
    assert!(
        captured
            .invalidations
            .iter()
            .any(|notice| notice.affected_decision_id == Some(seeded.decision.id)),
        "edge2 capture must recompute invalidation against the pulled edge1 decision; got {:?}",
        captured.invalidations
    );
    let edge2_invalidation = assert_basis_invalidation(
        &fixture.edge2.store,
        &seeded,
        Some(captured_trigger_id),
        "edge2 before push",
    );
    let trigger_id = edge2_invalidation.trigger_observation_id;
    drain_push(&fixture.edge2_client).await;
    drain_pull(&fixture.edge1_client).await;

    assert_basis_invalidation(
        &fixture.edge1.store,
        &seeded,
        Some(trigger_id),
        "edge1 after pull",
    );
    assert_user_facing_warning_surface(&fixture.edge1.store, &seeded);
    assert_real_observation_content(
        &fixture.edge2.store,
        trigger_id,
        "agent-session",
        "path",
        serde_json::json!("src/authentication.py"),
    );
    assert_sync_contract_has_real_table_policies();
}

fn assert_basis_invalidation(
    store: &Store,
    seeded: &SeededGraph,
    trigger_id: Option<ObservationId>,
    location: &str,
) -> Invalidation {
    let invalidations = store
        .list_invalidations(seeded.context_id, Some(seeded.decision.id), None)
        .unwrap_or_else(|err| panic!("read invalidations at {location}: {err}"));
    let invalidation = invalidations
        .iter()
        .find(|invalidation| {
            invalidation.affected_decision_id == seeded.decision.id
                && trigger_id
                    .map(|trigger_id| invalidation.trigger_observation_id == trigger_id)
                    .unwrap_or(true)
                && invalidation
                    .threatened_intention_ids
                    .contains(&seeded.intention.id)
        })
        .unwrap_or_else(|| {
            panic!(
                "{location} must hold the invalidation created by edge2 capture; got {invalidations:?}"
            )
        });
    assert!(invalidation.basis_diff.is_invalidated_candidate);
    let divergence = invalidation
        .basis_diff
        .divergences
        .iter()
        .find(|divergence| {
            divergence.entity_id == seeded.entity_id
                && divergence.basis_snapshot_id == seeded.basis_snapshot_id
        })
        .expect("invalidation must carry the changed synced basis entity");
    assert!(
        divergence.value_diffs.iter().any(|diff| {
            diff.field == "path"
                && diff.basis_value == Some(serde_json::json!("src/auth.py"))
                && diff.current_value == Some(serde_json::json!("src/authentication.py"))
        }),
        "basis diff must name old and new path values; got {divergence:?}"
    );
    let decision = store
        .get_decision(seeded.decision.id)
        .unwrap_or_else(|err| panic!("read invalidated decision at {location}: {err}"))
        .expect("decision exists");
    assert_eq!(decision.status, DecisionStatus::Invalidated);
    invalidation.clone()
}

fn assert_user_facing_warning_surface(store: &Store, seeded: &SeededGraph) {
    let standing = standing_impact(store, seeded.context_id).expect("standing impact after sync");
    assert_synced_warning_notice(&standing, seeded, "standing impact after sync");
    let changed = changed_since_context_cursor(store, seeded.context_id)
        .expect("changed warnings after sync");
    assert_synced_warning_notice(&changed, seeded, "changed warnings after sync");
    let repeated = changed_since_context_cursor(store, seeded.context_id)
        .expect("changed warning cursor after first read");
    assert!(
        repeated.is_empty(),
        "changed warnings must advance their cursor instead of replaying the same synced warning forever; repeated={repeated:?}"
    );
}

fn assert_synced_warning_notice(
    notices: &[MemoryInvalidationNotice],
    seeded: &SeededGraph,
    location: &str,
) {
    let notice = notices
        .iter()
        .find(|notice| notice.affected_decision_id == Some(seeded.decision.id))
        .unwrap_or_else(|| {
            panic!("{location} must surface the synced invalidation warning; got {notices:?}")
        });
    assert_eq!(
        (
            notice.decision_text.as_str(),
            notice.served_goal.as_str(),
            notice.basis_line.as_str(),
            notice.goal_now_unserved,
            notice.current_decision_text.as_deref(),
        ),
        (
            seeded.decision.description.as_str(),
            seeded.intention.description.as_str(),
            "src/auth.py -> src/authentication.py",
            true,
            None,
        ),
        "{location} must render the same warning the owner sees, not only persist the raw invalidation row"
    );
}
