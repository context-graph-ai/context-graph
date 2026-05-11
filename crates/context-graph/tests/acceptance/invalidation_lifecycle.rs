#[path = "../common/invalidation_fixtures.rs"]
mod invalidation_fixtures;

use context_graph::*;
use invalidation_fixtures::*;

fn agent(name: &str) -> Principal {
    Principal::Agent(name.to_string())
}

#[test]
fn get_invalidation_preserves_row_shape_byte_identical() {
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);

    let pulled = store
        .with_principal(agent("agent-test"))
        .get_invalidation(fixture.invalidation.id)
        .expect("get invalidation");

    assert_eq!(pulled, fixture.invalidation);
    drop(store);

    let reopened = reopen(&fixture.config);
    let pulled_after_reopen = reopened
        .with_principal(agent("agent-test"))
        .get_invalidation(fixture.invalidation.id)
        .expect("get invalidation after reopen");
    assert_eq!(pulled_after_reopen, fixture.invalidation);
}

#[test]
fn get_invalidation_unknown_id_returns_not_found() {
    let (_dir, _config, store) = fresh_store();
    let id = InvalidationId::new_v7();

    assert_not_found(store.with_system_principal().get_invalidation(id), id);
}

#[test]
fn get_invalidation_cross_context_returns_not_found() {
    let fixture = seed_pending_fixture();
    let admin = reopen(&fixture.config);
    let other = create_context(&admin, "tenant-beta");
    drop(admin);
    let scoped = reopen_scoped(&fixture.config, [other.id]);

    assert_not_found(
        scoped
            .with_principal(agent("consultant"))
            .get_invalidation(fixture.invalidation.id),
        fixture.invalidation.id,
    );

    let admin = reopen(&fixture.config);
    let still_present = find_invalidation(&admin, fixture.context_id, fixture.invalidation.id);
    assert_eq!(still_present, fixture.invalidation);
}

#[test]
fn acknowledge_invalidation_pending_to_acknowledged() {
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);

    let acknowledged = store
        .with_principal(agent("agent-test"))
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("acknowledge");

    assert_ack_shape(&fixture.invalidation, &acknowledged);
    drop(store);

    let persisted = reopen(&fixture.config)
        .with_principal(agent("agent-test"))
        .get_invalidation(fixture.invalidation.id)
        .expect("persisted invalidation");
    assert_eq!(persisted, acknowledged);
}

#[test]
fn acknowledge_invalidation_re_ack_returns_invalid_transition() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);

    assert_invalid_transition(
        store
            .with_principal(agent("agent-test"))
            .acknowledge_invalidation(fixture.invalidation.id),
        InvalidationStatus::Acknowledged,
        InvalidationStatus::Acknowledged,
    );

    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn acknowledge_invalidation_on_resolved_returns_invalid_transition() {
    let fixture = seed_resolved_fixture();
    let store = reopen(&fixture.config);

    assert_invalid_transition(
        store
            .with_principal(agent("agent-test"))
            .acknowledge_invalidation(fixture.invalidation.id),
        InvalidationStatus::Resolved,
        InvalidationStatus::Acknowledged,
    );

    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn acknowledge_invalidation_on_dismissed_returns_invalid_transition() {
    let fixture = seed_dismissed_fixture();
    let store = reopen(&fixture.config);

    assert_invalid_transition(
        store
            .with_principal(agent("agent-test"))
            .acknowledge_invalidation(fixture.invalidation.id),
        InvalidationStatus::Dismissed,
        InvalidationStatus::Acknowledged,
    );

    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn acknowledge_invalidation_unknown_id_returns_not_found() {
    let (_dir, _config, store) = fresh_store();
    let id = InvalidationId::new_v7();

    assert_not_found(
        store
            .with_principal(agent("agent-test"))
            .acknowledge_invalidation(id),
        id,
    );
    assert!(audit_rows(&store, AuditFilter::default()).is_empty());
}

#[test]
fn acknowledge_invalidation_succeeds_when_affected_decision_is_non_active() {
    for status in [DecisionStatus::Superseded, DecisionStatus::Archived] {
        let fixture = seed_pending_fixture();
        set_decision_status(&fixture.config, fixture.decision_id, status);
        let store = reopen(&fixture.config);
        let acknowledged = store
            .with_principal(agent("agent-test"))
            .acknowledge_invalidation(fixture.invalidation.id)
            .expect("acknowledge non-active decision invalidation");

        assert_ack_shape(&fixture.invalidation, &acknowledged);
    }
}

#[test]
fn resolve_invalidation_acknowledged_with_new_decision_succeeds() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let new_decision =
        create_serving_decision(&store, fixture.context_id, fixture.intention_id, "serving");

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve");

    assert_resolved_shape(
        &fixture.invalidation,
        &resolved,
        "new_decision_created",
        Some(new_decision.id),
    );
}

#[test]
fn resolve_invalidation_with_decision_not_serving_threatened_intention_refused() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let unrelated = create_unrelated_decision(&store, fixture.context_id, "unrelated");

    let result = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            Some(unrelated.id),
        );

    assert!(
        matches!(
            result,
            Err(CgError::ResolutionDecisionDoesNotServeIntention {
                invalidation_id,
                decision_id,
                ref expected_intention_ids
            }) if invalidation_id == fixture.invalidation.id
                && decision_id == unrelated.id
                && expected_intention_ids == &fixture.invalidation.threatened_intention_ids
        ),
        "expected decision-does-not-serve refusal, got {result:?}"
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_with_empty_threatened_intention_ids_refused() {
    let fixture = seed_fixture_with_options(
        "empty-threatened",
        DirectSeedOptions {
            threatened_intention_ids: Some(Vec::new()),
            ..Default::default()
        },
    );
    set_invalidation_status(
        &fixture.config,
        fixture.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    let before = find_invalidation(
        &reopen(&fixture.config),
        fixture.context_id,
        fixture.invalidation.id,
    );
    let store = reopen(&fixture.config);
    let decision = create_serving_decision(&store, fixture.context_id, fixture.intention_id, "any");

    let result = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            before.id,
            ResolveAction::NewDecisionCreated,
            Some(decision.id),
        );

    assert!(
        matches!(
            result,
            Err(CgError::ResolutionDecisionDoesNotServeIntention { .. })
        ),
        "expected empty-threatened refusal, got {result:?}"
    );
    let after = find_invalidation(&store, fixture.context_id, before.id);
    assert_refusal_preserved(&before, &after);
    assert_no_audit_rows_for(&store, before.id);
}

#[test]
fn resolve_invalidation_new_decision_in_other_context_returns_decision_not_found() {
    let fixture = seed_acknowledged_fixture();
    let admin = reopen(&fixture.config);
    let other = create_context(&admin, "tenant-beta");
    let foreign = create_serving_decision(
        &admin,
        other.id,
        create_intention(&admin, other.id, "foreign").id,
        "foreign",
    );
    drop(admin);
    let scoped = reopen_scoped(&fixture.config, [fixture.context_id]);

    assert_decision_not_found(
        scoped
            .with_principal(agent("agent-test"))
            .resolve_invalidation(
                fixture.invalidation.id,
                ResolveAction::NewDecisionCreated,
                Some(foreign.id),
            ),
        foreign.id,
    );

    let after = find_invalidation(
        &reopen(&fixture.config),
        fixture.context_id,
        fixture.invalidation.id,
    );
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&scoped, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_with_unknown_new_decision_id_returns_not_found() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let missing = DecisionId::new_v7();

    assert_decision_not_found(
        store
            .with_principal(agent("agent-test"))
            .resolve_invalidation(
                fixture.invalidation.id,
                ResolveAction::NewDecisionCreated,
                Some(missing),
            ),
        missing,
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_basis_still_valid_on_active_decision_succeeds() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let decision = store
        .get_decision(fixture.decision_id)
        .expect("get decision")
        .expect("decision");
    assert_eq!(decision.status, DecisionStatus::Active);

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            None,
        )
        .expect("resolve basis still valid");

    assert_resolved_shape(&fixture.invalidation, &resolved, "basis_still_valid", None);
}

#[test]
fn resolve_invalidation_basis_still_valid_on_superseded_decision_refused() {
    let fixture = seed_acknowledged_fixture();
    set_decision_status(
        &fixture.config,
        fixture.decision_id,
        DecisionStatus::Superseded,
    );
    let store = reopen(&fixture.config);

    assert!(
        matches!(
            store
                .with_principal(agent("agent-test"))
                .resolve_invalidation(
                    fixture.invalidation.id,
                    ResolveAction::BasisStillValid,
                    None,
                ),
            Err(CgError::BasisStillValidRequiresActiveDecision {
                current_status: DecisionStatus::Superseded,
                ..
            })
        ),
        "expected active-decision refusal"
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_basis_still_valid_on_archived_decision_refused() {
    let fixture = seed_acknowledged_fixture();
    set_decision_status(
        &fixture.config,
        fixture.decision_id,
        DecisionStatus::Archived,
    );
    let store = reopen(&fixture.config);

    assert!(
        matches!(
            store
                .with_principal(agent("agent-test"))
                .resolve_invalidation(
                    fixture.invalidation.id,
                    ResolveAction::BasisStillValid,
                    None,
                ),
            Err(CgError::BasisStillValidRequiresActiveDecision {
                current_status: DecisionStatus::Archived,
                ..
            })
        ),
        "expected active-decision refusal"
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_basis_still_valid_on_invalidated_decision_refused() {
    let fixture = seed_acknowledged_fixture();
    set_decision_status(
        &fixture.config,
        fixture.decision_id,
        DecisionStatus::Invalidated,
    );
    let store = reopen(&fixture.config);

    assert!(
        matches!(
            store
                .with_principal(agent("agent-test"))
                .resolve_invalidation(
                    fixture.invalidation.id,
                    ResolveAction::BasisStillValid,
                    None,
                ),
            Err(CgError::BasisStillValidRequiresActiveDecision {
                current_status: DecisionStatus::Invalidated,
                ..
            })
        ),
        "expected active-decision refusal"
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_new_decision_created_succeeds_on_non_active_affected_decision() {
    let fixture = seed_acknowledged_fixture();
    set_decision_status(
        &fixture.config,
        fixture.decision_id,
        DecisionStatus::Superseded,
    );
    let store = reopen(&fixture.config);
    let new_decision =
        create_serving_decision(&store, fixture.context_id, fixture.intention_id, "new");

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve non-active affected decision");

    assert_resolved_shape(
        &fixture.invalidation,
        &resolved,
        "new_decision_created",
        Some(new_decision.id),
    );
}

#[test]
fn resolve_invalidation_new_decision_created_without_new_decision_id_refused() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);

    let result = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            None,
        );

    assert!(
        matches!(result, Err(CgError::ResolveActionMissingDecision { invalidation_id }) if invalidation_id == fixture.invalidation.id),
        "expected missing-decision refusal, got {result:?}"
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_basis_still_valid_with_new_decision_id_refused() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let new_decision =
        create_serving_decision(&store, fixture.context_id, fixture.intention_id, "new");

    let result = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            Some(new_decision.id),
        );

    assert!(
        matches!(result, Err(CgError::ResolveActionUnexpectedDecision { invalidation_id }) if invalidation_id == fixture.invalidation.id),
        "expected unexpected-decision refusal, got {result:?}"
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_on_pending_row_refused() {
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);

    assert_invalid_transition(
        store
            .with_principal(agent("agent-test"))
            .resolve_invalidation(
                fixture.invalidation.id,
                ResolveAction::BasisStillValid,
                None,
            ),
        InvalidationStatus::Pending,
        InvalidationStatus::Resolved,
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn resolve_invalidation_unknown_id_returns_not_found() {
    let (_dir, _config, store) = fresh_store();
    let id = InvalidationId::new_v7();

    assert_not_found(
        store
            .with_principal(agent("agent-test"))
            .resolve_invalidation(id, ResolveAction::BasisStillValid, None),
        id,
    );
    assert!(audit_rows(&store, AuditFilter::default()).is_empty());
}

#[test]
fn dismiss_invalidation_acknowledged_with_reason_succeeds() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);

    let dismissed = store
        .with_principal(agent("pr-bot"))
        .dismiss_invalidation(fixture.invalidation.id, Some("not relevant".to_string()))
        .expect("dismiss");

    assert_dismissed_shape(&fixture.invalidation, &dismissed);
    assert_eq!(dismissed.reason, fixture.invalidation.reason);
}

#[test]
fn dismiss_invalidation_with_no_reason_succeeds() {
    for reason in [None, Some(String::new())] {
        let fixture = seed_acknowledged_fixture();
        let store = reopen(&fixture.config);

        let dismissed = store
            .with_principal(agent("agent-test"))
            .dismiss_invalidation(fixture.invalidation.id, reason)
            .expect("dismiss");

        assert_dismissed_shape(&fixture.invalidation, &dismissed);
    }
}

#[test]
fn dismiss_invalidation_on_pending_row_refused() {
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);

    assert_invalid_transition(
        store
            .with_principal(agent("agent-test"))
            .dismiss_invalidation(fixture.invalidation.id, None),
        InvalidationStatus::Pending,
        InvalidationStatus::Dismissed,
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn dismiss_invalidation_on_dismissed_row_refused() {
    let fixture = seed_dismissed_fixture();
    let store = reopen(&fixture.config);

    assert_invalid_transition(
        store
            .with_principal(agent("agent-test"))
            .dismiss_invalidation(fixture.invalidation.id, None),
        InvalidationStatus::Dismissed,
        InvalidationStatus::Dismissed,
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn dismiss_invalidation_on_resolved_row_refused() {
    let fixture = seed_resolved_fixture();
    let store = reopen(&fixture.config);

    assert_invalid_transition(
        store
            .with_principal(agent("agent-test"))
            .dismiss_invalidation(fixture.invalidation.id, None),
        InvalidationStatus::Resolved,
        InvalidationStatus::Dismissed,
    );
    let after = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&store, fixture.invalidation.id);
}

#[test]
fn dismiss_invalidation_unknown_id_returns_not_found() {
    let (_dir, _config, store) = fresh_store();
    let id = InvalidationId::new_v7();

    assert_not_found(
        store
            .with_principal(agent("agent-test"))
            .dismiss_invalidation(id, None),
        id,
    );
    assert!(audit_rows(&store, AuditFilter::default()).is_empty());
}

#[test]
fn intention_pointers_preserved_through_ack() {
    let first = IntentionId::new_v7();
    let second = IntentionId::new_v7();
    let fixture = seed_fixture_with_options(
        "intentions",
        DirectSeedOptions {
            threatened_intention_ids: Some(vec![first, second]),
            intention_now_un_served: true,
            ..Default::default()
        },
    );
    let store = reopen(&fixture.config);

    let acknowledged = store
        .with_principal(agent("agent-test"))
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("ack");

    assert_ack_shape(&fixture.invalidation, &acknowledged);
    assert_eq!(acknowledged.threatened_intention_ids, vec![first, second]);
    assert!(acknowledged.intention_now_un_served);
}

#[test]
fn intention_pointers_preserved_through_resolve() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let new_decision =
        create_serving_decision(&store, fixture.context_id, fixture.intention_id, "resolve");

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve");

    assert_eq!(
        resolved.threatened_intention_ids,
        fixture.invalidation.threatened_intention_ids
    );
    assert_eq!(
        resolved.intention_now_un_served,
        fixture.invalidation.intention_now_un_served
    );
    assert_eq!(resolved.cascade_from, fixture.invalidation.cascade_from);
}

#[test]
fn intention_pointers_preserved_through_dismiss() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);

    let dismissed = store
        .with_principal(agent("agent-test"))
        .dismiss_invalidation(fixture.invalidation.id, Some("duplicate".to_string()))
        .expect("dismiss");

    assert_eq!(
        dismissed.threatened_intention_ids,
        fixture.invalidation.threatened_intention_ids
    );
    assert_eq!(
        dismissed.intention_now_un_served,
        fixture.invalidation.intention_now_un_served
    );
    assert_eq!(dismissed.cascade_from, fixture.invalidation.cascade_from);
}

#[test]
fn cascade_from_preserved_with_some_value_through_full_lifecycle() {
    let root = seed_pending_fixture();
    let cascade = seed_invalidation_in_context_with_options(
        &root.config,
        root.context_id,
        "cascade",
        DirectSeedOptions {
            cascade_from: Some(root.decision_id),
            ..Default::default()
        },
    );
    set_invalidation_status(
        &root.config,
        cascade.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    let before = find_invalidation(
        &reopen(&root.config),
        root.context_id,
        cascade.invalidation.id,
    );
    let store = reopen(&root.config);
    let new_decision =
        create_serving_decision(&store, root.context_id, cascade.intention_id, "cascade-new");

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            before.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve cascade");

    assert_resolved_shape(
        &before,
        &resolved,
        "new_decision_created",
        Some(new_decision.id),
    );
    assert_eq!(resolved.cascade_from, Some(root.decision_id));
}

#[test]
fn cascade_row_lifecycled_independently_of_direct_row() {
    let direct = seed_pending_fixture();
    let b = seed_invalidation_in_context_with_options(
        &direct.config,
        direct.context_id,
        "cascade-b",
        DirectSeedOptions {
            cascade_from: Some(direct.decision_id),
            ..Default::default()
        },
    );
    let c = seed_invalidation_in_context_with_options(
        &direct.config,
        direct.context_id,
        "cascade-c",
        DirectSeedOptions {
            cascade_from: Some(direct.decision_id),
            ..Default::default()
        },
    );
    set_invalidation_status(
        &direct.config,
        direct.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    let before_direct = find_invalidation(
        &reopen(&direct.config),
        direct.context_id,
        direct.invalidation.id,
    );
    let store = reopen(&direct.config);
    let new_decision =
        create_serving_decision(&store, direct.context_id, direct.intention_id, "direct-new");

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            before_direct.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve direct");

    assert_eq!(resolved.status, InvalidationStatus::Resolved);
    assert_eq!(
        find_invalidation(&store, direct.context_id, b.invalidation.id).status,
        InvalidationStatus::Pending
    );
    assert_eq!(
        find_invalidation(&store, direct.context_id, c.invalidation.id).status,
        InvalidationStatus::Pending
    );
}

#[test]
fn cascade_row_dismissed_independently_records_per_row_audit_trail() {
    let direct = seed_pending_fixture();
    let cascade = seed_invalidation_in_context_with_options(
        &direct.config,
        direct.context_id,
        "cascade-dismiss",
        DirectSeedOptions {
            cascade_from: Some(direct.decision_id),
            ..Default::default()
        },
    );
    set_invalidation_status(
        &direct.config,
        cascade.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    let before = find_invalidation(
        &reopen(&direct.config),
        direct.context_id,
        cascade.invalidation.id,
    );
    let store = reopen(&direct.config);
    let dismissed = store
        .with_principal(agent("agent-test"))
        .dismiss_invalidation(before.id, Some("cascade not relevant".to_string()))
        .expect("dismiss cascade");

    assert_dismissed_shape(&before, &dismissed);
    let rows = audit_rows_for_invalidation(&store, cascade.invalidation.id);
    assert_eq!(rows.len(), 2);
    assert!(
        rows.iter()
            .all(|entry| entry.target == AuditTarget::Invalidation(cascade.invalidation.id))
    );
}

#[test]
fn cascade_row_resolved_independently_records_per_row_audit_trail() {
    let direct = seed_pending_fixture();
    let cascade = seed_invalidation_in_context_with_options(
        &direct.config,
        direct.context_id,
        "cascade-resolve",
        DirectSeedOptions {
            cascade_from: Some(direct.decision_id),
            ..Default::default()
        },
    );
    set_invalidation_status(
        &direct.config,
        cascade.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    let before = find_invalidation(
        &reopen(&direct.config),
        direct.context_id,
        cascade.invalidation.id,
    );
    let store = reopen(&direct.config);
    let new_decision = create_serving_decision(
        &store,
        direct.context_id,
        cascade.intention_id,
        "cascade-fix",
    );

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            before.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve cascade");

    assert_resolved_shape(
        &before,
        &resolved,
        "new_decision_created",
        Some(new_decision.id),
    );
    let rows = audit_rows_for_invalidation(&store, cascade.invalidation.id);
    assert_eq!(rows.len(), 2);
}

#[test]
fn lifecycle_op_on_invalidation_in_different_context_returns_not_found() {
    let fixture = seed_pending_fixture();
    let admin = reopen(&fixture.config);
    let other = create_context(&admin, "tenant-beta");
    drop(admin);
    let scoped = reopen_scoped(&fixture.config, [other.id]);
    let scoped = scoped.with_principal(agent("consultant"));

    assert_not_found(
        scoped.acknowledge_invalidation(fixture.invalidation.id),
        fixture.invalidation.id,
    );
    assert_not_found(
        scoped.resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            None,
        ),
        fixture.invalidation.id,
    );
    assert_not_found(
        scoped.dismiss_invalidation(fixture.invalidation.id, None),
        fixture.invalidation.id,
    );
    let admin = reopen(&fixture.config);
    let after = find_invalidation(&admin, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&admin, fixture.invalidation.id);
}

#[test]
fn every_lifecycle_op_enforces_context_isolation_property() {
    let fixture = seed_pending_fixture();
    let admin = reopen(&fixture.config);
    let other = create_context(&admin, "tenant-beta");
    drop(admin);
    let scoped = reopen_scoped(&fixture.config, [other.id]);
    let scoped = scoped.with_principal(agent("consultant"));

    assert_not_found(
        scoped.get_invalidation(fixture.invalidation.id),
        fixture.invalidation.id,
    );
    assert_not_found(
        scoped.acknowledge_invalidation(fixture.invalidation.id),
        fixture.invalidation.id,
    );
    assert_not_found(
        scoped.resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            None,
        ),
        fixture.invalidation.id,
    );
    assert_not_found(
        scoped.dismiss_invalidation(fixture.invalidation.id, None),
        fixture.invalidation.id,
    );
    let admin = reopen(&fixture.config);
    let after = find_invalidation(&admin, fixture.context_id, fixture.invalidation.id);
    assert_refusal_preserved(&fixture.invalidation, &after);
    assert_no_audit_rows_for(&admin, fixture.invalidation.id);
}

#[test]
fn consultant_same_principal_multi_context_handle_full_lifecycle_aggregates_on_principal_join() {
    let fixture = seed_pending_fixture();
    let admin = reopen(&fixture.config);
    let other = create_context(&admin, "client-b");
    drop(admin);
    let other_inv = seed_pending_invalidation_in_context(&fixture.config, other.id, "client-b");
    set_invalidation_status(
        &fixture.config,
        fixture.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    set_invalidation_status(
        &fixture.config,
        other_inv.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    let scoped = reopen_scoped(&fixture.config, [fixture.context_id, other.id]);
    let principal = Principal::Human("consultant@firm.com".to_string());
    let handle = scoped.with_principal(principal.clone());

    let left = handle
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            None,
        )
        .expect("resolve tenant a");
    let right = handle
        .dismiss_invalidation(
            other_inv.invalidation.id,
            Some("not applicable".to_string()),
        )
        .expect("dismiss tenant b");

    assert_eq!(left.status, InvalidationStatus::Resolved);
    assert_eq!(right.status, InvalidationStatus::Dismissed);
    let rows = audit_rows(
        &scoped,
        AuditFilter {
            actor: Some(principal),
            ..Default::default()
        },
    );
    assert_eq!(rows.len(), 4);
    let only_b = reopen_scoped(&fixture.config, [other.id]);
    assert_not_found(
        only_b
            .with_principal(Principal::Human("consultant@firm.com".to_string()))
            .get_invalidation(fixture.invalidation.id),
        fixture.invalidation.id,
    );
}

#[test]
fn webhook_bot_principal_records_bot_identity_not_human_operator() {
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);
    let principal = agent("whatsapp-bot");

    let acknowledged = store
        .with_principal(principal.clone())
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("ack");

    assert_ack_shape(&fixture.invalidation, &acknowledged);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].actor, principal);
}

#[test]
fn audit_log_replay_reconstructs_six_month_narrative_chain() {
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);
    let principal = agent("pr-bot");
    let scoped = store.with_principal(principal.clone());

    let acknowledged = scoped
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("ack");
    let dismissed = scoped
        .dismiss_invalidation(
            fixture.invalidation.id,
            Some("operator note: remediation moved elsewhere".to_string()),
        )
        .expect("dismiss");

    assert!(acknowledged.detected_at <= dismissed.detected_at);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].operation, AuditOperation::AcknowledgeInvalidation);
    assert_eq!(rows[1].operation, AuditOperation::DismissInvalidation);
    assert!(rows[0].txid < rows[1].txid);
    assert_eq!(rows[0].actor, principal);
    assert_eq!(rows[1].actor, principal);
}

#[test]
fn lifecycle_ops_identical_on_mode_one_and_mode_two_invalidations() {
    let mode_one = seed_pending_fixture();
    let mode_two = seed_fixture_with_options(
        "mode-two",
        DirectSeedOptions {
            mode_two_decision: true,
            ..Default::default()
        },
    );
    for fixture in [&mode_one, &mode_two] {
        set_invalidation_status(
            &fixture.config,
            fixture.invalidation.id,
            InvalidationStatus::Acknowledged,
            None,
            None,
        );
        let store = reopen(&fixture.config);
        let new_decision =
            create_serving_decision(&store, fixture.context_id, fixture.intention_id, "new");
        let resolved = store
            .with_principal(agent("agent-test"))
            .resolve_invalidation(
                fixture.invalidation.id,
                ResolveAction::NewDecisionCreated,
                Some(new_decision.id),
            )
            .expect("resolve");
        assert_eq!(resolved.status, InvalidationStatus::Resolved);
    }
    let mode_two_decision = reopen(&mode_two.config)
        .get_decision(mode_two.decision_id)
        .expect("get mode-two decision")
        .expect("mode-two decision");
    assert_eq!(mode_two_decision.agent_id.as_deref(), Some("system"));
}

#[test]
fn regression_get_invalidations_filter_query_still_works() {
    let fixture = seed_pending_fixture();
    let ack = seed_pending_invalidation_in_context(&fixture.config, fixture.context_id, "ack");
    let resolved =
        seed_pending_invalidation_in_context(&fixture.config, fixture.context_id, "resolved");
    set_invalidation_status(
        &fixture.config,
        ack.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    set_invalidation_status(
        &fixture.config,
        resolved.invalidation.id,
        InvalidationStatus::Resolved,
        Some("basis_still_valid"),
        None,
    );
    let store = reopen(&fixture.config);

    assert_eq!(
        store
            .list_invalidations(fixture.context_id, None, Some(InvalidationStatus::Pending))
            .expect("pending")
            .len(),
        1
    );
    assert_eq!(
        store
            .list_invalidations(
                fixture.context_id,
                None,
                Some(InvalidationStatus::Acknowledged)
            )
            .expect("ack")
            .len(),
        1
    );
    assert_eq!(
        store
            .list_invalidations(fixture.context_id, None, Some(InvalidationStatus::Resolved))
            .expect("resolved")
            .len(),
        1
    );
    assert_eq!(
        store
            .list_invalidations(fixture.context_id, None, None)
            .expect("all")
            .len(),
        3
    );

    let acknowledged = store
        .with_system_principal()
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("acknowledge pending row through lifecycle API");
    assert_ack_shape(&fixture.invalidation, &acknowledged);
    assert_eq!(
        store
            .list_invalidations(
                fixture.context_id,
                None,
                Some(InvalidationStatus::Acknowledged)
            )
            .expect("ack after lifecycle")
            .len(),
        2
    );
}

#[test]
fn regression_archive_intention_refuses_with_active_decisions_reference_post_helper_inlining() {
    let (_dir, config, store) = fresh_store();
    let context = create_context(&store, "archive-intention");
    let intention = create_intention(&store, context.id, "archive-intention");
    let decision = create_serving_decision(&store, context.id, intention.id, "active");

    let result = store.archive_intention(intention.id, "done".to_string());

    assert!(
        matches!(
            result,
            Err(CgError::ActiveDecisionsReference {
                intention_id,
                ref decision_ids
            }) if intention_id == intention.id
                && decision_ids.contains(&decision.id.to_string())
        ),
        "expected active-decision archive refusal, got {result:?}"
    );
    let still_active = store
        .get_intention(intention.id)
        .expect("get intention")
        .expect("intention");
    assert_eq!(still_active.status, IntentionStatus::Active);
    drop(store);

    let invalidation =
        seed_pending_invalidation_in_context(&config, context.id, "archive-regression-lifecycle");
    let store = reopen(&config);
    let acknowledged = store
        .with_system_principal()
        .acknowledge_invalidation(invalidation.invalidation.id)
        .expect("lifecycle remains usable after archive refusal");
    assert_ack_shape(&invalidation.invalidation, &acknowledged);
}

#[test]
fn regression_store_open_admin_path_continues_to_work_post_plan_8() {
    let fixture = seed_pending_fixture();
    let admin = reopen(&fixture.config);
    let other = create_context(&admin, "client-b");
    drop(admin);
    let other_inv = seed_pending_invalidation_in_context(&fixture.config, other.id, "client-b");

    let admin = reopen(&fixture.config);
    assert_eq!(
        find_invalidation(&admin, fixture.context_id, fixture.invalidation.id).id,
        fixture.invalidation.id
    );
    assert_eq!(
        find_invalidation(&admin, other.id, other_inv.invalidation.id).id,
        other_inv.invalidation.id
    );
    drop(admin);

    let scoped = reopen_scoped(&fixture.config, [fixture.context_id]);
    assert_eq!(
        scoped
            .list_invalidations(fixture.context_id, None, None)
            .expect("scoped own")
            .len(),
        1
    );
    assert!(
        scoped
            .list_invalidations(other.id, None, None)
            .expect("scoped other")
            .is_empty()
    );

    let acknowledged = scoped
        .with_system_principal()
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("scoped lifecycle op on visible row");
    assert_ack_shape(&fixture.invalidation, &acknowledged);
}

#[test]
fn regression_get_invalidations_filter_cross_context_returns_only_scoped_rows() {
    let fixture = seed_pending_fixture();
    let admin = reopen(&fixture.config);
    let other = create_context(&admin, "client-b");
    drop(admin);
    seed_pending_invalidation_in_context(&fixture.config, fixture.context_id, "client-a-2");
    seed_pending_invalidation_in_context(&fixture.config, other.id, "client-b-1");
    seed_pending_invalidation_in_context(&fixture.config, other.id, "client-b-2");

    let scoped_a = reopen_scoped(&fixture.config, [fixture.context_id]);
    let rows_a = scoped_a
        .list_invalidations(fixture.context_id, None, Some(InvalidationStatus::Pending))
        .expect("list a");
    assert_context_set(&rows_a, &[fixture.context_id], 2);

    let scoped_b = reopen_scoped(&fixture.config, [other.id]);
    let rows_b = scoped_b
        .list_invalidations(other.id, None, Some(InvalidationStatus::Pending))
        .expect("list b");
    assert_context_set(&rows_b, &[other.id], 2);

    assert_not_found(
        scoped_a
            .with_system_principal()
            .get_invalidation(rows_b[0].id),
        rows_b[0].id,
    );
}

#[test]
fn regression_invalidations_are_suggestions_not_actions_on_dismiss() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);

    let dismissed = store
        .with_principal(agent("agent-test"))
        .dismiss_invalidation(fixture.invalidation.id, None)
        .expect("dismiss");

    assert_dismissed_shape(&fixture.invalidation, &dismissed);
    let decision = store
        .get_decision(fixture.decision_id)
        .expect("get decision")
        .expect("decision");
    assert_eq!(decision.status, DecisionStatus::Active);
}

#[test]
fn regression_invalidations_are_suggestions_not_actions_on_resolve() {
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let new_decision =
        create_serving_decision(&store, fixture.context_id, fixture.intention_id, "new");

    let resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve");

    assert_resolved_shape(
        &fixture.invalidation,
        &resolved,
        "new_decision_created",
        Some(new_decision.id),
    );
    let decision = store
        .get_decision(fixture.decision_id)
        .expect("get decision")
        .expect("decision");
    assert_eq!(decision.status, DecisionStatus::Active);
}

#[test]
fn regression_invalidation_status_cannot_go_backward_full_sweep() {
    for terminal in [
        InvalidationStatus::Acknowledged,
        InvalidationStatus::Resolved,
        InvalidationStatus::Dismissed,
    ] {
        let fixture = match terminal {
            InvalidationStatus::Acknowledged => seed_acknowledged_fixture(),
            InvalidationStatus::Resolved => seed_resolved_fixture(),
            InvalidationStatus::Dismissed => seed_dismissed_fixture(),
            InvalidationStatus::Pending => continue,
        };
        let store = reopen(&fixture.config);
        assert_invalid_transition(
            store
                .with_principal(agent("agent-test"))
                .acknowledge_invalidation(fixture.invalidation.id),
            terminal,
            InvalidationStatus::Acknowledged,
        );
        if matches!(
            terminal,
            InvalidationStatus::Resolved | InvalidationStatus::Dismissed
        ) {
            assert_invalid_transition(
                store
                    .with_principal(agent("agent-test"))
                    .resolve_invalidation(
                        fixture.invalidation.id,
                        ResolveAction::BasisStillValid,
                        None,
                    ),
                terminal,
                InvalidationStatus::Resolved,
            );
            assert_invalid_transition(
                store
                    .with_principal(agent("agent-test"))
                    .dismiss_invalidation(fixture.invalidation.id, None),
                terminal,
                InvalidationStatus::Dismissed,
            );
        }
    }
}

#[test]
fn resolution_action_column_carries_three_canonical_strings_set_by_plan_8_ops() {
    let new_decision_fixture = seed_acknowledged_fixture();
    let store = reopen(&new_decision_fixture.config);
    let new_decision = create_serving_decision(
        &store,
        new_decision_fixture.context_id,
        new_decision_fixture.intention_id,
        "new",
    );
    let new_decision_resolved = store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            new_decision_fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve new decision");
    assert_eq!(
        new_decision_resolved.resolution_action.as_deref(),
        Some("new_decision_created")
    );

    let basis_fixture = seed_acknowledged_fixture();
    let basis_resolved = reopen(&basis_fixture.config)
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            basis_fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            None,
        )
        .expect("resolve basis");
    assert_eq!(
        basis_resolved.resolution_action.as_deref(),
        Some("basis_still_valid")
    );

    let dismiss_fixture = seed_acknowledged_fixture();
    let dismissed = reopen(&dismiss_fixture.config)
        .with_principal(agent("agent-test"))
        .dismiss_invalidation(dismiss_fixture.invalidation.id, None)
        .expect("dismiss");
    assert_eq!(
        dismissed.resolution_action.as_deref(),
        Some("dismissed_as_irrelevant")
    );
}
