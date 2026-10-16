use context_graph::*;

use super::invalidation_fixtures::*;

fn agent(name: &str) -> Principal {
    Principal::Agent(name.to_string())
}

#[test]
fn get_invalidation_emits_no_audit_log_row() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);
    let before = audit_rows(&store, AuditFilter::default()).len();

    for _ in 0..3 {
        let pulled = store
            .with_system_principal()
            .get_invalidation(fixture.invalidation.id)
            .expect("get invalidation");
        assert_eq!(pulled, fixture.invalidation);
    }

    let after = audit_rows(&store, AuditFilter::default()).len();
    assert_eq!(after, before);
}

#[test]
fn acknowledge_emits_audit_row_with_full_identity() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);
    let principal = agent("agent-test");

    let acknowledged = store
        .with_principal(principal.clone())
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("ack");

    assert_ack_shape(&fixture.invalidation, &acknowledged);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].txid > fixture.invalidation.detected_at);
    assert_eq!(rows[0].actor, principal);
    assert_eq!(rows[0].operation, AuditOperation::AcknowledgeInvalidation);
    assert_eq!(
        rows[0].target,
        AuditTarget::Invalidation(fixture.invalidation.id)
    );
    assert_eq!(rows[0].context_id, fixture.context_id);
    assert_eq!(rows[0].payload, None);

    let dismissed = store
        .with_principal(agent("dismissor"))
        .dismiss_invalidation(fixture.invalidation.id, None)
        .expect("dismiss after ack");
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].operation, AuditOperation::AcknowledgeInvalidation);
    assert_eq!(rows[1].operation, AuditOperation::DismissInvalidation);
    assert!(rows[0].txid < rows[1].txid);
    assert_eq!(rows[1].txid, dismissed.resolved_at.expect("dismissed txid"));
}

#[test]
fn human_principal_with_reserved_agent_prefix_round_trips_as_human() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);
    let principal = Principal::Human("agent:alice@example.com".to_string());

    store
        .with_principal(principal.clone())
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("ack");

    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].actor, principal);
}

#[test]
fn resolve_emits_audit_row_with_full_identity() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let principal = agent("agent-test");
    let new_decision =
        create_serving_decision(&store, fixture.context_id, fixture.intention_id, "new");

    let resolved = store
        .with_principal(principal.clone())
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
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_audit_identity(
        &rows[0],
        resolved.resolved_at.expect("resolved txid"),
        &principal,
        AuditOperation::ResolveInvalidation,
        fixture.invalidation.id,
        fixture.context_id,
        Some(AuditPayload::Resolution(ResolutionPayload {
            action: ResolveAction::NewDecisionCreated,
            resolution_decision_id: Some(new_decision.id),
        })),
    );
    assert_eq!(rows[0].txid, resolved.resolved_at.expect("resolved txid"));
}

#[test]
fn resolve_basis_still_valid_emits_audit_row_with_full_identity() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let principal = agent("agent-test");

    let resolved = store
        .with_principal(principal.clone())
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            None,
        )
        .expect("resolve");

    assert_resolved_shape(&fixture.invalidation, &resolved, "basis_still_valid", None);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_audit_identity(
        &rows[0],
        resolved.resolved_at.expect("resolved txid"),
        &principal,
        AuditOperation::ResolveInvalidation,
        fixture.invalidation.id,
        fixture.context_id,
        Some(AuditPayload::Resolution(ResolutionPayload {
            action: ResolveAction::BasisStillValid,
            resolution_decision_id: None,
        })),
    );
}

#[test]
fn dismiss_emits_audit_row_with_full_identity_and_reason() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let principal = agent("pr-bot");
    let reason = "abstraction removed, remediation moved".to_string();

    let dismissed = store
        .with_principal(principal.clone())
        .dismiss_invalidation(fixture.invalidation.id, Some(reason.clone()))
        .expect("dismiss");

    assert_dismissed_shape(&fixture.invalidation, &dismissed);
    assert_eq!(dismissed.reason, fixture.invalidation.reason);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_audit_identity(
        &rows[0],
        dismissed.resolved_at.expect("dismissed txid"),
        &principal,
        AuditOperation::DismissInvalidation,
        fixture.invalidation.id,
        fixture.context_id,
        Some(AuditPayload::Dismissal(DismissalPayload {
            reason: Some(reason),
        })),
    );
}

#[test]
fn lifecycle_actor_defaults_to_system_when_no_principal_scoped_for_ack() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);
    let scoped = store.with_system_principal();
    assert_eq!(scoped.principal(), &Principal::System);

    let acknowledged = scoped
        .acknowledge_invalidation(fixture.invalidation.id)
        .expect("ack");

    assert_ack_shape(&fixture.invalidation, &acknowledged);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].actor, Principal::System);
}

#[test]
fn lifecycle_actor_defaults_to_system_when_no_principal_scoped_for_resolve() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let scoped = store.with_system_principal();
    assert_eq!(scoped.principal(), &Principal::System);

    let resolved = scoped
        .resolve_invalidation(
            fixture.invalidation.id,
            ResolveAction::BasisStillValid,
            None,
        )
        .expect("resolve");

    assert_resolved_shape(&fixture.invalidation, &resolved, "basis_still_valid", None);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].actor, Principal::System);
}

#[test]
fn lifecycle_actor_defaults_to_system_when_no_principal_scoped_for_dismiss() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_acknowledged_fixture();
    let store = reopen(&fixture.config);
    let scoped = store.with_system_principal();
    assert_eq!(scoped.principal(), &Principal::System);

    let dismissed = scoped
        .dismiss_invalidation(fixture.invalidation.id, Some("not relevant".to_string()))
        .expect("dismiss");

    assert_dismissed_shape(&fixture.invalidation, &dismissed);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].actor, Principal::System);
}

#[test]
fn lifecycle_actor_defaults_to_system_when_no_principal_scoped_for_get() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = seed_pending_fixture();
    let store = reopen(&fixture.config);
    let scoped = store.with_system_principal();
    assert_eq!(scoped.principal(), &Principal::System);

    let pulled = scoped
        .get_invalidation(fixture.invalidation.id)
        .expect("get invalidation");

    assert_eq!(pulled, fixture.invalidation);
    assert!(audit_rows_for_invalidation(&store, fixture.invalidation.id).is_empty());
}

#[test]
fn decode_audit_entry_round_trips_resolution_payload() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let resolve_fixture = seed_acknowledged_fixture();
    let resolve_store = reopen(&resolve_fixture.config);
    let new_decision = create_serving_decision(
        &resolve_store,
        resolve_fixture.context_id,
        resolve_fixture.intention_id,
        "new",
    );

    let resolved = resolve_store
        .with_principal(agent("agent-test"))
        .resolve_invalidation(
            resolve_fixture.invalidation.id,
            ResolveAction::NewDecisionCreated,
            Some(new_decision.id),
        )
        .expect("resolve");

    assert_resolved_shape(
        &resolve_fixture.invalidation,
        &resolved,
        "new_decision_created",
        Some(new_decision.id),
    );
    let resolve_rows = audit_rows_for_invalidation(&resolve_store, resolve_fixture.invalidation.id);
    assert_eq!(resolve_rows.len(), 1);
    assert_eq!(
        resolve_rows[0].payload,
        Some(AuditPayload::Resolution(ResolutionPayload {
            action: ResolveAction::NewDecisionCreated,
            resolution_decision_id: Some(new_decision.id),
        }))
    );
    assert_eq!(resolve_rows[0].outcome, None);

    let dismiss_fixture = seed_acknowledged_fixture();
    let dismiss_store = reopen(&dismiss_fixture.config);
    let dismissed = dismiss_store
        .with_principal(agent("agent-test"))
        .dismiss_invalidation(
            dismiss_fixture.invalidation.id,
            Some("reason text".to_string()),
        )
        .expect("dismiss");

    assert_dismissed_shape(&dismiss_fixture.invalidation, &dismissed);
    let dismiss_rows = audit_rows_for_invalidation(&dismiss_store, dismiss_fixture.invalidation.id);
    assert_eq!(dismiss_rows.len(), 1);
    assert_eq!(
        dismiss_rows[0].payload,
        Some(AuditPayload::Dismissal(DismissalPayload {
            reason: Some("reason text".to_string()),
        }))
    );
    assert_eq!(dismiss_rows[0].outcome, None);
}
