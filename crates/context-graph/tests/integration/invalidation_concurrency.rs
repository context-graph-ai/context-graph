use std::sync::{Arc, Barrier};
use std::thread;

use context_graph::*;

use super::invalidation_fixtures::*;

fn agent(name: &str) -> Principal {
    Principal::Agent(name.to_string())
}

#[test]
fn concurrent_acknowledge_first_wins_second_returns_typed_error() {
    let fixture = seed_pending_fixture();
    let store = Arc::new(reopen(&fixture.config));
    let barrier = Arc::new(Barrier::new(2));

    let handles = ["agent-a", "agent-b"]
        .into_iter()
        .map(|name| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let invalidation_id = fixture.invalidation.id;
            thread::spawn(move || {
                barrier.wait();
                store
                    .with_principal(agent(name))
                    .acknowledge_invalidation(invalidation_id)
            })
        })
        .collect::<Vec<_>>();

    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("join ack worker"))
        .collect::<Vec<_>>();

    let successes = results
        .iter()
        .filter(
            |result| matches!(result, Ok(row) if row.status == InvalidationStatus::Acknowledged),
        )
        .count();
    let typed_losers = results
        .iter()
        .filter(|result| {
            matches!(
                result,
                Err(CgError::InvalidTransition {
                    from: InvalidationStatus::Acknowledged,
                    to: InvalidationStatus::Acknowledged
                })
            )
        })
        .count();
    assert_eq!(successes, 1, "results: {results:?}");
    assert_eq!(typed_losers, 1, "results: {results:?}");

    let persisted = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert_eq!(persisted.status, InvalidationStatus::Acknowledged);
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
}

#[test]
fn concurrent_resolve_and_dismiss_first_wins_second_typed_error() {
    let fixture = seed_acknowledged_fixture();
    let store = Arc::new(reopen(&fixture.config));
    let barrier = Arc::new(Barrier::new(2));

    let resolve_store = Arc::clone(&store);
    let resolve_barrier = Arc::clone(&barrier);
    let invalidation_id = fixture.invalidation.id;
    let resolve = thread::spawn(move || {
        resolve_barrier.wait();
        resolve_store
            .with_principal(agent("resolver"))
            .resolve_invalidation(invalidation_id, ResolveAction::BasisStillValid, None)
    });

    let dismiss_store = Arc::clone(&store);
    let dismiss_barrier = Arc::clone(&barrier);
    let dismiss = thread::spawn(move || {
        dismiss_barrier.wait();
        dismiss_store
            .with_principal(agent("dismissor"))
            .dismiss_invalidation(invalidation_id, None)
    });

    let results = vec![
        resolve.join().expect("join resolver"),
        dismiss.join().expect("join dismissor"),
    ];
    let terminal_successes = results
        .iter()
        .filter(|result| {
            matches!(
                result,
                Ok(row)
                    if matches!(
                        row.status,
                        InvalidationStatus::Resolved | InvalidationStatus::Dismissed
                    )
            )
        })
        .count();
    let typed_losers = results
        .iter()
        .filter(|result| {
            matches!(
                result,
                Err(CgError::InvalidTransition {
                    from: InvalidationStatus::Resolved | InvalidationStatus::Dismissed,
                    to: InvalidationStatus::Resolved | InvalidationStatus::Dismissed
                })
            )
        })
        .count();
    assert_eq!(terminal_successes, 1, "results: {results:?}");
    assert_eq!(typed_losers, 1, "results: {results:?}");

    let persisted = find_invalidation(&store, fixture.context_id, fixture.invalidation.id);
    assert!(matches!(
        persisted.status,
        InvalidationStatus::Resolved | InvalidationStatus::Dismissed
    ));
    let rows = audit_rows_for_invalidation(&store, fixture.invalidation.id);
    assert_eq!(rows.len(), 1);
}

#[test]
fn concurrent_basis_still_valid_and_archive_never_resolves_after_archive_commit() {
    for _ in 0..8 {
        let fixture = seed_acknowledged_fixture();
        let store = Arc::new(reopen(&fixture.config));
        let barrier = Arc::new(Barrier::new(2));

        let resolve_store = Arc::clone(&store);
        let resolve_barrier = Arc::clone(&barrier);
        let invalidation_id = fixture.invalidation.id;
        let resolve = thread::spawn(move || {
            resolve_barrier.wait();
            resolve_store
                .with_principal(agent("resolver"))
                .resolve_invalidation(invalidation_id, ResolveAction::BasisStillValid, None)
        });

        let archive_store = Arc::clone(&store);
        let archive_barrier = Arc::clone(&barrier);
        let decision_id = fixture.decision_id;
        let archive = thread::spawn(move || {
            archive_barrier.wait();
            archive_store.archive_decision(decision_id, "decision retired".to_string())
        });

        let resolve_result = resolve.join().expect("join resolver");
        let archive_result = archive.join().expect("join archiver");
        assert!(
            archive_result.is_ok(),
            "archive should remain a legal concurrent operation: {archive_result:?}"
        );
        let archive_rows = audit_rows(
            store.as_ref(),
            AuditFilter {
                target: Some(AuditTarget::Decision(fixture.decision_id)),
                ..Default::default()
            },
        )
        .into_iter()
        .filter(|entry| matches!(entry.operation, AuditOperation::Archive))
        .collect::<Vec<_>>();
        assert_eq!(archive_rows.len(), 1);

        match resolve_result {
            Ok(row) => {
                let resolved_at = row.resolved_at.expect("resolved txid");
                assert!(
                    resolved_at < archive_rows[0].txid,
                    "BasisStillValid must not commit after the affected decision was archived"
                );
            }
            Err(CgError::BasisStillValidRequiresActiveDecision {
                invalidation_id,
                decision_id,
                current_status: DecisionStatus::Archived,
            }) if invalidation_id == fixture.invalidation.id
                && decision_id == fixture.decision_id => {}
            other => panic!("unexpected resolve/archive race result: {other:?}"),
        }
    }
}
