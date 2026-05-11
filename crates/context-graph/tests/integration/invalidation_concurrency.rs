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
