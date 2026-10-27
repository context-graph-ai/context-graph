use std::collections::{BTreeSet, HashMap};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use contextdb_core::{Value, types::ContextId};
use contextdb_engine::{Database, SinkError, SinkEvent};
use tempfile::TempDir;
use uuid::Uuid;

fn empty() -> HashMap<String, Value> {
    HashMap::new()
}

fn declare_event_schema(db: &Database) {
    db.execute(
        "CREATE TABLE invalidations (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID, severity TEXT, reason TEXT)",
        &empty(),
    )
    .expect("create table");
    db.execute(
        "CREATE EVENT TYPE inv_match WHEN INSERT ON invalidations",
        &empty(),
    )
    .expect("create event type");
    db.execute("CREATE SINK slack TYPE callback", &empty())
        .expect("create sink");
    db.execute(
        "CREATE ROUTE inv_to_slack EVENT inv_match TO slack",
        &empty(),
    )
    .expect("create route");
}

fn insert_invalidation(db: &Database, context_id: Uuid, severity: &str, reason: &str) {
    db.execute(
        "INSERT INTO invalidations (id, context_id, severity, reason) VALUES ($id, $ctx, $sev, $rsn)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(Uuid::new_v4())),
            ("ctx".to_string(), Value::Uuid(context_id)),
            ("sev".to_string(), Value::Text(severity.to_string())),
            ("rsn".to_string(), Value::Text(reason.to_string())),
        ]),
    )
    .expect("insert invalidation");
}

fn wait_for_count(counter: &(Mutex<u64>, Condvar), expected: u64) {
    let guard = counter.0.lock().expect("delivery counter");
    let _guard = counter
        .1
        .wait_while(guard, |count| *count < expected)
        .expect("delivery event");
}

fn wait_for_log(log: &Arc<(Mutex<Vec<SinkEvent>>, Condvar)>, expected: usize) {
    let guard = log.0.lock().expect("sink log");
    let _guard = log
        .1
        .wait_while(guard, |events| events.len() < expected)
        .expect("sink event");
}

fn wait_for_metric(db: &Database, sink: &str, predicate: impl Fn(u64, u64, u64, u64) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        let metrics = db.sink_metrics_for_test(sink);
        if predicate(
            metrics.delivered,
            metrics.queued,
            metrics.retried,
            metrics.permanent_failures,
        ) {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

// Use case: queued sink events survive reopening and resume once the subscriber is registered again.
#[test]
fn engine_sink_queue_persists_across_reopen_subscriber_re_registers_resumes_delivery() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("event-bus.redb");
    let ctx = Uuid::new_v4();

    {
        let db = Database::open(&path).expect("open database");
        declare_event_schema(&db);
        insert_invalidation(&db, ctx, "warning", "queued-before-subscriber");
    }

    let db = Database::open(&path).expect("reopen database");
    let log = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
    let seen = Arc::clone(&log);
    db.register_sink("slack", None, move |event| {
        seen.0.lock().expect("sink log").push(event.clone());
        seen.1.notify_all();
        Ok(())
    })
    .expect("register sink");

    wait_for_log(&log, 1);
    let captured = log.0.lock().expect("sink log");
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0].row_values.get("reason"),
        Some(&Value::Text("queued-before-subscriber".to_string()))
    );
}

// Use case: permanent delivery failures are acknowledged once and counted distinctly from retries.
#[test]
fn engine_callback_returning_sink_error_permanent_drops_event_without_retry_metrics_distinguish() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let db = Database::open_memory();
    let ctx = Uuid::new_v4();
    declare_event_schema(&db);
    db.register_sink("slack", None, |_| {
        Err(SinkError::Permanent("expected failure".to_string()))
    })
    .expect("register sink");

    insert_invalidation(&db, ctx, "warning", "permanent");
    wait_for_metric(&db, "slack", |_, _, _, permanent| permanent == 1);

    let metrics = db.sink_metrics_for_test("slack");
    assert_eq!(metrics.delivered, 0);
    assert_eq!(metrics.retried, 0);
    assert_eq!(metrics.permanent_failures, 1);
}

// Use case: a failing sink cannot block an independent sink under sustained delivery.
#[test]
fn engine_per_sink_queue_isolation_one_sink_panic_does_not_block_other_sink_delivery_at_scale() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let db = Database::open_memory();
    let ctx = Uuid::new_v4();
    declare_event_schema(&db);
    db.execute("CREATE SINK audit TYPE callback", &empty())
        .expect("create audit sink");
    db.execute(
        "CREATE ROUTE inv_to_audit EVENT inv_match TO audit",
        &empty(),
    )
    .expect("create audit route");

    db.register_sink("slack", None, |_| panic!("expected sink panic"))
        .expect("register panic sink");
    let audit_count = Arc::new((Mutex::new(0), Condvar::new()));
    let seen = Arc::clone(&audit_count);
    db.register_sink("audit", None, move |_| {
        *seen.0.lock().expect("delivery counter") += 1;
        seen.1.notify_all();
        Ok(())
    })
    .expect("register audit sink");

    for n in 0..50 {
        insert_invalidation(&db, ctx, "warning", &format!("event-{n}"));
    }

    wait_for_count(&audit_count, 50);
    wait_for_metric(&db, "audit", |delivered, _, _, permanent| {
        delivered == 50 && permanent == 0
    });
    wait_for_metric(&db, "slack", |_, queued, retried, permanent| {
        queued == 50 && retried > 0 && permanent == 0
    });

    assert_eq!(*audit_count.0.lock().expect("delivery counter"), 50);
    let audit_metrics = db.sink_metrics_for_test("audit");
    assert_eq!(audit_metrics.delivered, 50);
    assert_eq!(audit_metrics.permanent_failures, 0);
    let panic_metrics = db.sink_metrics_for_test("slack");
    assert!(
        panic_metrics.uncertain > 0 && panic_metrics.queued == 50,
        "panic sink should retain uncertain actions for retry, got {panic_metrics:?}"
    );
}

// Use case: cross-context events are filtered by the engine before typed payload handling.
#[test]
fn cross_context_isolation_enforced_at_engine_route_not_at_typed_wrapper() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let db = Database::open_memory();
    let ctx_a = Uuid::new_v4();
    let ctx_b = Uuid::new_v4();
    declare_event_schema(&db);

    let scoped = db.scoped_with_contexts(BTreeSet::from([ContextId::new(ctx_a)]));
    let log = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
    let seen = Arc::clone(&log);
    scoped
        .register_sink("slack", None, move |event| {
            seen.0.lock().expect("sink log").push(event.clone());
            seen.1.notify_all();
            Ok(())
        })
        .expect("register scoped sink");

    insert_invalidation(&db, ctx_a, "warning", "allowed");
    insert_invalidation(&db, ctx_b, "warning", "foreign");

    wait_for_log(&log, 1);
    let captured = log.0.lock().expect("sink log");
    assert_eq!(
        captured.len(),
        1,
        "scoped sink should receive the allowed context and reject the foreign context"
    );
    assert_eq!(
        captured[0].row_values.get("reason"),
        Some(&Value::Text("allowed".to_string()))
    );
}

// Use case: sink metrics expose route-side filtering using shipped dispatcher counters only.
#[test]
fn engine_sink_metrics_for_test_distinguishes_route_filter_from_dispatcher_filter() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let db = Database::open_memory();
    let ctx_a = Uuid::new_v4();
    let ctx_b = Uuid::new_v4();
    declare_event_schema(&db);

    let delivered = Arc::new(AtomicU64::new(0));
    let seen = Arc::clone(&delivered);
    let scoped = db.scoped_with_contexts(BTreeSet::from([ContextId::new(ctx_a)]));
    scoped
        .register_sink("slack", None, move |_| {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .expect("register scoped sink");

    for n in 0..3 {
        insert_invalidation(&db, ctx_b, "warning", &format!("foreign-{n}"));
    }

    std::thread::sleep(Duration::from_millis(250));
    let metrics = db.sink_metrics_for_test("slack");
    assert_eq!(delivered.load(Ordering::SeqCst), 0);
    assert_eq!(metrics.delivered, 0);
    assert_eq!(metrics.retried, 0);
    assert_eq!(metrics.permanent_failures, 0);
}
