//! Context erasure removes durable search-resume state without touching another Context.
use super::*;
use context_graph::ForgetSelection;
use context_graph::test_support::{set_projection_unit_cut, take_projection_unit_receipts};

fn resume_rows(store: &Store, context: ContextId, table: &str) -> Vec<Vec<Value>> {
    let order = match table {
        "event_search_fold_progress" => "generation_id, fold",
        "event_search_capture_manifest" => {
            "generation_id, connector_id, source_instance_ref, session_ref"
        }
        _ => panic!("unexpected resume table"),
    };
    store
        .execute_sql_for_test(
            &format!("SELECT * FROM {table} WHERE context_id = $context_id ORDER BY {order}"),
            &HashMap::from([("context_id".into(), Value::Uuid(context.as_uuid()))]),
        )
        .unwrap()
        .rows()
        .to_vec()
}

#[test]
fn forgetting_a_context_erases_its_partial_search_fold_and_capture_manifest() {
    let _clock = Wallclock::test_clock_guard(|| 1_800_000_000_000);
    let root = tempfile::tempdir().unwrap();
    let cfg = config(root.path().join("forget-search-resume.db"));
    let store = open(&cfg);
    let forgotten = create_context_named(&store, "forgotten");
    let kept = create_context_named(&store, "kept");
    for context in [forgotten, kept] {
        let events = (1..=6)
            .map(|ordinal| {
                draft(
                    &format!("captured-{ordinal}"),
                    ordinal,
                    EventRenderRole::HumanMessage,
                    &format!("captured search-resume event {ordinal}"),
                )
            })
            .collect();
        store
            .append_event_batch(AppendEventBatch::live(
                context,
                declaration(),
                "source",
                events,
            ))
            .unwrap();
    }
    drop(store);
    let embedder = ControlledEmbedder::new();
    for context in [forgotten, kept] {
        take_projection_unit_receipts();
        // The first durable unit stores the frozen manifest and its fold
        // progress. Stop there, before capture consumes the manifest.
        set_projection_unit_cut(Some(1));
        let partial = run(&cfg, context, &embedder, 8, OBSERVED_LEASE_MS).unwrap();
        set_projection_unit_cut(None);
        assert!(
            partial.completed_at.is_none(),
            "the projection is deliberately incomplete"
        );
        let units = take_projection_unit_receipts();
        assert_eq!(
            units.iter().filter(|unit| unit.durable).count(),
            1,
            "exactly one real projection unit committed: {units:?}"
        );
    }
    let tables = [
        "event_search_fold_progress",
        "event_search_capture_manifest",
    ];
    let store = open(&cfg);
    let before: Vec<_> = tables
        .iter()
        .map(|table| {
            let gone = resume_rows(&store, forgotten, table);
            let retained = resume_rows(&store, kept, table);
            assert!(
                !gone.is_empty() && !retained.is_empty(),
                "both Contexts must have real durable resume rows in {table}"
            );
            eprintln!(
                "durable resume witnesses {table}: forgotten={} retained={}",
                gone.len(),
                retained.len()
            );
            retained
        })
        .collect();
    let receipt = store
        .forget_event_history(ForgetSelection::Context {
            context_id: forgotten,
        })
        .unwrap();
    assert!(receipt.changed);
    assert_eq!(receipt.events, 6);
    let mut leftovers = Vec::new();
    for (table, expected) in tables.iter().zip(&before) {
        leftovers.push((
            format!("{table}: immediate"),
            resume_rows(&store, forgotten, table),
        ));
        assert_eq!(
            resume_rows(&store, kept, table),
            *expected,
            "forgetting one Context keeps the other Context's {table}"
        );
    }
    drop(store);
    let reopened = open(&cfg);
    for (table, expected) in tables.iter().zip(&before) {
        leftovers.push((
            format!("{table}: reopened"),
            resume_rows(&reopened, forgotten, table),
        ));
        assert_eq!(
            resume_rows(&reopened, kept, table),
            *expected,
            "the other Context's {table} remains after reopen"
        );
    }
    assert!(
        leftovers.iter().all(|(_, rows)| rows.is_empty()),
        "Context forget must durably erase its fold progress and frozen manifest: {leftovers:?}"
    );
}
