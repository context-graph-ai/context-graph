//! Declaring a maintained total on a table costs the tables it is NOT on exactly nothing.
//!
//! ## The promise
//!
//! Capture is the hot path: every event a user imports goes through one append. Whatever this
//! schema declares elsewhere — the graph-memory totals `cg status` reports, the coverage-gap
//! total, the raw-chunk footprint — the per-event append writes only its declared capture tables.
//!
//! ## Why this is measured and not reasoned about
//!
//! The engine maintains a declared total on every write that touches that table's indexed columns.
//! "The capture path does not write those tables" is the whole argument for the declarations being
//! free, and an argument is not a measurement: a later change that made the append touch one of
//! them would make every one of those declarations a per-event cost, silently. So this counts the
//! index postings and prefix counts one real append actually maintains, through the engine's own
//! append-work instrument, and names every table it is allowed to touch.

use std::collections::BTreeSet;

use context_graph::{
    AppendEventBatch, CreateContext, EventAdapterDeclaration, EventCoverage, EventDraft,
    EventRenderRole, Store, StoreConfig,
};
use contextdb_engine::append_work;
use serde_json::json;

/// The tables one captured event is entitled to write. Everything else this schema holds is
/// written by another path -- recording a decision, building an interpretation, chunking a source
/// -- and a total declared on one of them is not paid per event.
const TABLES_ONE_APPEND_MAY_WRITE: [&str; 10] = [
    "events",
    "event_anchors",
    "event_local_commits",
    "event_context_counts",
    "event_context_sessions",
    "event_context_revisions",
    "event_placement",
    "event_source_locators",
    "event_source_ordering",
    "event_source_checkpoints",
];

/// The tables whose `(context_id)` total this schema declares for a read, none of which the
/// capture path writes.
const TABLES_DECLARED_FOR_A_READ: [&str; 4] = [
    "intentions",
    "decisions",
    "observations",
    "adapter_coverage_gaps",
];

fn adapter() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: "append-cost".into(),
        adapter_version: "v1".into(),
        anchor_declaration_version: "v1".into(),
        anchor_schema: json!({"type":"object","additionalProperties":true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(ordinal: i64) -> EventDraft {
    EventDraft {
        session_ref: "session-append-cost".into(),
        source_event_key: format!("event-{ordinal}"),
        source_position: json!({ "ordinal": ordinal }),
        occurred_at: None,
        author: "human".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("captured event {ordinal}")),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

#[test]
fn one_captured_event_maintains_no_total_this_schema_declared_for_a_read() {
    let root = tempfile::tempdir().expect("isolated store root");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("append-cost.db"),
        default_text_embedder: None,
        ..StoreConfig::default()
    })
    .expect("open the store");
    let context_id = store
        .create_context(CreateContext {
            name: "append-cost".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create the Context")
        .id;

    // The first append installs what a first append installs; the one that follows it is the
    // per-event cost this measures.
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            adapter(),
            "append-cost-source",
            vec![draft(1)],
        ))
        .expect("the first append");

    append_work::reset();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            adapter(),
            "append-cost-source",
            vec![draft(2)],
        ))
        .expect("the measured append");
    let postings = append_work::take_index_postings();
    let counts = append_work::take_index_prefix_counts();

    let written: BTreeSet<&str> = postings
        .iter()
        .map(|touch| touch.table.as_str())
        .chain(counts.iter().map(|touch| touch.table.as_str()))
        .collect();

    for declared in TABLES_DECLARED_FOR_A_READ {
        assert!(
            !written.contains(declared),
            "capture wrote `{declared}`, so the total this schema declares on it for a read is \
             paid on every captured event after all -- the declaration has to be re-argued, not \
             kept. Tables this append wrote: {written:?}"
        );
    }
    let allowed: BTreeSet<&str> = TABLES_ONE_APPEND_MAY_WRITE.into_iter().collect();
    let unexpected = written.difference(&allowed).collect::<Vec<_>>();
    assert!(
        unexpected.is_empty(),
        "one captured event wrote a table this proof does not know about, so what a declaration \
         costs the capture path is no longer settled: {unexpected:?}"
    );
}

#[test]
fn the_only_totals_one_captured_event_maintains_are_the_two_declared_on_events() {
    let root = tempfile::tempdir().expect("isolated store root");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("append-counts.db"),
        default_text_embedder: None,
        ..StoreConfig::default()
    })
    .expect("open the store");
    let context_id = store
        .create_context(CreateContext {
            name: "append-counts".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create the Context")
        .id;
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            adapter(),
            "append-counts-source",
            vec![draft(1)],
        ))
        .expect("the first append");

    append_work::reset();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            adapter(),
            "append-counts-source",
            vec![draft(2)],
        ))
        .expect("the measured append");
    let mut maintained = append_work::take_index_prefix_counts()
        .into_iter()
        .map(|touch| (touch.table, touch.prefix_columns))
        .collect::<Vec<_>>();
    maintained.sort();

    assert_eq!(
        vec![
            (
                "events".to_string(),
                vec![
                    "context_id".to_string(),
                    "connector_id".to_string(),
                    "source_instance_ref".to_string()
                ]
            ),
            ("events".to_string(), vec!["context_id".to_string()]),
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>(),
        maintained
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>(),
        "one captured event keeps exactly the two totals the event reads ask for and nothing else; \
         any third one here is write cost for an answer nobody asked for"
    );
}
