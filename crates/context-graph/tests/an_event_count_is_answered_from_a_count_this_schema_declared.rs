//! Asking how much history a Context holds is a lookup, and the store says so.
//!
//! ## The promise
//!
//! `cg status` and the capture path ask two questions of the events table often enough that their
//! cost is the user's cost: how many events this Context holds, and how many of them came from one
//! source stream. Both are exact-prefix `COUNT(*)`s, and the store keeps a current total for each,
//! so the answer is read rather than counted however much history is there.
//!
//! ## What the store maintains, and what it does not
//!
//! contextdb maintains an index-prefix count only for a column list the table DECLARED; a table
//! that declares nothing maintains nothing and every `COUNT(*)` walks the index range instead. The
//! answer is identical either way — the difference is entirely what it costs, and how much of every
//! append is spent keeping totals nothing reads.
//!
//! So this schema declares exactly the two lists its reads ask for, on the one table they ask it
//! of, and declares that the two tables written beside it maintain none. The event archive is
//! append-heavy: a total kept for a list nothing reads is paid on every single captured event.
//!
//! ## What is proven here
//!
//! Each count the read route issues, over a real store with real captured history, against the
//! route the shipped commands actually read through — and what the store discloses about where
//! each answer came from.

#[path = "common/event_search_statement_fixture.rs"]
mod event_search_statement_fixture;

use std::collections::HashMap;

use context_graph::{ConsumerReader, event_history_reader_declaration};
use contextdb_core::{SortDirection, Value};
use contextdb_engine::maintained_counts::MaintainedCountDisclosure;

use crate::event_search_statement_fixture::{CONNECTOR, Fixture, SOURCE, fixture};

/// The two counts the reads ask of the events table, in the exact text they issue.
const CAPTURED_EVENTS_OF_ONE_CONTEXT: &str =
    "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id";
const EVENTS_OF_ONE_SOURCE_STREAM: &str = "SELECT COUNT(*) AS n FROM events WHERE context_id = \
     $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref";

/// The one count over the events table that no index leads with, so no declaration could serve it
/// and the store never offers one.
const EVENTS_OF_ONE_COVERAGE: &str =
    "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id AND coverage = $coverage";

/// The tables written beside every captured event. Nothing counts either of them by a prefix, so
/// every total kept for one would be a total nothing reads, paid on every append.
const COUNT_OF_ONE_CONTEXTS_ANCHORS: &str =
    "SELECT COUNT(*) AS n FROM event_anchors WHERE context_id = $context_id";

/// The graph-memory totals `cg status` reports for every Context it is asked about, in the exact
/// text it issues them, and which this proof can put rows behind through the public store.
const GRAPH_MEMORY_TOTALS_OF_ONE_CONTEXT: [&str; 3] = [
    "SELECT COUNT(*) AS n FROM intentions WHERE context_id = $context_id",
    "SELECT COUNT(*) AS n FROM decisions WHERE context_id = $context_id",
    "SELECT COUNT(*) AS n FROM observations WHERE context_id = $context_id",
];

/// The coverage-gap total status reports beside them. Rows reach it only through an interpretation
/// build, so this proof reads it empty and holds it to the one thing an empty table can show: that
/// the declaration is in force.
const COVERAGE_GAP_TOTAL_OF_ONE_CONTEXT: &str =
    "SELECT COUNT(*) AS n FROM adapter_coverage_gaps WHERE context_id = $context_id";

fn columns(names: &[&str]) -> Vec<(String, SortDirection)> {
    names
        .iter()
        .map(|name| ((*name).to_string(), SortDirection::Asc))
        .collect()
}

fn reader(fixture: &Fixture) -> ConsumerReader {
    ConsumerReader::open(&fixture.config.db_path, event_history_reader_declaration())
        .expect("open the read-only reader the shipped commands use")
}

fn disclosure(
    reader: &ConsumerReader,
    sql: &str,
    params: &HashMap<String, Value>,
) -> Option<MaintainedCountDisclosure> {
    let result = reader.execute(sql, params).expect("the count answers");
    result.trace.maintained_count.map(|boxed| *boxed)
}

fn context_params(fixture: &Fixture) -> HashMap<String, Value> {
    HashMap::from([(
        "context_id".to_string(),
        Value::Uuid(fixture.context_ids[0].as_uuid()),
    )])
}

#[test]
fn the_total_a_context_holds_is_read_from_a_maintained_count_rather_than_counted() {
    let fixture = fixture(2, false);
    let reader = reader(&fixture);

    assert_eq!(
        Some(MaintainedCountDisclosure::Used {
            table: "events".to_string(),
            columns: columns(&["context_id"]),
        }),
        disclosure(
            &reader,
            CAPTURED_EVENTS_OF_ONE_CONTEXT,
            &context_params(&fixture)
        ),
        "the count of one Context's captured events is the total this schema declares, so it is \
         read rather than counted however much history the Context holds"
    );
}

#[test]
fn the_total_one_source_stream_contributed_is_read_from_a_maintained_count_rather_than_counted() {
    let fixture = fixture(2, false);
    let reader = reader(&fixture);
    let mut params = context_params(&fixture);
    params.insert(
        "connector_id".to_string(),
        Value::Text(CONNECTOR.to_string()),
    );
    params.insert(
        "source_instance_ref".to_string(),
        Value::Text(SOURCE.to_string()),
    );

    assert_eq!(
        Some(MaintainedCountDisclosure::Used {
            table: "events".to_string(),
            columns: columns(&["context_id", "connector_id", "source_instance_ref"]),
        }),
        disclosure(&reader, EVENTS_OF_ONE_SOURCE_STREAM, &params),
        "status reports one line per source stream, so the per-stream total is the second list \
         this schema declares"
    );
}

#[test]
fn a_count_by_coverage_is_answered_exactly_and_offers_no_declaration_that_would_change_it() {
    let fixture = fixture(2, false);
    let reader = reader(&fixture);
    let mut params = context_params(&fixture);
    params.insert("coverage".to_string(), Value::Text("complete".to_string()));

    assert_eq!(
        None,
        disclosure(&reader, EVENTS_OF_ONE_COVERAGE, &params),
        "no index over the events table leads with (context_id, coverage), so no declaration \
         could ever serve this count and the store must not advise one"
    );
}

#[test]
fn the_tables_written_beside_an_event_keep_no_total_and_say_which_one_would_be_kept() {
    let fixture = fixture(2, false);
    let reader = reader(&fixture);

    let anchors = disclosure(
        &reader,
        COUNT_OF_ONE_CONTEXTS_ANCHORS,
        &context_params(&fixture),
    );
    assert_eq!(
        Some(MaintainedCountDisclosure::NotDeclared {
            table: "event_anchors".to_string(),
            columns: columns(&["context_id"]),
            already_declared: Vec::new(),
        }),
        anchors,
        "nothing reads a per-Context anchor total, so keeping one would be write cost paid on \
         every captured event for an answer nobody asks for"
    );
    assert_eq!(
        Some("ALTER TABLE event_anchors SET MAINTAIN COUNT (context_id)".to_string()),
        anchors
            .expect("the anchors count discloses its route")
            .advised_statement(),
        "a read that wanted the total anyway is told the exact statement that would keep one"
    );
}

#[test]
fn every_total_status_reports_for_a_context_is_read_rather_than_counted() {
    let fixture = fixture(2, false);
    let context_id = fixture.context_ids[0];
    seed_graph_memory(&fixture, context_id);
    let reader = reader(&fixture);
    let params = context_params(&fixture);

    for statement in GRAPH_MEMORY_TOTALS_OF_ONE_CONTEXT {
        assert_eq!(
            Some(MaintainedCountDisclosure::Used {
                table: table_of(statement).to_string(),
                columns: columns(&["context_id"]),
            }),
            disclosure(&reader, statement, &params),
            "`cg status` issues this total once per Context in the selection, so it is read from a \
             total this schema declares rather than walked however much memory the Context holds"
        );
    }
}

/// A table with no rows has no stored total to serve, so an empty one cannot show `Used`. What it
/// can show is the difference that matters: the engine consulted a declaration and found one.
/// `NotDeclared` here would mean status walks this table's index on every Context it reports.
#[test]
fn the_coverage_gap_total_status_reports_is_declared_even_on_a_context_that_has_none() {
    let fixture = fixture(2, false);
    let reader = reader(&fixture);

    assert_eq!(
        Some(MaintainedCountDisclosure::DeclaredNotServed {
            table: "adapter_coverage_gaps".to_string(),
            columns: columns(&["context_id"]),
        }),
        disclosure(
            &reader,
            COVERAGE_GAP_TOTAL_OF_ONE_CONTEXT,
            &context_params(&fixture)
        ),
        "the coverage-gap total status reports must be one this schema declares; a `NotDeclared` \
         here is status walking an index once per Context"
    );
}

/// A total and a sum in one statement, which is how the raw-chunk footprint asks its question.
const A_TOTAL_AND_A_SUM_TOGETHER: &str =
    "SELECT COUNT(*) AS n, SUM(seq) AS s FROM events WHERE context_id = $context_id";

/// No declaration can serve a `COUNT(*)` that shares its statement with another aggregate: the
/// statement projects two columns, so the route that reads a maintained total is not even reached
/// and the trace carries nothing to disclose.
///
/// This is why `cg_source_chunks` declares no total. Its one read is
/// `SELECT COUNT(*), SUM(byte_len) FROM cg_source_chunks WHERE context_id = ...`, so a total
/// declared on it could never be read — it would only be written, once per chunk of every
/// imported source, which is the cost the event tables' declarations exist to avoid.
#[test]
fn a_total_asked_for_alongside_another_aggregate_is_never_served_from_a_declared_one() {
    let fixture = fixture(2, false);
    let reader = reader(&fixture);

    assert_eq!(
        None,
        disclosure(
            &reader,
            A_TOTAL_AND_A_SUM_TOGETHER,
            &context_params(&fixture)
        ),
        "a statement that asks for a total and a sum together reads rows either way, so the \
         maintained total this table DOES declare cannot serve it and the store does not pretend \
         otherwise"
    );
}

/// The table under a `SELECT COUNT(*) ... FROM <table> WHERE ...` statement.
fn table_of(statement: &str) -> &str {
    statement
        .split(" FROM ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .expect("every total names its table")
}

/// One intention, one decision and one observation in this Context, through the public store, so
/// the totals status reports have something to count.
fn seed_graph_memory(fixture: &Fixture, context_id: context_graph::ContextId) {
    use context_graph::{
        CreateDecision, CreateEntity, CreateIntention, EntityType, EvidenceId, EvidenceKind,
        EvidenceProducer, EvidenceRef, ObservationId, RecordObservation, RetentionStatus, Store,
    };
    let store = Store::open(fixture.config.clone()).expect("reopen the fixture store to seed it");
    let intention = store
        .create_intention(CreateIntention {
            description: "keep status a lookup".into(),
            context_id,
            ..Default::default()
        })
        .expect("record an intention");
    store
        .create_decision(CreateDecision {
            decision_type: "policy".into(),
            description: "declare the totals status reads".into(),
            intention_ids: vec![intention.id],
            context_id,
            ..Default::default()
        })
        .expect("record a decision");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "status".into(),
            context_id,
            ..Default::default()
        })
        .expect("create an entity to observe");
    store
        .record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id: entity.id,
            context_id,
            observation_type: "measured".into(),
            source: "maintained-count-proof".into(),
            observed_at: chrono::DateTime::UNIX_EPOCH,
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                kind: EvidenceKind::Text,
                source_ref: "cas://sha256:0".into(),
                mime_type: Some("text/plain".into()),
                retention_status: RetentionStatus::RetainedExternal,
                producer: EvidenceProducer {
                    system: "maintained-count-proof".into(),
                    model_name: "none".into(),
                    model_version: "1".into(),
                    pipeline_version: "1".into(),
                },
                ..Default::default()
            }],
            observed_properties: Default::default(),
            state_delta: Default::default(),
            properties: Default::default(),
            embeddings: Vec::new(),
        })
        .expect("record an observation");
    drop(store);
}
