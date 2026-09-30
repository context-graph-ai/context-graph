//! Asking what a Context holds is a lookup, never a scan of its events.
//!
//! ## The promise
//!
//! Every captured-event total a read reports is maintained beside the events, in the same
//! transaction that wrote them. `cg status`, `cg find` and the projection's work assessment read
//! that maintained row. None of them counts the events table to answer, because counting is
//! `SELECT COUNT(*) FROM events WHERE context_id = …` five times over for every Context in the
//! selection -- measured at roughly 128 ms PER CONTEXT -- and an operator asking what their memory
//! holds is entitled to a lookup.
//!
//! ## The silent path this closes
//!
//! A read-only consumer whose declaration leaves the counts table out used to fall through to that
//! recount without saying so: the same command, the same answer, a hundredfold the cost, and
//! nothing in the output to tell anyone. It is now refused by the table's own name, which is what
//! every other undeclared read in this crate does, so a consumer fixes its declaration instead of
//! silently paying.
//!
//! ## What is measured
//!
//! Statements, never milliseconds. Every statement the event-history reads issue goes through one
//! place, and this file asserts that not one of them counts rows.

#[path = "common/event_search_statement_fixture.rs"]
mod event_search_statement_fixture;

use context_graph::test_support::{
    record_event_history_statements_for_test, take_recorded_event_history_statements_for_test,
};
use context_graph::{
    CgError, ConsumerReader, ContextSelector, EventSearchFilter, EventSearchProfile,
    EventSearchWorkSpec, event_history_reader_declaration,
};

use crate::event_search_statement_fixture::{Fixture, fixture};

/// The table whose maintained row makes a recount unnecessary.
const COUNTS_TABLE: &str = "event_context_counts";

/// How many Contexts each read is asked about. More than one, so a per-Context recount would show
/// up as a multiple rather than as a single statement.
const CONTEXTS: usize = 4;

/// The exact statements a recount of one Context's captured events issues: the total, and one per
/// coverage value. Five per Context, measured at roughly 128 ms per Context, and every one of them
/// is what the maintained count row exists to make unnecessary.
const RECOUNT_STATEMENTS: [&str; 2] = [
    "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
    "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id AND coverage = $coverage",
];

fn assert_no_statement_recounts_captured_events(statements: &[String], path: &str) {
    let counting = statements
        .iter()
        .filter(|statement| RECOUNT_STATEMENTS.contains(&statement.as_str()))
        .collect::<Vec<_>>();
    assert!(
        counting.is_empty(),
        "{path} recounted a Context's captured events from the events table, which the \
         maintained count row exists to make unnecessary: {counting:#?}"
    );
}

fn selector(fixture: &Fixture) -> ContextSelector {
    ContextSelector::new(fixture.context_ids.iter().copied())
        .expect("a non-empty Context selection")
}

fn reader(fixture: &Fixture) -> ConsumerReader {
    let mut reader =
        ConsumerReader::open(&fixture.config.db_path, event_history_reader_declaration())
            .expect("open the read-only reader the shipped commands use");
    reader
        .attach_event_search_embedder_registrations(vec![fixture.registration.clone()])
        .expect("attach the event-search embedder to the reader");
    reader
}

fn work_spec(fixture: &Fixture) -> EventSearchWorkSpec {
    let space = fixture.registration.space.clone();
    EventSearchWorkSpec {
        profile: EventSearchProfile::default(),
        encoding_profile_id: space.embedding_space_id.clone(),
        embedding_space_id: space.embedding_space_id,
    }
}

#[test]
fn status_find_and_the_work_assessment_answer_without_counting_a_single_event() {
    let fixture = fixture(CONTEXTS, true);
    let reader = reader(&fixture);

    for (path, run) in [
        (
            "cg status",
            Box::new(|| {
                let _ = reader.event_history_status(selector(&fixture));
            }) as Box<dyn Fn()>,
        ),
        (
            "cg find",
            Box::new(|| {
                let _ = reader.find_event_history(
                    selector(&fixture),
                    "why the broker-free path?",
                    EventSearchFilter::default(),
                );
            }),
        ),
        (
            "the projection work assessment",
            Box::new(|| {
                let _ = reader.event_search_work(selector(&fixture), &work_spec(&fixture));
            }),
        ),
    ] {
        record_event_history_statements_for_test();
        run();
        let statements = take_recorded_event_history_statements_for_test();
        assert!(
            !statements.is_empty(),
            "{path} issued no statement at all, so this proof would pass without reading anything"
        );
        assert_no_statement_recounts_captured_events(&statements, path);
    }
}

#[test]
fn a_reader_that_left_the_counts_table_out_is_refused_by_that_table_name() {
    let fixture = fixture(CONTEXTS, true);
    let mut declaration = event_history_reader_declaration();
    declaration
        .schema
        .tables
        .retain(|table| table.name != COUNTS_TABLE);
    let mut reader = ConsumerReader::open(&fixture.config.db_path, declaration)
        .expect("a declaration that leaves one table out still opens");
    reader
        .attach_event_search_embedder_registrations(vec![fixture.registration.clone()])
        .expect("attach the event-search embedder to the reader");

    record_event_history_statements_for_test();
    let refused = reader
        .event_history_status(selector(&fixture))
        .expect_err("a reader that cannot read the counts table is refused, never quietly slower");
    let statements = take_recorded_event_history_statements_for_test();

    assert!(
        matches!(&refused, CgError::UndeclaredTableRefused { table } if table == COUNTS_TABLE),
        "the refusal names the table the consumer must declare, so it can fix its declaration \
         rather than guess: {refused:?}"
    );
    assert_no_statement_recounts_captured_events(&statements, "a reader without the counts table");
}
