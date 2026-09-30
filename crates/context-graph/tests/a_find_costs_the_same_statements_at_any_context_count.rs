//! One `cg find` costs the same number of statements over 64 Contexts as over one.
//!
//! ## The promise
//!
//! A find is answered from a maintained search index, so widening it from one Context to sixty-four
//! is supposed to widen the SCOPE of one search, not multiply the reads. An agent asking its memory
//! a question across its whole workspace waits for a search, not for sixty-four round trips of
//! bookkeeping. Every statement a find issues before it ever embeds the question is set-shaped:
//! it names the whole selected Context set at once.
//!
//! ## What is measured
//!
//! Statements, never milliseconds. `context_graph::test_support::record_event_history_statements_for_test`
//! records every statement the event-history reads issue on this thread — the one place they all go
//! through — and the proof is that the recorded COUNT is identical at 1, 8 and 64 selected
//! Contexts. A per-Context read shows up as a count that grows with the Context count and cannot
//! hide behind a fast machine.
//!
//! Both shapes a find meets are covered, because they are different code paths: Contexts whose
//! projection is ready (the find searches them) and Contexts whose projection is still owed (the
//! find refuses and reports coverage). Coverage bookkeeping is reported in both.
//!
//! One read in a find IS allowed to repeat: the source links behind each returned passage are read
//! per hit, and the hit count is bounded by the caller's own limit (10 by default, 100 at most) and
//! not by how many Contexts were selected. So this file separates the two and pins both: the source
//! link read appears exactly once per hit, and EVERY OTHER statement is issued at most ONCE per
//! find, at 1 Context and at 64 alike.
//!
//! "At most once each" is the contract itself rather than a total, and that is deliberate. A total
//! also moves when an optional read is skipped -- a find over Contexts with nothing ready never
//! issues the coverage reads at all -- and a proof that fails on that is a proof that fails on a
//! busy machine for no defect. A statement issued twice in one find, on the other hand, is the
//! defect: it is what a read per Context looks like, because the per-Context reads all carried the
//! same statement text and differed only in the Context bound into it.

#[path = "common/event_search_statement_fixture.rs"]
mod event_search_statement_fixture;

use std::collections::HashMap;

use context_graph::test_support::{
    record_event_history_statements_for_test, take_recorded_event_history_statements_for_test,
};
use context_graph::{
    ConsumerReader, ContextSelector, EventSearchFilter, event_history_reader_declaration,
};

use crate::event_search_statement_fixture::{Fixture, fixture};

/// The Context counts the delivery matrix is frozen at.
const CONTEXT_COUNTS: [usize; 3] = [1, 8, 64];

/// The per-hit read: the source links behind one returned passage. It repeats once per hit, which
/// the caller's limit bounds, and never once per selected Context.
const PER_HIT_READ: &str = "FROM event_search_passage_sources";

/// What one find over the whole fixture cost: the statements it issued, and how many hits it
/// answered with.
struct FindCost {
    statements: Vec<String>,
    hits: usize,
}

impl FindCost {
    /// Every statement except the per-hit source-link read.
    fn per_context_candidates(&self) -> Vec<&str> {
        self.statements
            .iter()
            .map(String::as_str)
            .filter(|statement| !statement.contains(PER_HIT_READ))
            .collect()
    }

    fn per_hit_reads(&self) -> usize {
        self.statements
            .iter()
            .filter(|statement| statement.contains(PER_HIT_READ))
            .count()
    }
}

/// Every statement one find over the whole fixture issues, through the read-only door a `cg find`
/// process uses.
fn statements_for_one_find(fixture: &Fixture) -> FindCost {
    let mut reader =
        ConsumerReader::open(&fixture.config.db_path, event_history_reader_declaration())
            .expect("open the read-only reader in place of a fresh cg find process");
    reader
        .attach_event_search_embedder_registrations(vec![fixture.registration.clone()])
        .expect("attach the event-search embedder to the reader");
    let selector = ContextSelector::new(fixture.context_ids.iter().copied())
        .expect("a non-empty Context selection");

    record_event_history_statements_for_test();
    let answered = reader.find_event_history(
        selector,
        "why the broker-free path?",
        EventSearchFilter::default(),
    );
    FindCost {
        statements: take_recorded_event_history_statements_for_test(),
        hits: answered.map(|result| result.hits.len()).unwrap_or(0),
    }
}

/// `statement -> how many times a find issued it`, so a failure names the statement that fanned out
/// rather than only the total.
fn by_statement(statements: &[&str]) -> Vec<(usize, String)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for statement in statements {
        *counts.entry(statement).or_default() += 1;
    }
    let mut ranked = counts
        .into_iter()
        .map(|(statement, count)| (count, statement.to_string()))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    ranked
}

fn assert_statement_count_is_independent_of_context_count(project: bool, shape: &str) {
    for contexts in CONTEXT_COUNTS {
        let fixture = fixture(contexts, project);
        let cost = statements_for_one_find(&fixture);
        assert!(
            !cost.statements.is_empty(),
            "{shape}: a find over {contexts} Contexts issued no statement at all, so this proof \
             would pass without reading anything"
        );
        assert_eq!(
            cost.per_hit_reads(),
            cost.hits,
            "{shape}: at {contexts} Contexts the source links behind each returned passage are \
             read once per hit and never more"
        );
        let repeated = by_statement(&cost.per_context_candidates())
            .into_iter()
            .filter(|(count, _)| *count > 1)
            .collect::<Vec<_>>();
        assert!(
            repeated.is_empty(),
            "{shape}: a find over {contexts} Contexts issued a statement more than once, which is \
             what a read per Context looks like -- the per-Context reads all carry the same text \
             and differ only in the Context bound into them: {repeated:#?}"
        );
    }
}

#[test]
fn a_find_over_ready_contexts_issues_the_same_statements_at_one_eight_and_sixty_four() {
    assert_statement_count_is_independent_of_context_count(true, "ready");
}

#[test]
fn a_find_over_contexts_whose_projection_is_owed_issues_the_same_statements_at_any_count() {
    assert_statement_count_is_independent_of_context_count(false, "projection still owed");
}

/// The ranking statement names the (Context, active generation) pairs it searches as one standard
/// row-value list. Two independent id lists would mean their cross product, which the engine has to
/// plan as one index run per combination -- the square of the Context count, nearly all empty --
/// while the pairs themselves are one run each. The find still answers through that statement.
#[test]
fn a_find_names_the_pairs_it_searches_as_one_row_value_list() {
    for contexts in [1usize, 8] {
        let fixture = fixture(contexts, true);
        let cost = statements_for_one_find(&fixture);
        let ranking = cost
            .statements
            .iter()
            .find(|statement| statement.contains("ORDER BY vector_text <=>"))
            .expect("a find over ready Contexts issues its ranking statement");
        if contexts == 1 {
            assert!(
                ranking.contains("context_id = $") && ranking.contains("generation_id = $"),
                "one searched pair is two equalities: {ranking}"
            );
        } else {
            assert!(
                ranking.contains("(context_id, generation_id) IN (($"),
                "the searched pairs are one row-value list: {ranking}"
            );
        }
        assert!(
            !ranking.contains("context_id IN (") && !ranking.contains("generation_id IN ("),
            "no independent id list whose cross product the engine would have to plan: {ranking}"
        );
        assert!(
            cost.hits > 0,
            "the find over {contexts} Contexts answers through the pair statement"
        );
    }
}
