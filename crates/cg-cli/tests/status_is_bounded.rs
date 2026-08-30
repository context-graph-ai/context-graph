//! `cg status` reports the same correct totals on a store no single read can scan.
//!
//! The library proof
//! (`context-graph/tests/status_totals_are_bounded_on_a_large_store.rs`) proves that the
//! understanding-queue counts and the raw-source footprint are computed through the bounded
//! reading door. This is the operator-facing half: the command a person actually runs must print
//! those totals, on a context holding more than 62,000 chunk rows — well past the shipped `work`
//! (50,000) and `result_rows` (500) ceilings — rather than failing, truncating, or reporting a
//! capped count as if it were the whole store.
//!
//! The fixture is the SAME builder the library proof uses, so both sides assert against one piece
//! of arithmetic. The store is written in-process and the handle dropped before `cg` is spawned:
//! contextdb is single-owner, so the binary has to be the one holding it when it answers.
//!
//! This test is slow by construction (it writes 62,000 rows through the production capture door).
//! The repo runs one tier — there is no `#[ignore]` or nightly gate anywhere in `crates/*/tests` —
//! so it stays in the normal suite and is left as one test, not several, to build the fixture once.

mod support;

#[path = "../../context-graph/tests/common/large_status_fixture.rs"]
mod large_status_fixture;

use serde_json::Value as JsonValue;
use support::CliHarness;

use large_status_fixture::{CONTEXT, fill, open_store};

fn total(json: &JsonValue, pointer: &str) -> u64 {
    json.pointer(pointer)
        .and_then(JsonValue::as_u64)
        .unwrap_or_else(|| panic!("cg status reported no `{pointer}`: {json}"))
}

#[test]
fn status_reports_whole_store_totals_on_a_context_larger_than_one_read() {
    let harness = CliHarness::new();
    let db_path = harness.data_root.path().join("context-graph.db");

    let fixture = {
        let store = open_store(&db_path);
        let fixture = fill(&store);
        // Hand the store back before the binary asks for it.
        drop(store);
        fixture
    };

    let status = harness.run(&["status", "--context", CONTEXT, "--format", "json"]);
    assert_eq!(
        status.code, 0,
        "cg status failed on a large store: {}\n{}",
        status.stdout, status.stderr
    );
    let json: JsonValue = serde_json::from_str(&status.stdout)
        .unwrap_or_else(|err| panic!("cg status output was not JSON ({err}): {}", status.stdout));

    assert_eq!(
        total(&json, "/raw_footprint/chunks"),
        u64::try_from(fixture.chunk_count).expect("a chunk count fits"),
        "cg status must report every kept chunk, not the first page of them"
    );
    assert_eq!(
        total(&json, "/raw_footprint/bytes"),
        fixture.raw_bytes,
        "cg status must report the whole kept-byte total"
    );
    assert_eq!(
        total(&json, "/understanding_queue/pending"),
        u64::try_from(fixture.chunk_count).expect("a digest count fits"),
        "every captured digest is queued for understanding, and status counts all of them"
    );
    for state in ["in_flight", "structured", "failed", "skipped"] {
        assert_eq!(
            total(&json, &format!("/understanding_queue/{state}")),
            0,
            "nothing has been understood, claimed, failed or purged in this fixture"
        );
    }
}
