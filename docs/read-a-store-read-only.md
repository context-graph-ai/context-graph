# Read a store without changing it

Prerequisite: the store path and the tables your application needs.

Add `context-graph` from a git checkout using the [path dependency instructions](library-quickstart.md#add-workspace-crates).

A read-only open of an absent store creates no store. <!-- enforced by: `an_absent_store_is_refused_as_missing_and_leaves_the_directory_untouched` -->

It leaves the containing directory unchanged. <!-- enforced by: `an_absent_store_is_refused_as_missing_and_leaves_the_directory_untouched` -->

A store needing writable recovery is not classified as unreadable. <!-- enforced by: `a_store_needing_writable_recovery_is_its_own_typed_outcome` -->

A store that cannot be inspected is not classified as missing. <!-- enforced by: `a_store_this_process_cannot_look_at_is_refused_as_unreadable_never_as_missing` -->

A missing-store refusal names the absent path. <!-- enforced by: `an_absent_store_is_refused_as_missing_and_leaves_the_directory_untouched` -->

A recovery-needing read-only refusal leaves the store bytes unchanged. <!-- enforced by: `a_store_needing_writable_recovery_is_its_own_typed_outcome` -->

It preserves the substrate's reason. <!-- enforced by: `a_store_needing_writable_recovery_is_its_own_typed_outcome` -->

An unreadable damaged-store refusal carries a nonempty reason. <!-- enforced by: `a_genuinely_damaged_store_is_still_reported_unreadable` -->

It leaves the damaged store bytes unchanged. <!-- enforced by: `a_genuinely_damaged_store_is_still_reported_unreadable` -->

Recoverable and unreadable stores produce distinct typed refusals. <!-- enforced by: `recoverable_and_unreadable_are_two_typed_answers_not_one` -->

A declared ConsumerReader returns its table's SQL rows. <!-- enforced by: `a_declared_reader_journeys_from_open_through_sql_metadata_and_cursor_to_close` -->

It returns metadata for that table. <!-- enforced by: `a_declared_reader_journeys_from_open_through_sql_metadata_and_cursor_to_close` -->

Its cursor's first page returns the declared rows. <!-- enforced by: `a_declared_reader_journeys_from_open_through_sql_metadata_and_cursor_to_close` -->

<!-- concept -->
Consult the [consumer graph read reference](library-api.md#consumer-graph-reads) for table requirements, nested answers and audit snapshots.

```rust,ignore
use std::collections::BTreeSet;

use context_graph::schema::TABLES;
use context_graph::{
    CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, DecisionId,
};

/// Why a decision fired, read out of a store this process must not change.
fn explain(store_path: &std::path::Path, decision: DecisionId) {
    let declaration = ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "my-app".to_string(),
            tables: ["decisions", "edges", "decision_basis_snapshots", "outcomes"]
                .iter()
                .map(|wanted| {
                    let spec = TABLES
                        .iter()
                        .find(|spec| spec.name == *wanted)
                        .expect("context-graph publishes a spec for every table it owns");
                    ConsumerTable {
                        name: spec.name.to_string(),
                        ddl: spec.ddl.to_string(),
                    }
                })
                .collect(),
        },
        scope_labels: BTreeSet::new(),
    };

    let reader = match ConsumerReader::open(store_path, declaration) {
        Ok(reader) => reader,
        // The store is fine and only needs settling: take the writable
        // existing-only door, recover it, and carry on serving.
        Err(CgError::StoreNeedsWritableRecovery { path, reason }) => {
            eprintln!("{} needs a writable open first: {reason}", path.display());
            return;
        }
        Err(other) => {
            eprintln!("no read-only answer from this store: {other}");
            return;
        }
    };

    match reader.graph().get_decision(decision) {
        Ok(Some(decision)) => println!("{} — {}", decision.decision_type, decision.description),
        Ok(None) => println!("no such decision in this store"),
        Err(CgError::UndeclaredTableRefused { table }) => {
            eprintln!("declare `{table}` to read this");
        }
        Err(other) => eprintln!("{other}"),
    }

    reader.close().expect("let the read session go");
}
```

These examples are illustrative; this page does not run doctests.

See [configuration](configuration.md#read-budgets) for read budgets.
