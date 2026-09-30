//! A second node receives the whole startup schema as the one commit it was authored in.
//!
//! The half of the one-commit promise that only two live nodes can prove. A peer replays each
//! schema commit it receives over its whole schema, so the cost of a node's first pull is decided
//! by how many commits the startup schema arrives in, not by how many statements it holds. The
//! fresh-store half -- that the author node writes exactly one -- is in
//! `startup_schema_is_one_commit.rs`.

#[allow(dead_code)]
#[path = "common/capture_sync_common.rs"]
mod capture_sync_common;
#[allow(dead_code)]
#[path = "common/fake_extractor.rs"]
mod fake_extractor;
#[path = "common/startup_schema_log.rs"]
mod startup_schema_log;

use std::collections::BTreeSet;

use crate::capture_sync_common::{bootstrap_sync_schema_without_rows, real_sync_fixture};
use crate::startup_schema_log::{assert_whole_startup_schema_is_one_commit, schema_log};

#[tokio::test(flavor = "multi_thread")]
async fn a_second_node_receives_the_whole_startup_schema_in_author_order() {
    let fixture = real_sync_fixture("whole_startup_schema_reaches_a_second_node").await;
    assert_whole_startup_schema_is_one_commit(&fixture.edge1.store, "author node");
    bootstrap_sync_schema_without_rows(&fixture).await;

    let authored = schema_log(&fixture.edge1.store);
    let received = schema_log(&fixture.edge2.store);
    assert!(
        received.author_order.starts_with(&authored.author_order),
        "the receiving node replays the startup schema in the order it was authored; it holds \
         {:?} where the author holds {:?}",
        received.author_order,
        authored.author_order
    );
    // Anything the receiving node declares after the startup schema is its own -- a sync client
    // records the peers it has met -- and is never one of the startup declarations.
    let startup = authored.author_order.iter().collect::<BTreeSet<_>>();
    for later in received
        .author_order
        .iter()
        .skip(authored.author_order.len())
    {
        assert!(
            !startup.contains(later),
            "the receiving node re-declares startup schema it already received: {later}"
        );
    }
    assert_eq!(
        assert_whole_startup_schema_is_one_commit(&fixture.edge2.store, "receiving node"),
        *received
            .table_commit
            .get("contexts")
            .expect("the receiving node holds the startup schema"),
        "the receiving node holds the whole startup schema as the one commit it arrived in"
    );
}
