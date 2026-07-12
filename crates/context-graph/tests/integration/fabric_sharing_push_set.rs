//! Fabric first-light — the global sharing switch, proven at the REAL push level.
//!
//! Non-frozen companion to the privacy contract (`fabric_privacy`). Sharing OFF keeps the ephemeral
//! understanding input off the wire because the copy is NEVER WRITTEN — submission uses an
//! `InputRef::local_path` and no `work_inputs` row exists — not because a downstream filter hides an
//! existing copy. These tests read the actual rows that cross a real in-process hub: with sharing off
//! a push carries zero `work_inputs` rows (hence zero understanding-input bytes) while the job row
//! still syncs; with sharing on the input copy crosses to the same-tenant peer.

use context_graph::Store;
use context_graph::fabric::{self, FabricCapture, FabricConfig};

use crate::fabric_common::{fabric_fixture, push_then_pull, tier0_request, work_row_count};

const SESSION: &str = "User: Goal G1: keep secrets local. Assistant: Decision D1: keep the auth module at src/auth.py.";

/// Total bytes across every `work_inputs` row on a node — the understanding-input material that would
/// travel on a push. Zero rows means zero bytes, so this stays 0 exactly when no copy was written.
fn understanding_input_bytes(store: &Store) -> usize {
    let db = store.sync_database();
    db.scan("work_inputs", db.snapshot())
        .map(|rows| {
            rows.iter()
                .map(|row| format!("{:?}", row.values).len())
                .sum()
        })
        .unwrap_or(0)
}

fn tenant_config(sharing: bool) -> FabricConfig {
    FabricConfig {
        tenant: Some("tenant-alpha".to_string()),
        sharing,
        ..Default::default()
    }
}

#[tokio::test]
async fn sharing_off_push_carries_no_input_rows_or_bytes() {
    let fixture = fabric_fixture("sharing_off_push").await;
    let config = tenant_config(false);
    let mut capture = FabricCapture::new(tier0_request("daily-loop", "s1", SESSION));
    capture.config = Some(&config);
    fabric::capture(&fixture.edge1.store, capture).expect("capture with sharing off");

    // The copy never existed at the SOURCE — nothing to leak on a push.
    assert_eq!(
        work_row_count(&fixture.edge1.store, "work_inputs"),
        0,
        "sharing off writes no work_inputs copy on the submitter"
    );
    assert_eq!(
        understanding_input_bytes(&fixture.edge1.store),
        0,
        "sharing off means zero understanding-input bytes staged for the wire"
    );

    // A REAL push to the in-process hub, then a pull on the peer. The peer receives the job but no
    // input copy — the OFF switch dominated at the source, not by any downstream filter.
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    assert!(
        work_row_count(&fixture.edge2.store, "work_jobs") >= 1,
        "the understanding job still syncs across the hub with sharing off"
    );
    assert_eq!(
        work_row_count(&fixture.edge2.store, "work_inputs"),
        0,
        "a real push carries zero work_inputs rows to the peer with sharing off"
    );
    assert_eq!(
        understanding_input_bytes(&fixture.edge2.store),
        0,
        "the peer received zero understanding-input bytes with sharing off"
    );
}

#[tokio::test]
async fn sharing_on_push_carries_the_input() {
    let fixture = fabric_fixture("sharing_on_push").await;
    let config = tenant_config(true);
    let mut capture = FabricCapture::new(tier0_request("daily-loop", "s1", SESSION));
    capture.config = Some(&config);
    fabric::capture(&fixture.edge1.store, capture).expect("capture with sharing on");

    // With sharing on the copy is written at the source and enters the push set.
    assert!(
        work_row_count(&fixture.edge1.store, "work_inputs") >= 1,
        "sharing on writes the ephemeral input copy on the submitter"
    );
    assert!(
        understanding_input_bytes(&fixture.edge1.store) > 0,
        "sharing on stages understanding-input bytes for the wire"
    );

    // A REAL push + peer pull: the input copy crosses to the same-tenant peer.
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    assert!(
        work_row_count(&fixture.edge2.store, "work_inputs") >= 1,
        "a real push carries the work_inputs copy to the same-tenant peer with sharing on"
    );
    assert!(
        understanding_input_bytes(&fixture.edge2.store) > 0,
        "the peer received the understanding-input bytes with sharing on"
    );
}
