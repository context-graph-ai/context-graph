use context_graph::sync_status_report;
use contextdb_core::Lsn;

use crate::capture_sync_common::{
    TestStore, TestSyncStatusBoundary, assert_lsn_advanced, assert_real_decision_content,
    expected_pending_push_change_count, push_pull, real_sync_fixture, seed_decision_graph,
};

#[tokio::test(flavor = "multi_thread")]
async fn status_reports_configured_remote_and_sync_watermark() {
    let fixture = real_sync_fixture("status_reports_configured_remote_and_sync_watermark").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "status watermark");
    let before = fixture.edge1_client.push_watermark();
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let status = sync_status_report(
        &fixture.edge1.store,
        Some(&fixture.nats_url),
        Some(&fixture.tenant_id),
        Some(&TestSyncStatusBoundary(&fixture.edge1_client)),
    );

    assert_eq!(
        status.remote_url.as_deref(),
        Some(fixture.nats_url.as_str())
    );
    assert_eq!(
        status.tenant_id.as_deref(),
        Some(fixture.tenant_id.as_str())
    );
    assert_lsn_advanced(before, status.push_watermark);
    assert_eq!(status.push_watermark, fixture.edge1_client.push_watermark());
    assert_eq!(status.pull_watermark, fixture.edge1_client.pull_watermark());
    assert_real_decision_content(
        &fixture.edge2.store,
        seeded.decision.id,
        &seeded.decision.description,
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn status_reports_pending_push_count() {
    let fixture = real_sync_fixture("status_reports_pending_push_count").await;
    let _seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "pending status");
    assert!(
        fixture
            .edge1_client
            .has_pending_push_changes()
            .expect("pending push detection before report"),
        "seeded rows must be pending before the status report"
    );
    let expected_pending = expected_pending_push_change_count(&fixture.edge1_client);
    assert!(
        expected_pending > 1,
        "pending setup must produce a real count, not a boolean-sized sentinel"
    );
    let pending = sync_status_report(
        &fixture.edge1.store,
        Some(&fixture.nats_url),
        Some(&fixture.tenant_id),
        Some(&TestSyncStatusBoundary(&fixture.edge1_client)),
    );
    fixture
        .edge1_client
        .push()
        .await
        .expect("push pending rows");
    let expected_drained = expected_pending_push_change_count(&fixture.edge1_client);
    let drained = sync_status_report(
        &fixture.edge1.store,
        Some(&fixture.nats_url),
        Some(&fixture.tenant_id),
        Some(&TestSyncStatusBoundary(&fixture.edge1_client)),
    );

    assert_eq!(pending.pending_push_count, expected_pending);
    assert!(
        !fixture
            .edge1_client
            .has_pending_push_changes()
            .expect("pending push detection after push"),
        "push must drain pending rows before the second status report"
    );
    assert_eq!(expected_drained, 0);
    assert_eq!(drained.pending_push_count, expected_drained);
}

#[test]
fn no_remote_configured_opens_and_behaves_exactly_as_today() {
    let store = TestStore::new("no_remote_configured_opens_and_behaves_exactly_as_today");
    let seeded = seed_decision_graph(&store.store, "tenant-alpha", "standalone");
    let status = sync_status_report(&store.store, None, None, None);

    assert_eq!(status.remote_url, None);
    assert_eq!(status.tenant_id, None);
    assert_eq!(status.push_watermark, Lsn(0));
    assert_eq!(status.pull_watermark, Lsn(0));
    assert_eq!(status.pending_push_count, 0);
    assert!(!status.sync_client_constructed);
    assert!(!status.nats_connection_opened);
    assert!(
        store
            .store
            .get_decision(seeded.decision.id)
            .expect("read local decision")
            .is_some()
    );
}
