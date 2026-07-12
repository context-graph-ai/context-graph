//! Fabric first-light — the privacy contract: the sharing switch, per-source opt-out, fail-closed
//! unknown ref kinds, cross-tenant isolation, and the never-sync permanent raw store.

use context_graph::fabric::{
    self, FabricCapture, FabricConfig, FabricPollOutcome, FabricWorkerConfig, MovementDecision,
};
use contextdb_engine::sync_types::SyncDirection;

use crate::fabric_common::{
    CountingExtractor, FabricNode, context_id, fabric_fixture_with_tenants, memory_total,
    push_then_pull, tier0_request, work_row_count,
};

const SESSION: &str = "User: Goal G1: keep secrets local. Assistant: Decision D1: keep the auth module at src/auth.py.";

fn sharing_off_config() -> FabricConfig {
    FabricConfig {
        tenant: Some("tenant-alpha".to_string()),
        sharing: false,
        ..Default::default()
    }
}

#[tokio::test]
async fn sharing_off_no_input_material_leaves_machine() {
    let node = FabricNode::open("sharing_off_none");
    let config = sharing_off_config();
    let mut capture = FabricCapture::new(tier0_request("daily-loop", "s1", SESSION));
    capture.config = Some(&config);
    fabric::capture(&node.store, capture).expect("capture");

    assert_eq!(
        fabric::input_material_in_push_set(&node.store, &config).expect("push set"),
        0,
        "with sharing off, no input copy enters any outbound changeset"
    );
    let extractor = CountingExtractor::new();
    let drained = fabric::drain_local_worker(&node.store, &extractor, "local-node", None)
        .expect("local drain");
    assert_eq!(
        drained, 1,
        "capture and local understanding still complete with sharing off"
    );
}

#[test]
fn sharing_off_with_stale_ticket_still_never_emits_input() {
    let node = FabricNode::open("sharing_off_ticket");
    let config = FabricConfig {
        hub_endpoint: Some("stale-unreachable-ticket".to_string()),
        tenant: Some("tenant-alpha".to_string()),
        sharing: false,
        ..Default::default()
    };
    let mut capture = FabricCapture::new(tier0_request("daily-loop", "s1", SESSION));
    capture.config = Some(&config);
    fabric::capture(&node.store, capture).expect("capture");

    assert_eq!(
        work_row_count(&node.store, "work_jobs"),
        1,
        "capture still submits a local job even with sharing off"
    );
    assert_eq!(
        fabric::input_material_in_push_set(&node.store, &config).expect("push set"),
        0,
        "an OFF switch dominates a present ticket — zero input material in any changeset"
    );
}

#[test]
fn no_fabric_toml_is_exactly_single_node_behavior() {
    let node = FabricNode::open("no_fabric_toml");
    assert_eq!(
        FabricConfig::load(&node.data_root()).expect("load fabric config"),
        None,
        "absent fabric.toml, the node has no hub to contact"
    );
    let report = fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("single-node capture");
    assert!(
        report.digests_written >= 1,
        "capture produces a digest locally"
    );
    let extractor = CountingExtractor::new();
    assert!(
        fabric::drain_local_worker(&node.store, &extractor, "local-node", None).is_ok(),
        "the worker drains locally with no hub contact, exactly as today"
    );
}

#[test]
fn per_source_opt_out_keeps_that_sources_input_local_under_sharing_on() {
    // Sharing on: a default source's input may move; an opted-out source's input stays local.
    assert_eq!(
        fabric::movement_decision("ledger_input", true, false),
        MovementDecision::MayMove,
        "a default source's ledger input may move under sharing on"
    );
    assert_eq!(
        fabric::movement_decision("local_path", true, false),
        MovementDecision::LocalOnly,
        "an opted-out source keeps its input local even under sharing on"
    );
}

#[test]
fn capture_flag_opt_out_matches_config_opt_out() {
    // Flag opt-out on one node; config opt-out on another. Both yield the identical local-only
    // outcome: a submitted job whose input never enters the push set.
    let flag_node = FabricNode::open("optout_flag");
    let mut flag_capture = FabricCapture::new(tier0_request("daily-loop", "s1", SESSION));
    flag_capture.source_opt_out = true;
    fabric::capture(&flag_node.store, flag_capture).expect("flag capture");

    let config = FabricConfig {
        tenant: Some("tenant-alpha".to_string()),
        per_source_opt_out: vec!["cg-record".to_string()],
        ..Default::default()
    };
    let cfg_node = FabricNode::open("optout_config");
    let mut cfg_capture = FabricCapture::new(tier0_request("daily-loop", "s1", SESSION));
    cfg_capture.config = Some(&config);
    fabric::capture(&cfg_node.store, cfg_capture).expect("config capture");

    assert_eq!(
        work_row_count(&flag_node.store, "work_jobs"),
        1,
        "the flag opt-out still submits a local job"
    );
    assert_eq!(
        fabric::input_material_in_push_set(&flag_node.store, &FabricConfig::default())
            .expect("flag push set"),
        fabric::input_material_in_push_set(&cfg_node.store, &config).expect("config push set"),
        "flag opt-out and config opt-out yield the same local-only outcome"
    );
}

#[test]
fn unknown_input_reference_kind_fails_closed() {
    assert_eq!(
        fabric::movement_decision("some_future_kind", true, false),
        MovementDecision::Refused,
        "an unrecognized input-reference kind is refused (default-deny), never admitted as movable"
    );
}

#[tokio::test]
async fn input_copy_scoped_to_tenant_a_rejected_at_apply_on_tenant_b_handle() {
    let fixture = fabric_fixture_with_tenants("cross_tenant_reject", "tenant-a", "tenant-b").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on tenant-a edge");

    // Tenant A's own edge may execute its job.
    let ext_a = CountingExtractor::new();
    let config_a = FabricWorkerConfig::for_node("edge-a");
    let out_a = fabric::run_worker_once(
        &fixture.edge1_client,
        &fixture.edge1.store,
        &ext_a,
        &config_a,
        1_000,
    )
    .expect("tenant-a poll");
    assert!(
        matches!(out_a, FabricPollOutcome::Executed { .. }),
        "tenant A's own edge executes its job"
    );

    // A move toward tenant B is walled by the engine guard: B admits nothing.
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let ext_b = CountingExtractor::new();
    let config_b = FabricWorkerConfig::for_node("edge-b");
    let _ = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext_b,
        &config_b,
        1_000,
    );
    if let Some(ctx_b) = context_id(&fixture.edge2.store, "daily-loop") {
        assert_eq!(
            memory_total(&fixture.edge2.store, ctx_b),
            0,
            "a B-scoped handle admits none of tenant A's input — cross-tenant movement is impossible"
        );
    }
}

#[tokio::test]
async fn default_on_sharing_is_same_tenant_scoped() {
    let fixture = fabric_fixture_with_tenants("default_on_tenant", "tenant-a", "tenant-b").await;
    let config = FabricConfig {
        tenant: Some("tenant-a".to_string()),
        ..Default::default()
    };
    let mut capture = FabricCapture::new(tier0_request("daily-loop", "s1", SESSION));
    capture.config = Some(&config);
    fabric::capture(&fixture.edge1.store, capture).expect("capture on tenant-a edge");

    assert!(
        fabric::input_material_in_push_set(&fixture.edge1.store, &config).expect("push set") > 0,
        "default-ON sharing moves an input copy within tenant A"
    );

    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let ext_b = CountingExtractor::new();
    let config_b = FabricWorkerConfig::for_node("edge-b");
    let out_b = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext_b,
        &config_b,
        1_000,
    )
    .expect("tenant-b poll");
    assert_eq!(
        out_b,
        FabricPollOutcome::NoWork,
        "a differently-tenanted edge never becomes eligible to execute tenant A's input"
    );
}

#[test]
fn permanent_raw_store_never_syncs() {
    let contract = context_graph::context_graph_sync_table_contract();
    let raw = contract
        .iter()
        .find(|policy| policy.table == "cg_source_chunks")
        .expect("cg_source_chunks is in the sync contract");
    assert_eq!(
        raw.direction,
        SyncDirection::None,
        "the permanent raw store carries no-sync direction; no push changeset ever contains it"
    );
}
