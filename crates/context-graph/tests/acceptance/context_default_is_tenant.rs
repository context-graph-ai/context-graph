use context_graph::capture_with_default_context;

use crate::capture_sync_common::{TestStore, assert_decision_landed_in_context, extractor};

#[test]
fn default_capture_context_follows_tenant_not_working_directory() {
    let store = TestStore::new("default_capture_context_follows_tenant_not_working_directory");
    let second_machine =
        TestStore::new("default_capture_context_follows_tenant_not_working_directory_second");
    let cwd_a = tempfile::tempdir().expect("cwd a");
    let cwd_b = tempfile::tempdir().expect("cwd b");
    let cwd_c = tempfile::tempdir().expect("cwd c");

    let tenant_capture_a = capture_with_default_context(
        &store.store,
        Some("owner-tenant"),
        None,
        cwd_a.path(),
        "Goal G1: converge by tenant. Entity E1: auth module path=src/auth.py. Decision D1: tenant memory lands together serves G1 based_on E1.",
        &extractor(),
    )
    .expect("tenant capture from cwd a");
    let tenant_capture_b = capture_with_default_context(
        &store.store,
        Some("owner-tenant"),
        None,
        cwd_b.path(),
        "Goal G2: converge by tenant again. Entity E2: config file path=config/app.toml. Decision D2: second tenant capture lands together serves G2 based_on E2.",
        &extractor(),
    )
    .expect("tenant capture from cwd b");
    let tenant_capture_second_machine = capture_with_default_context(
        &second_machine.store,
        Some("owner-tenant"),
        None,
        cwd_c.path(),
        "Goal G7: converge by tenant from another machine. Entity E7: second machine file path=machine.rs. Decision D7: tenant memory from second machine lands in the same context serves G7 based_on E7.",
        &extractor(),
    )
    .expect("tenant capture from second machine");
    let other_tenant_capture = capture_with_default_context(
        &store.store,
        Some("other-owner-tenant"),
        None,
        cwd_a.path(),
        "Goal G3: isolate other tenant. Entity E3: billing file path=billing.toml. Decision D3: other tenant stays separate serves G3 based_on E3.",
        &extractor(),
    )
    .expect("other tenant capture");
    let explicit_capture = capture_with_default_context(
        &store.store,
        Some("owner-tenant"),
        Some("client-wall"),
        cwd_a.path(),
        "Goal G4: honor explicit wall. Entity E4: client file path=client.toml. Decision D4: explicit client wall stays separate serves G4 based_on E4.",
        &extractor(),
    )
    .expect("explicit context capture");
    let default_capture_a = capture_with_default_context(
        &store.store,
        None,
        None,
        cwd_a.path(),
        "Goal G5: stable anonymous default. Entity E5: default file path=default.toml. Decision D5: anonymous default is stable serves G5 based_on E5.",
        &extractor(),
    )
    .expect("default capture from cwd a");
    let default_capture_b = capture_with_default_context(
        &store.store,
        None,
        None,
        cwd_b.path(),
        "Goal G6: stable anonymous default again. Entity E6: default file path=default2.toml. Decision D6: anonymous default ignores cwd serves G6 based_on E6.",
        &extractor(),
    )
    .expect("default capture from cwd b");

    assert_eq!(tenant_capture_a.context_id, tenant_capture_b.context_id);
    assert_eq!(
        tenant_capture_a.context_id, tenant_capture_second_machine.context_id,
        "same tenant must resolve to one stable context id across independent stores"
    );
    assert_ne!(tenant_capture_a.context_id, other_tenant_capture.context_id);
    assert_ne!(tenant_capture_a.context_id, explicit_capture.context_id);
    assert_eq!(default_capture_a.context_id, default_capture_b.context_id);
    assert_decision_landed_in_context(
        &store.store,
        tenant_capture_a.context_id,
        "tenant memory lands together",
    );
    assert_decision_landed_in_context(
        &store.store,
        tenant_capture_b.context_id,
        "second tenant capture lands together",
    );
    assert_decision_landed_in_context(
        &store.store,
        other_tenant_capture.context_id,
        "other tenant stays separate",
    );
    assert_decision_landed_in_context(
        &store.store,
        explicit_capture.context_id,
        "explicit client wall stays separate",
    );
    assert_decision_landed_in_context(
        &store.store,
        default_capture_a.context_id,
        "anonymous default is stable",
    );
    assert_decision_landed_in_context(
        &store.store,
        default_capture_b.context_id,
        "anonymous default ignores cwd",
    );
    assert_decision_landed_in_context(
        &second_machine.store,
        tenant_capture_second_machine.context_id,
        "tenant memory from second machine lands in the same context",
    );
}
