use cg_node_fs::{
    EVENT_SEARCH_WORKER_POLICY, acquire_event_search_worker_policy_update,
    read_event_search_worker_policy, write_event_search_worker_policy,
};

#[test]
fn missing_policy_is_absent_then_writes_and_replacements_round_trip_at_the_fixed_path() {
    let root = tempfile::tempdir().expect("isolated data root");
    assert_eq!(
        read_event_search_worker_policy(root.path()).expect("read absent policy"),
        None
    );

    for expected in [
        r#"{"embedding_workers":2}"#,
        r#"{"embedding_workers":null}"#,
    ] {
        write_event_search_worker_policy(root.path(), expected).expect("replace worker policy");
        assert_eq!(
            std::fs::read_to_string(root.path().join(EVENT_SEARCH_WORKER_POLICY))
                .expect("policy exists at its fixed path"),
            expected
        );
        assert_eq!(
            read_event_search_worker_policy(root.path()).expect("read replaced policy"),
            Some(expected.to_string())
        );
    }
}

#[test]
fn policy_record_refuses_more_than_four_kibibytes_without_replacing_the_last_value() {
    let root = tempfile::tempdir().expect("isolated data root");
    let valid = r#"{"embedding_workers":2}"#;
    write_event_search_worker_policy(root.path(), valid).expect("write initial valid policy");

    let refusal = write_event_search_worker_policy(root.path(), &"x".repeat(4 * 1024 + 1));
    assert!(refusal.is_err(), "an oversized policy record was accepted");
    assert_eq!(
        read_event_search_worker_policy(root.path()).expect("read policy after refusal"),
        Some(valid.to_string()),
        "oversized input replaced the last valid policy"
    );
}

#[test]
fn update_guard_reads_and_replaces_one_bounded_record() {
    let root = tempfile::tempdir().expect("isolated data root");
    let guard = acquire_event_search_worker_policy_update(root.path()).expect("take update guard");
    assert_eq!(guard.read().expect("read absent policy"), None);
    guard
        .write(r#"{"publication":"one","embedding_workers":1,"pending_background_request":{"token":"wake","embedding_workers":2,"ignored_publication":"one"}}"#)
        .expect("write under update guard");
    assert_eq!(
        guard.read().expect("read updated policy"),
        Some(
            r#"{"publication":"one","embedding_workers":1,"pending_background_request":{"token":"wake","embedding_workers":2,"ignored_publication":"one"}}"#
                .into()
        )
    );
}
