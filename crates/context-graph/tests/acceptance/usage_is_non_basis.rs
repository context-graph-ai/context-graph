use context_graph::capture_with_usage_visibility;

use crate::capture_sync_common::{
    TestStore, assert_no_observation_basis_lane_contains_usage,
    assert_usage_does_not_affect_basis_warnings, assert_usage_observation_is_operational_metadata,
    capture_request, flat_pricing, provider_backed_extractor,
    provider_with_unique_usage_for_chunks, provider_with_usage, seed_decision_graph,
    seed_decisions_based_on_usage_entities,
};

#[test]
fn usage_observation_never_triggers_invalidation() {
    let store = TestStore::new("usage_observation_never_triggers_invalidation");
    let seeded = seed_decision_graph(&store.store, "tenant-alpha", "usage barrier");
    let provider = provider_with_usage();
    let extractor = provider_backed_extractor(&provider);
    let transcript = "Goal G1: record usage. Entity E1: model. Decision D1: keep usage operational serves G1 based_on E1.";
    let report = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", transcript),
        two_chunk_window(transcript),
        &extractor,
        &provider,
        flat_pricing(),
    )
    .expect("capture succeeds");
    let decision = store
        .store
        .get_decision(seeded.decision.id)
        .expect("read decision")
        .expect("decision exists");

    assert_eq!(report.calls.len(), 2);
    assert_usage_observation_is_operational_metadata(&store.store);
    let usage_basis_decisions = seed_decisions_based_on_usage_entities(&store.store);
    let (next_provider, _) = provider_with_unique_usage_for_chunks(2);
    let next_extractor = provider_backed_extractor(&next_provider);
    let _ = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", transcript),
        two_chunk_window(transcript),
        &next_extractor,
        &next_provider,
        flat_pricing(),
    )
    .expect("second usage capture succeeds");
    assert_no_observation_basis_lane_contains_usage(&store.store);
    assert_usage_does_not_affect_basis_warnings(&store.store, &usage_basis_decisions);
    assert_eq!(decision.status, context_graph::DecisionStatus::Active);
}

#[test]
fn usage_fields_excluded_from_record_observation_basis_inputs() {
    let store = TestStore::new("usage_fields_excluded_from_record_observation_basis_inputs");
    let provider = provider_with_usage();
    let extractor = provider_backed_extractor(&provider);
    let transcript = "Goal G1: keep usage out of basis. Entity E1: model. Decision D1: usage is metadata serves G1 based_on E1.";
    let _ = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", transcript),
        two_chunk_window(transcript),
        &extractor,
        &provider,
        flat_pricing(),
    )
    .expect("capture succeeds");

    assert_usage_observation_is_operational_metadata(&store.store);
    let usage_basis_decisions = seed_decisions_based_on_usage_entities(&store.store);
    let (next_provider, _) = provider_with_unique_usage_for_chunks(2);
    let next_extractor = provider_backed_extractor(&next_provider);
    let _ = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", transcript),
        two_chunk_window(transcript),
        &next_extractor,
        &next_provider,
        flat_pricing(),
    )
    .expect("second usage capture succeeds");
    assert_no_observation_basis_lane_contains_usage(&store.store);
    assert_usage_does_not_affect_basis_warnings(&store.store, &usage_basis_decisions);
}

#[test]
fn cost_metadata_cannot_become_decision_basis() {
    let store = TestStore::new("cost_metadata_cannot_become_decision_basis");
    let provider = provider_with_usage();
    let extractor = provider_backed_extractor(&provider);
    let transcript = "Goal G1: keep cost operational. Entity E1: model. Decision D1: cost cannot alter basis serves G1 based_on E1.";
    let _ = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", transcript),
        two_chunk_window(transcript),
        &extractor,
        &provider,
        flat_pricing(),
    )
    .expect("capture succeeds");

    assert_usage_observation_is_operational_metadata(&store.store);
    let usage_basis_decisions = seed_decisions_based_on_usage_entities(&store.store);
    let (next_provider, _) = provider_with_unique_usage_for_chunks(2);
    let next_extractor = provider_backed_extractor(&next_provider);
    let _ = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", transcript),
        two_chunk_window(transcript),
        &next_extractor,
        &next_provider,
        flat_pricing(),
    )
    .expect("second usage capture succeeds");
    assert_no_observation_basis_lane_contains_usage(&store.store);
    assert_usage_does_not_affect_basis_warnings(&store.store, &usage_basis_decisions);
}

fn two_chunk_window(transcript: &str) -> usize {
    transcript.len().div_ceil(2)
}
