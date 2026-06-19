use std::time::Duration;

use context_graph::{CapturePricing, HostedProviderBoundary, capture_with_usage_visibility};

use crate::capture_sync_common::{
    TestStore, assert_context_has_decision_and_digest, capture_request, expected_chunk_count,
    expected_total_tokens, expected_zero_absent_and_real_usage, flat_pricing, haiku_pricing,
    provider_absent_and_zero_usage, provider_backed_extractor,
    provider_with_unique_usage_for_chunks, provider_with_usage,
};

#[test]
fn usage_recorded_per_model_call_and_summed_per_run() {
    let store = TestStore::new("usage_recorded_per_model_call_and_summed_per_run");
    let (transcript, max_chars_per_chunk, expected_chunks) = three_plus_chunk_transcript(
        "Goal G1: share memory. Entity E1: laptop. Decision D1: sync through contextdb serves G1 based_on E1.",
    );
    let (provider, expected_calls) = provider_with_unique_usage_for_chunks(expected_chunks);
    let extractor = provider_backed_extractor(&provider);
    let report = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", &transcript),
        max_chars_per_chunk,
        &extractor,
        &provider,
        flat_pricing(),
    )
    .expect("capture succeeds");

    assert_eq!(provider.call_count(), expected_calls.len());
    assert_eq!(report.calls, expected_calls);
    assert_eq!(
        report.total_tokens,
        Some(expected_total_tokens(&expected_calls))
    );
}

#[test]
fn missing_provider_usage_reports_unavailable_without_crash() {
    let store = TestStore::new("missing_provider_usage_reports_unavailable_without_crash");
    let expected_states = expected_zero_absent_and_real_usage();
    let real_provider = provider_with_usage();
    let real_transcript = "Goal G1: keep the receipt. Entity E1: capture run. Decision D1: preserve missing usage serves G1 based_on E1.";
    let real_chunk_window = real_transcript.len() + 1;
    assert_eq!(expected_chunk_count(real_transcript, real_chunk_window), 1);
    let real_extractor = provider_backed_extractor(&real_provider);
    let real_report = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", real_transcript),
        real_chunk_window,
        &real_extractor,
        &real_provider,
        flat_pricing(),
    )
    .expect("capture with real usage succeeds");
    let absent_provider = provider_absent_and_zero_usage();
    let (absent_transcript, absent_chunk_window) = two_chunk_transcript(
        "Goal G2: keep zero distinct. Entity E2: capture run. Decision D2: zero is not missing serves G2 based_on E2.",
    );
    let absent_extractor = provider_backed_extractor(&absent_provider);
    let absent_report = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", &absent_transcript),
        absent_chunk_window,
        &absent_extractor,
        &absent_provider,
        flat_pricing(),
    )
    .expect("capture succeeds even when usage is absent");

    let actual_states = real_report
        .calls
        .iter()
        .take(1)
        .chain(absent_report.calls.iter())
        .map(|call| call.usage.clone())
        .collect::<Vec<_>>();
    assert_eq!(actual_states, expected_states);
    assert_context_has_decision_and_digest(&store.store, "tenant-alpha", "preserve missing usage");
    assert_context_has_decision_and_digest(&store.store, "tenant-alpha", "zero is not missing");
}

#[test]
fn run_receipt_reports_wall_clock_time() {
    let store = TestStore::new("run_receipt_reports_wall_clock_time");
    let (transcript, max_chars_per_chunk, expected_chunks) = three_plus_chunk_transcript(
        "Goal G1: see elapsed time. Entity E1: import job. Decision D1: show receipt serves G1 based_on E1.",
    );
    let (provider, expected_calls) = provider_with_unique_usage_for_chunks(expected_chunks);
    let extractor = provider_backed_extractor(&provider);
    let report = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", &transcript),
        max_chars_per_chunk,
        &extractor,
        &provider,
        flat_pricing(),
    )
    .expect("capture succeeds");

    assert_eq!(provider.call_count(), expected_chunks);
    let expected_run_elapsed = expected_calls
        .iter()
        .map(|call| call.elapsed)
        .sum::<Duration>();
    assert!(
        report.elapsed >= expected_run_elapsed,
        "run elapsed {:?} must cover summed per-chunk elapsed {:?}",
        report.elapsed,
        expected_run_elapsed
    );
    assert_eq!(
        report.chunk_elapsed,
        expected_calls
            .iter()
            .map(|call| call.elapsed)
            .collect::<Vec<_>>()
    );
}

#[test]
fn dollars_dark_on_flat_path_lit_on_pay_per_token_path() {
    let store = TestStore::new("dollars_dark_on_flat_path_lit_on_pay_per_token_path");
    let (pay_transcript, pay_chunk_window, pay_chunks) = three_plus_chunk_transcript(
        "Goal G1: price the run. Entity E1: haiku path. Decision D1: show dollars serves G1 based_on E1.",
    );
    let (pay_provider, expected_pay_calls) = provider_with_unique_usage_for_chunks(pay_chunks);
    let pay_extractor = provider_backed_extractor(&pay_provider);
    let pay_report = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", &pay_transcript),
        pay_chunk_window,
        &pay_extractor,
        &pay_provider,
        haiku_pricing(),
    )
    .expect("pay-per-token capture succeeds");
    let (flat_transcript, flat_chunk_window) = two_chunk_transcript(
        "Goal G2: hide flat dollars. Entity E2: codex path. Decision D2: time is the meter serves G2 based_on E2.",
    );
    let flat_provider = provider_with_usage();
    let flat_extractor = provider_backed_extractor(&flat_provider);
    let flat_report = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", &flat_transcript),
        flat_chunk_window,
        &flat_extractor,
        &flat_provider,
        CapturePricing::FlatSubscription,
    )
    .expect("flat capture succeeds");

    assert_eq!(pay_report.calls, expected_pay_calls);
    assert_eq!(
        pay_report.dollars,
        Some(expected_total_tokens(&expected_pay_calls) as f64 * 0.00000025)
    );
    assert_eq!(flat_report.dollars, None);
}

fn two_chunk_transcript(stem: &str) -> (String, usize) {
    let transcript = format!("{stem}\n{stem}");
    let max_chars_per_chunk = transcript.len().div_ceil(2);
    assert_eq!(expected_chunk_count(&transcript, max_chars_per_chunk), 2);
    (transcript, max_chars_per_chunk)
}

fn three_plus_chunk_transcript(stem: &str) -> (String, usize, usize) {
    let transcript = format!("{stem}\n{stem}\n{stem}\n{stem}");
    let max_chars_per_chunk = 120;
    let chunks = expected_chunk_count(&transcript, max_chars_per_chunk);
    assert!(
        chunks >= 3,
        "fixture transcript must drive three or more provider calls"
    );
    (transcript, max_chars_per_chunk, chunks)
}
