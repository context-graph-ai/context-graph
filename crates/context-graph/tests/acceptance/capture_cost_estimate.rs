use std::time::Duration;

use context_graph::{CapturePricing, HostedProviderBoundary, estimate_capture};

use crate::capture_sync_common::{
    BACKLOG, expected_chunk_count, expected_projected_tokens, haiku_pricing, provider_with_usage,
};

#[test]
fn estimate_makes_zero_provider_calls() {
    let provider = provider_with_usage();
    let estimate = estimate_capture(
        BACKLOG,
        160,
        Duration::from_secs(20),
        CapturePricing::FlatSubscription,
        &provider,
    );

    assert_eq!(provider.call_count(), 0);
    assert_eq!(estimate.provider_calls, 0);
    assert_eq!(
        estimate.projected_chunks,
        expected_chunk_count(BACKLOG, 160)
    );
}

#[test]
fn estimate_projects_chunk_count_and_time() {
    let provider = provider_with_usage();
    let transcript = format!("{BACKLOG}\n{BACKLOG}");
    let max_chars_per_chunk = 180;
    let estimate = estimate_capture(
        &transcript,
        max_chars_per_chunk,
        Duration::from_secs(20),
        CapturePricing::FlatSubscription,
        &provider,
    );
    let chunks = expected_chunk_count(&transcript, max_chars_per_chunk);

    assert_eq!(provider.call_count(), 0);
    assert_eq!(estimate.projected_chunks, chunks);
    assert_eq!(
        estimate.projected_time,
        Duration::from_secs(20) * chunks as u32
    );
}

#[test]
fn estimate_projects_dollars_on_pay_per_token_path() {
    let provider = provider_with_usage();
    let transcript = format!("{BACKLOG}\nextra terminal chunk that changes the token forecast");
    let max_chars_per_chunk = 140;
    let estimate = estimate_capture(
        &transcript,
        max_chars_per_chunk,
        Duration::from_secs(20),
        haiku_pricing(),
        &provider,
    );
    let projected_tokens = expected_projected_tokens(&transcript, max_chars_per_chunk);

    assert_eq!(provider.call_count(), 0);
    assert_eq!(estimate.provider_calls, 0);
    assert_eq!(estimate.projected_tokens, Some(projected_tokens));
    assert_eq!(
        estimate.projected_dollars,
        Some(projected_tokens as f64 * 0.00000025)
    );
}
