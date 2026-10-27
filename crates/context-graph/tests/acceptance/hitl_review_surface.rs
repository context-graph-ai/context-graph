// Library review contracts cover receipt ordering, persistent pending items,
// local review policy, category visibility and uncapped per-category counts.

use context_graph::ingest::{CaptureRequest, record_conversation_with_extractor};
use context_graph::review_surface::{
    MessageKind, ReviewCategory, ReviewCursor, ReviewItem, ReviewVerb, act_on_item,
    build_review_summary_with_currency_category, emit_capture_receipt, gather_review_summary,
};
use context_graph::skill_store::REVIEW_CURSOR_TABLE;
use context_graph::{Store, StoreConfig};

use super::action_test_extractors::SpanFaithfulExtractor;
use super::fake_extractor::open_store_with_lexical_embedder;

fn fresh_store(name: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join(format!("{name}.db")),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn make_review_item(id: &str, category: ReviewCategory) -> ReviewItem {
    ReviewItem {
        id: id.to_string(),
        kind: format!("{:?}", category),
        content: format!("item-{id}"),
        category,
    }
}

// CLI message kinds and action parsing are exercised through the binary in
// crates/cg-cli/tests/understanding_actions_cli.rs.

#[test]
fn capture_receipt_leads_with_intention() {
    let msg = emit_capture_receipt("test-report");
    assert_eq!(
        msg.kind,
        MessageKind::CaptureReceipt,
        "emitted message must have CaptureReceipt kind"
    );
    assert!(
        msg.content.starts_with("Intentions"),
        "capture receipt must lead with 'Intentions' (got {:?})",
        msg.content
    );
}

// The library summary counts CurrencyFlips items alongside the single-edge categories.
#[test]
fn cg_review_surfaces_three_single_edge_categories_currency_flips_structurally_empty() {
    // Seed one item per single-edge category, plus one CurrencyFlips placeholder.
    // On a single edge the CurrencyFlips category comes from server-core (none here);
    // we represent its structurally-empty presence via a StructurallyEmpty-typed item.
    let items = vec![
        make_review_item("unc-001", ReviewCategory::Uncertain),
        make_review_item("grp-001", ReviewCategory::Grouping),
        make_review_item("dft-001", ReviewCategory::Drift),
        make_review_item("cf-001", ReviewCategory::CurrencyFlips),
    ];

    let summary = build_review_summary_with_currency_category(&items);

    // Three single-edge categories must be populated.
    assert_eq!(summary.uncertain_count, 1, "uncertain_count must be 1");
    assert_eq!(summary.grouping_count, 1, "grouping_count must be 1");
    assert_eq!(summary.drift_count, 1, "drift_count must be 1");

    assert_eq!(
        summary.currency_flips_count, 1,
        "currency_flips_count must count CurrencyFlips items rather than a constant zero"
    );

    // Category membership: items must land under their correct categories.
    assert!(
        summary.uncertain_items.iter().any(|i| i.id == "unc-001"),
        "uncertain item must appear under uncertain"
    );
    assert!(
        summary.grouping_items.iter().any(|i| i.id == "grp-001"),
        "grouping item must appear under grouping"
    );
    assert!(
        summary.drift_items.iter().any(|i| i.id == "dft-001"),
        "drift item must appear under drift"
    );
}

// LIVE PATH: `cg review` summarizes STORE state (library `gather_review_summary` over store-gathered
// items), not a caller-supplied list. A REAL capture-produced flagged item lands in the UNCERTAIN
// category, and the per-category counts are the TRUE store totals (the display list may be capped).
// an incorrect implementation's gather returns an empty summary → the capture flag is absent and counts are 0 → regression.
#[test]
fn cg_review_summary_reports_true_per_category_counts() {
    let (_dir, store) = fresh_store("k_review_summary_store");

    // Run a REAL capture that produces a flagged (uncertain) item — a fabricated span absent from
    // the transcript, so grounding flags it.
    let report = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "k-review-summary-ctx".to_string(),
            transcript: "the team discussed only routine maintenance".to_string(),
            model_available: true,
        },
        &super::action_test_extractors::HighConfidenceFabricatedSpanExtractor,
    )
    .expect("capture producing a flagged item");
    assert!(
        !report.flagged_items.is_empty(),
        "the capture must produce a flagged item to gather into review"
    );
    let flagged_key = report.flagged_items[0].key.clone();

    // Gather the review summary FROM THE STORE (not a caller-supplied list).
    let cursor = ReviewCursor {
        position: 0,
        is_local: true,
    };
    let summary = gather_review_summary(&store, &cursor).expect("read succeeds");

    // The capture-produced flagged item must appear under UNCERTAIN, and the uncertain count must
    // reflect the store (≥ 1). an incorrect implementation's gather returns empty → both fail.
    assert!(
        summary.uncertain_count >= 1,
        "uncertain_count must reflect the store's capture-produced flagged item (≥1); an incorrect implementation's \
         store-gather returns an empty summary (0)"
    );
    assert!(
        summary
            .uncertain_items
            .iter()
            .any(|i| i.id == flagged_key || i.content.contains(&flagged_key)),
        "the capture-produced flagged item ('{flagged_key}') must appear under the UNCERTAIN \
         category in the store-gathered review summary; observed uncertain_items: {:?}",
        summary.uncertain_items
    );
    // The display list may be capped, but the count is the true magnitude (never the capped length).
    assert!(
        summary.uncertain_count >= summary.uncertain_items.len(),
        "the reported count must be the true magnitude (≥ displayed length), never the capped length"
    );
}

// A capture-produced flagged item persists in store-backed review windows until acted on.
#[test]
fn pending_item_persists_in_every_window_until_acted_on() {
    let (_dir, store) = fresh_store("k_pending_persists");

    // A REAL capture produces a flagged item (fabricated span).
    let report = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "k-pending-ctx".to_string(),
            transcript: "the team discussed only routine maintenance".to_string(),
            model_available: true,
        },
        &super::action_test_extractors::HighConfidenceFabricatedSpanExtractor,
    )
    .expect("capture producing a flagged item");
    assert!(
        !report.flagged_items.is_empty(),
        "capture must flag an item"
    );
    let item_id = report.flagged_items[0].key.clone();

    let cursor = ReviewCursor {
        position: 0,
        is_local: true,
    };

    // Both review windows gather the pending item from stored state without acting on it.
    for window in 0..2 {
        let summary = gather_review_summary(&store, &cursor).expect("read succeeds");
        assert!(
            summary
                .uncertain_items
                .iter()
                .any(|i| i.id == item_id || i.content.contains(&item_id)),
            "window {window}: the capture-produced flagged item ('{item_id}') must persist in the \
             store-gathered review summary until acted on; observed uncertain_items: {:?}",
            summary.uncertain_items
        );
    }

    // Act on it via the drop action seam; it must no longer appear.
    act_on_item(&store, &item_id, &ReviewVerb::Drop);
    let after = gather_review_summary(&store, &cursor).expect("read succeeds");
    assert!(
        !after
            .uncertain_items
            .iter()
            .any(|i| i.id == item_id || i.content.contains(&item_id)),
        "after acting (cg drop), the item must no longer appear in the review summary"
    );
}

// LIVE PATH: a pending flagged item from a REAL capture survives an intervening fully-confident
// (silent) capture that advances the cursor — `cg review` (store-gathered) still surfaces it until
// acted on. an incorrect implementation's store-gather returns empty → fails .
#[test]
fn pending_item_persists_through_an_intervening_confident_capture() {
    let (_dir, store) = fresh_store("k_pending_through_capture");

    // A REAL capture produces a flagged item X.
    let report = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "k-pending-flag-ctx".to_string(),
            transcript: "the team discussed only routine maintenance".to_string(),
            model_available: true,
        },
        &super::action_test_extractors::HighConfidenceFabricatedSpanExtractor,
    )
    .expect("capture producing flagged item X");
    assert!(!report.flagged_items.is_empty(), "capture must flag item X");
    let item_id = report.flagged_items[0].key.clone();

    let cursor = ReviewCursor {
        position: 0,
        is_local: true,
    };

    // Run an intervening fully-confident (silent) capture.
    let _ = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "k-pending-confident-ctx".to_string(),
            transcript: "Goal G1: test capture.\nDecision D1: use Rust serves G1".to_string(),
            model_available: true,
        },
        &SpanFaithfulExtractor,
    )
    .expect("confident capture must succeed");

    // The pending item remains in the store-backed review after a confident capture.
    let summary = gather_review_summary(&store, &cursor).expect("read succeeds");
    assert!(
        summary
            .uncertain_items
            .iter()
            .any(|i| i.id == item_id || i.content.contains(&item_id)),
        "the pending flagged item ('{item_id}') must survive an intervening confident capture in the \
         store-gathered review summary; observed uncertain_items: {:?}",
        summary.uncertain_items
    );

    // Act on it via the drop action seam; it must no longer appear.
    act_on_item(&store, &item_id, &ReviewVerb::Drop);
    let after = gather_review_summary(&store, &cursor).expect("read succeeds");
    assert!(
        !after
            .uncertain_items
            .iter()
            .any(|i| i.id == item_id || i.content.contains(&item_id)),
        "after acting (cg drop), the item must no longer appear in the review summary"
    );
}

// Test 9: the review cursor is local operational state.
#[test]
fn review_cursor_is_not_synced() {
    let (_dir, store) = fresh_store("review_cursor_is_not_synced");
    let meta = store
        .sync_database()
        .table_meta(REVIEW_CURSOR_TABLE)
        .expect("cg_review_cursor must be installed");
    assert_eq!(
        meta.sync_direction.map(|direction| direction.sql()),
        Some("SYNC OFF"),
        "cg_review_cursor must declare local-only behavior"
    );
    assert_eq!(
        meta.conflict_policy
            .and_then(|policy| policy.declared_clause()),
        None,
        "cg_review_cursor is SYNC OFF and must not declare a conflict policy"
    );
}
