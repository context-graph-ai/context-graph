// HITL review surface (9 acceptance tests).
//
// Pure-logic tests (no store): assert function/struct contract.
//   - five_message_kinds: placeholder returns 4 (missing DriftAlert)
//   - capture_receipt_leads_with_intention: placeholder prefixes "Decisions:"
//   - six_verbs_are_the_whole_action_surface: placeholder accepts unknown verbs
//
// Store-state tests: cursor persistence, pending-item persistence, sync-policy.
//   - review_cursor_is_local_only_bare_review_advances_it: placeholder always advances
//   - pending_item_persists_in_every_window_until_acted_on: placeholder returns empty
//   - pending_item_persists_through_an_intervening_confident_capture: placeholder clears on advance
//   - review_cursor_is_not_synced: placeholder policy is Both
//   - cg_review_surfaces_three_single_edge_categories_currency_flips_structurally_empty: placeholder omits category
//   - cg_review_summary_reports_true_per_category_counts: placeholder reports capped length

use std::collections::HashMap;

use context_graph::capture_sync_status::context_graph_sync_table_contract;
use context_graph::ingest::{CaptureRequest, record_conversation_with_extractor};
use context_graph::review_surface::{
    MessageKind, ReviewCategory, ReviewCursor, ReviewItem, ReviewVerb, act_on_item,
    all_message_kinds, build_review_summary, build_review_summary_with_currency_category,
    emit_capture_receipt, is_cursor_local, is_item_pending, list_pending_items,
    maybe_advance_cursor, parse_verb,
};
use context_graph::skill_store::REVIEW_CURSOR_TABLE;
use context_graph::{ContextGraphSyncTablePolicy, Store, StoreConfig, TxId};
use contextdb_core::Value as DbValue;
use contextdb_engine::sync_types::{ConflictPolicy, SyncDirection};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::SpanFaithfulExtractor;

fn fresh_store(name: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join(format!("{name}.db")),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn find_policy(
    contract: &[ContextGraphSyncTablePolicy],
    table: &str,
) -> Option<(SyncDirection, ConflictPolicy)> {
    contract
        .iter()
        .find(|p| p.table == table)
        .map(|p| (p.direction, p.conflict_policy))
}

fn insert_cursor_row(store: &Store, id: &str, position: i64) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_review_cursor (id, position, is_local, updated_at) VALUES ($id, $position, $is_local, $updated_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("position".to_string(), DbValue::Int64(position)),
            ("is_local".to_string(), DbValue::Bool(true)),
            ("updated_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert cursor row");
}

fn make_review_item(id: &str, category: ReviewCategory) -> ReviewItem {
    ReviewItem {
        id: id.to_string(),
        kind: format!("{:?}", category),
        content: format!("item-{id}"),
        category,
    }
}

// Test 1 (pure logic): placeholder all_message_kinds returns 4 items (missing DriftAlert).
// Test fails on the len() == 5 assertion and the contains(DriftAlert) assertion.
#[test]
fn five_message_kinds_emitted_at_one_library_seam_cli_only_renders() {
    let kinds = all_message_kinds();
    assert_eq!(
        kinds.len(),
        5,
        "all_message_kinds must return 5 kinds; wrong stub returns 4"
    );
    assert!(
        kinds.contains(&MessageKind::CaptureReceipt),
        "kinds must include CaptureReceipt"
    );
    assert!(
        kinds.contains(&MessageKind::ReviewItem),
        "kinds must include ReviewItem"
    );
    assert!(
        kinds.contains(&MessageKind::GroupingProposal),
        "kinds must include GroupingProposal"
    );
    assert!(
        kinds.contains(&MessageKind::SkillCard),
        "kinds must include SkillCard"
    );
    assert!(
        kinds.contains(&MessageKind::DriftAlert),
        "kinds must include DriftAlert; wrong stub omits it"
    );
}

// Test 2 (pure logic): placeholder emit_capture_receipt prefixes content with "Decisions:"
// instead of "Intentions:". Test fails on the starts_with("Intentions") assertion.
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
        "capture receipt must lead with 'Intentions'; wrong stub starts with 'Decisions:' (got {:?})",
        msg.content
    );
}

// Test 3 (pure logic): placeholder parse_verb returns Some(Accept) for unknown verbs.
// Test fails on the is_none() assertion for "unknown-action".
#[test]
fn six_verbs_are_the_whole_action_surface() {
    // All six valid verbs must parse.
    assert_eq!(parse_verb("accept"), Some(ReviewVerb::Accept));
    assert_eq!(parse_verb("correct"), Some(ReviewVerb::Correct));
    assert_eq!(parse_verb("drop"), Some(ReviewVerb::Drop));
    assert_eq!(parse_verb("group"), Some(ReviewVerb::Group));
    assert_eq!(parse_verb("undo"), Some(ReviewVerb::Undo));
    assert_eq!(parse_verb("skills-off"), Some(ReviewVerb::SkillsOff));
    assert_eq!(parse_verb("skills-on"), Some(ReviewVerb::SkillsOn));

    // Unknown verbs must be rejected. Placeholder returns Some(Accept).
    assert!(
        parse_verb("unknown-action").is_none(),
        "parse_verb must return None for unknown verbs; wrong stub returns Some(Accept)"
    );
    assert!(
        parse_verb("").is_none(),
        "parse_verb must return None for empty string"
    );
    assert!(
        parse_verb("ACCEPT").is_none(),
        "parse_verb must return None for wrongly-cased verb"
    );
}

// Test 4 (pure logic + struct contract): placeholder
// build_review_summary_with_currency_category delegates to build_review_summary which ignores
// CurrencyFlips items, leaving currency_flips_count=0 even when a CurrencyFlips item was seeded.
// Test fails because currency_flips_count must equal the number of CurrencyFlips items passed in.
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

    // The currency-flips category must count the CurrencyFlips item.
    // Placeholder: delegates to build_review_summary which ignores CurrencyFlips items
    // (no filter for that category), so currency_flips_count stays 0.
    assert_eq!(
        summary.currency_flips_count, 1,
        "currency_flips_count must count CurrencyFlips items; wrong stub returns 0 (drops category)"
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

// Test 5 (pure logic): placeholder build_review_summary uses capped count (.min(cap)) instead
// of true count. Test fails because seeding more than 5 items makes the capped count < true count.
#[test]
fn cg_review_summary_reports_true_per_category_counts() {
    // Seed MORE items per category than the display cap (cap=5 in the placeholder).
    let mut items = Vec::new();
    for i in 0..8usize {
        items.push(make_review_item(
            &format!("unc-{i:03}"),
            ReviewCategory::Uncertain,
        ));
    }
    for i in 0..7usize {
        items.push(make_review_item(
            &format!("grp-{i:03}"),
            ReviewCategory::Grouping,
        ));
    }
    for i in 0..6usize {
        items.push(make_review_item(
            &format!("dft-{i:03}"),
            ReviewCategory::Drift,
        ));
    }

    let summary = build_review_summary(&items);

    // True counts (not capped at 5). Placeholder returns min(8,5)=5 etc.
    assert_eq!(
        summary.uncertain_count, 8,
        "uncertain_count must be the true count (8); wrong stub returns capped count (5)"
    );
    assert_eq!(
        summary.grouping_count, 7,
        "grouping_count must be the true count (7); wrong stub returns capped count (5)"
    );
    assert_eq!(
        summary.drift_count, 6,
        "drift_count must be the true count (6); wrong stub returns capped count (5)"
    );

    // The display list may be capped — assert the true count >= displayed length.
    assert!(
        summary.uncertain_count >= summary.uncertain_items.len(),
        "true uncertain_count must be >= displayed item count"
    );
    assert!(
        summary.grouping_count >= summary.grouping_items.len(),
        "true grouping_count must be >= displayed item count"
    );
    assert!(
        summary.drift_count >= summary.drift_items.len(),
        "true drift_count must be >= displayed item count"
    );

    // A known seeded item must appear under its correct category in the visible window.
    assert!(
        summary
            .uncertain_items
            .iter()
            .any(|i| i.id.starts_with("unc-")),
        "at least one uncertain item must appear under uncertain"
    );
}

// Test 6 (store-state): placeholder maybe_advance_cursor always advances (even for filtered
// lens calls); is_cursor_local returns false instead of true. Test fails on the filtered-lens
// no-advance assertion and the is_cursor_local assertion.
#[test]
fn review_cursor_is_local_only_bare_review_advances_it() {
    let (_dir, store) = fresh_store("k_cursor_local");
    insert_cursor_row(&store, "cursor-local-001", 0);

    let mut cursor = ReviewCursor {
        position: 0,
        is_local: false, // placeholder default value
    };

    // is_cursor_local must return true. Placeholder returns false.
    assert!(
        is_cursor_local(&cursor),
        "is_cursor_local must return true; wrong stub returns false"
    );

    // A filtered lens call must NOT advance the cursor. Placeholder always advances.
    let position_before = cursor.position;
    maybe_advance_cursor(&store, &mut cursor, false);
    assert_eq!(
        cursor.position, position_before,
        "a filtered lens call (is_bare_review=false) must not advance the cursor; wrong stub always advances"
    );

    // A bare review call MUST advance the cursor.
    maybe_advance_cursor(&store, &mut cursor, true);
    assert!(
        cursor.position > position_before,
        "a bare review call (is_bare_review=true) must advance the cursor"
    );
}

// Test 7 (store-state): placeholder list_pending_items always returns empty;
// is_item_pending always returns false; act_on_item is a no-op. Test fails because
// the state before and after act_on_item must differ, proving the pending mechanism is real.
#[test]
fn pending_item_persists_in_every_window_until_acted_on() {
    let (_dir, store) = fresh_store("k_pending_persists");
    let item_id = "pending-item-007";
    insert_cursor_row(&store, "cursor-007", 0);

    let mut cursor = ReviewCursor {
        position: 0,
        is_local: true,
    };

    // Advance cursor twice without acting on item_id.
    maybe_advance_cursor(&store, &mut cursor, true);
    maybe_advance_cursor(&store, &mut cursor, true);

    // The item must still appear as pending before acting on it.
    let pending_before = is_item_pending(&store, item_id, &cursor);
    let list_before = list_pending_items(&store, &cursor);

    // Act on the item to resolve it.
    act_on_item(&store, item_id, &ReviewVerb::Accept);

    let pending_after = is_item_pending(&store, item_id, &cursor);
    let list_after = list_pending_items(&store, &cursor);

    // Load-bearing: before acting, the item must be pending (placeholder: always false/empty).
    // After acting, it must be gone (placeholder: also false/empty — indistinguishable).
    // We assert the pre-act state is pending AND the post-act state is resolved.
    assert!(
        pending_before,
        "is_item_pending must return true before act_on_item; wrong stub always returns false"
    );
    assert!(
        list_before.iter().any(|i| i.id == item_id),
        "list_pending_items must include item before acting; wrong stub returns empty"
    );
    assert!(
        !pending_after,
        "is_item_pending must return false after act_on_item"
    );
    assert!(
        !list_after.iter().any(|i| i.id == item_id),
        "list_pending_items must not include item after acting"
    );
}

// Test 8 (store-state): placeholder act_on_item is a no-op AND list_pending_items always
// returns empty — so the state before and after an intervening capture is indistinguishable.
// Test fails because a pending item must survive an intervening confident capture.
#[test]
fn pending_item_persists_through_an_intervening_confident_capture() {
    let (_dir, store) = fresh_store("k_pending_through_capture");
    let pending_item_id = "pending-item-008";
    insert_cursor_row(&store, "cursor-008", 0);

    let mut cursor = ReviewCursor {
        position: 0,
        is_local: true,
    };

    // Run an intervening fully-confident (silent) capture and advance the cursor.
    let extractor = SpanFaithfulExtractor;
    let transcript = "Goal G1: test capture.\nDecision D1: use Rust. serves G1";
    let _ = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "k-pending-ctx".to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &extractor,
    )
    .expect("confident capture must succeed");
    maybe_advance_cursor(&store, &mut cursor, true);

    // The pending item must still be visible after the intervening capture.
    let still_pending = is_item_pending(&store, pending_item_id, &cursor);
    let list_still = list_pending_items(&store, &cursor);

    // Act on it to resolve.
    act_on_item(&store, pending_item_id, &ReviewVerb::Accept);
    let after_act = is_item_pending(&store, pending_item_id, &cursor);

    // Load-bearing: item is pending before act, gone after.
    // Placeholder returns false/empty in both states → the before assertion fails.
    assert!(
        still_pending,
        "pending item must persist through an intervening confident capture; wrong stub returns false prematurely"
    );
    assert!(
        list_still.iter().any(|i| i.id == pending_item_id),
        "list_pending_items must include item after confident capture; wrong stub returns empty"
    );
    assert!(
        !after_act,
        "after act_on_item, pending item must be resolved"
    );
}

// Test 9 (store-state / sync policy): placeholder cg_review_cursor policy is
// (Both, InsertIfNotExists) — it would sync. Test fails on the SyncDirection::None assertion.
#[test]
fn review_cursor_is_not_synced() {
    let contract = context_graph_sync_table_contract();
    let policy = find_policy(&contract, REVIEW_CURSOR_TABLE);

    assert!(
        policy.is_some(),
        "cg_review_cursor must be registered in context_graph_sync_table_contract"
    );

    let (direction, _conflict_policy) = policy.unwrap();
    assert_eq!(
        direction,
        SyncDirection::None,
        "cg_review_cursor must have SyncDirection::None (local only); wrong stub returns Both"
    );
}
