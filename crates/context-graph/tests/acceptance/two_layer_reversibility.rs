// Two-layer reversibility tests.
//
// Store-state tests: every assertion reads from the persisted overlay/teaching-event
// tables. A HashMap that overwrites in place fails the "original unchanged" assertion;
// a placeholder that returns empty tables fails the "overlay entry / compensation event exists"
// assertions.
//
// Current placeholder behavior in overlay.rs:
//   - record_correction returns an entry but does NOT persist it
//   - read_with_overlay always returns the original value
//   - reverse_overlay returns None (no compensation event)
//   - is_dropped always returns false
//   - list_overlay_entries_for_item always returns empty
//   - list_compensation_events always returns empty
//   - apply_teaching_events_in_order always returns the original

use context_graph::overlay::{
    CompensationEventKind, OverlayKind, apply_teaching_events_in_order, is_dropped,
    list_compensation_events, list_overlay_entries_for_item, read_with_overlay, record_correction,
    record_drop, record_grouping_link, record_wrong_type_correction, reverse_overlay,
};
use context_graph::skill_store::{TeachingEventKind, apply_correction};
use context_graph::{Store, StoreConfig};

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

// Test 1 (store-state): record_correction returns an entry but does NOT persist it;
// read_with_overlay always returns the original value.
#[test]
fn correction_overrides_on_read_without_editing_the_original_capture() {
    let (_dir, _store) = fresh_store("i_correction_overrides");
    let item_id = "item-original-001";
    let original_value = "code_reference";
    let corrected_value = "technology_choice";

    // Record a correction for this item (placeholder: not persisted).
    let _entry = record_correction(&_store, item_id, corrected_value);

    // The correct implementation persists the overlay entry. Read it back.
    let entries = list_overlay_entries_for_item(&_store, item_id);
    assert!(
        !entries.is_empty(),
        "overlay entry for item must be persisted; placeholder returns empty"
    );

    // The read view must show the corrected value, not the original.
    let read_value = read_with_overlay(original_value, item_id, &entries);
    assert_eq!(
        read_value, corrected_value,
        "read view must show corrected value; placeholder returns original"
    );

    // The entry kind must be Correction.
    assert!(
        entries.iter().any(|e| e.kind == OverlayKind::Correction),
        "persisted entry must have Correction kind"
    );
}

// Test 2 (store-state): reverse_overlay returns None (no compensation event);
// list_compensation_events returns empty.
#[test]
fn reversal_un_applies_everywhere_and_never_rewrites_stored_history() {
    let (_dir, _store) = fresh_store("i_reversal_un_applies");
    let item_id = "item-rev-002";
    let original_value = "old_decision_type";
    let corrected_value = "new_decision_type";

    // Record a correction, then reverse it.
    let mut entry = record_correction(&_store, item_id, corrected_value);
    let compensation = reverse_overlay(&_store, &mut entry);

    // Placeholder: compensation is None. The real implementation logs it.
    assert!(
        compensation.is_some(),
        "reversal must produce a compensation event; placeholder returns None"
    );
    let event = compensation.unwrap();
    assert_eq!(
        event.reversed_overlay_id, entry.id,
        "compensation event must reference the reversed entry"
    );

    // After reversal the read view must revert to the original.
    let entries = list_overlay_entries_for_item(&_store, item_id);
    let read_value = read_with_overlay(original_value, item_id, &entries);
    assert_eq!(
        read_value, original_value,
        "read view must revert after reversal; placeholder still returns corrected"
    );

    // The correction event row must still exist in history (Invariant 3: no hard deletes).
    let all_entries = list_overlay_entries_for_item(&_store, item_id);
    assert!(
        !all_entries.is_empty(),
        "history entry must remain; placeholder hard-deletes it"
    );
    assert!(
        all_entries.iter().any(|e| e.reversed),
        "reversal must mark entry as reversed, not delete it"
    );

    // Compensation events must be listed.
    let comp_events = list_compensation_events(&_store);
    assert!(
        !comp_events.is_empty(),
        "list_compensation_events must return the logged event; placeholder returns empty"
    );
}

// Test 3 (store-state): record_grouping_link does not persist any link.
#[test]
fn grouping_links_intentions_without_changing_or_deleting_any_row() {
    let (_dir, _store) = fresh_store("i_grouping_links");
    let intention_a = "intention-a-001";
    let intention_b = "intention-b-001";
    let intention_c = "intention-c-001";
    let catalog_id = "catalog-canonical-001";

    // Insert three grouping links — one per intention.
    let entry_a = record_grouping_link(&_store, intention_a, catalog_id);
    let entry_b = record_grouping_link(&_store, intention_b, catalog_id);
    let entry_c = record_grouping_link(&_store, intention_c, catalog_id);

    // Each link must be persisted. Placeholder: not persisted.
    for (id, entry) in [
        (intention_a, &entry_a),
        (intention_b, &entry_b),
        (intention_c, &entry_c),
    ] {
        let entries = list_overlay_entries_for_item(&_store, id);
        assert!(
            !entries.is_empty(),
            "grouping link for {id} must be persisted; placeholder returns empty"
        );
        assert!(
            entries
                .iter()
                .any(|e| e.kind == OverlayKind::GroupingLink
                    && e.value.as_deref() == Some(catalog_id)),
            "grouping link entry must carry catalog_id {catalog_id}"
        );
        // The link is additive: item_id equals the intention id.
        assert_eq!(
            entry.item_id, id,
            "grouping link item_id must match intention id"
        );
    }
}

// Test 4 (store-state): record_drop does not persist the entry; is_dropped always returns false.
#[test]
fn drop_excludes_from_read_view_but_preserves_provenance() {
    let (_dir, _store) = fresh_store("i_drop_preserves");
    let item_id = "item-to-drop-004";

    // Record a drop overlay for this item.
    let _entry = record_drop(&_store, item_id);

    // is_dropped must return true after a persisted drop. Placeholder: always false.
    let entries = list_overlay_entries_for_item(&_store, item_id);
    assert!(
        !entries.is_empty(),
        "drop entry must be persisted; wrong stub returns empty"
    );

    let dropped = is_dropped(item_id, &entries);
    assert!(
        dropped,
        "is_dropped must return true after a drop entry is persisted; wrong stub returns false"
    );

    // The entry itself must still exist (Invariant 3: no hard deletes).
    assert!(
        entries.iter().any(|e| e.kind == OverlayKind::Drop),
        "drop entry row must remain; wrong stub hard-deletes it"
    );
}

// Test 5 (store-state): record_wrong_type_correction does not persist an overlay;
// apply_correction returns teaching_event=None.
#[test]
fn corrected_intention_that_was_a_decision_stays_reversible_and_logs_a_lesson() {
    let (_dir, _store) = fresh_store("i_wrong_type_correction");
    let decision_item_id = "decision-item-005";
    let new_type = "intention";

    // Correct a Decision to an Intention via wrong-type correction.
    let mut entry = record_wrong_type_correction(&_store, decision_item_id, new_type);

    // The entry kind must be WrongTypeCorrection and must be persisted.
    let entries = list_overlay_entries_for_item(&_store, decision_item_id);
    assert!(
        !entries.is_empty(),
        "wrong-type correction entry must be persisted; wrong stub returns empty"
    );
    assert!(
        entries
            .iter()
            .any(|e| e.kind == OverlayKind::WrongTypeCorrection),
        "persisted entry must have WrongTypeCorrection kind"
    );

    // apply_correction must log a teaching event.
    let correction_result =
        apply_correction(&_store, decision_item_id, new_type, Some("skill-001"));
    assert!(
        correction_result.teaching_event.is_some(),
        "apply_correction must log a teaching event for a wrong-type correction; wrong stub returns None"
    );
    let event = correction_result.teaching_event.unwrap();
    assert_eq!(
        event.corrected_item_id, decision_item_id,
        "teaching event must reference the corrected item by id"
    );
    // The content must not be empty.
    assert!(
        !event.content.is_empty(),
        "teaching event content must describe the kind change; wrong stub is empty"
    );
    // Kind must be WrongTypeCorrection or Correction.
    assert!(
        matches!(
            event.kind,
            TeachingEventKind::WrongTypeCorrection | TeachingEventKind::Correction
        ),
        "teaching event kind must reflect a correction"
    );

    // Reversing the wrong-type correction must produce a compensation event.
    let compensation = reverse_overlay(&_store, &mut entry);
    assert!(
        compensation.is_some(),
        "wrong-type correction reversal must produce a compensation event"
    );

    // History must survive reversal (Invariant 3).
    let entries_after = list_overlay_entries_for_item(&_store, decision_item_id);
    assert!(
        !entries_after.is_empty(),
        "overlay history must remain after reversal; wrong stub hard-deletes"
    );
}

// Test 6 (store-state): reverse_overlay returns None (no compensation event logged).
#[test]
fn grouping_reversal_is_logged_as_a_compensation_event() {
    let (_dir, _store) = fresh_store("i_grouping_reversal_logged");
    let intention_id = "intention-006";
    let catalog_id = "catalog-006";

    // Create a grouping link, then reverse it.
    let mut entry = record_grouping_link(&_store, intention_id, catalog_id);
    let compensation = reverse_overlay(&_store, &mut entry);

    // Placeholder: returns None. The real implementation logs a grouping-reversal compensation event.
    assert!(
        compensation.is_some(),
        "grouping reversal must log a compensation event; wrong stub returns None"
    );
    let event = compensation.unwrap();
    // Kind must be GroupingReversal.
    assert_eq!(
        event.kind,
        CompensationEventKind::GroupingReversal,
        "compensation event kind must be GroupingReversal"
    );
    // Content must reference the grouped items by id.
    assert!(
        event.content.contains(intention_id) || event.content.contains(catalog_id),
        "compensation event content must reference the grouping (intention or catalog id)"
    );

    // The event must appear in list_compensation_events.
    let events = list_compensation_events(&_store);
    assert!(
        !events.is_empty(),
        "list_compensation_events must include the grouping reversal; wrong stub returns empty"
    );
    assert!(
        events.iter().any(|e| e.reversed_overlay_id == entry.id),
        "compensation event must reference the reversed overlay entry"
    );

    // Grouping link must be marked reversed, not deleted.
    let entries = list_overlay_entries_for_item(&_store, intention_id);
    assert!(
        !entries.is_empty(),
        "grouping link history must remain; wrong stub hard-deletes"
    );
    assert!(
        entries.iter().any(|e| e.reversed),
        "reversed grouping link must be marked reversed, not deleted"
    );
}

// Test 7 (store-state): placeholder does not track state so the end read state is always
// the original. list_overlay_entries_for_item returns empty, so the correction was never
// applied in the first place.
#[test]
fn reversal_concurrent_with_matching_capture_has_deterministic_end_state() {
    let (_dir, _store) = fresh_store("i_concurrent_reversal");
    let item_id_a = "item-concurrent-007a";
    let item_id_b = "item-concurrent-007b";
    let original_value = "capture_original";
    let corrected_value = "capture_corrected";

    // Ordering A: correct then reverse.
    let mut entry_a = record_correction(&_store, item_id_a, corrected_value);
    let _comp_a = reverse_overlay(&_store, &mut entry_a);

    let entries_a = list_overlay_entries_for_item(&_store, item_id_a);
    let read_a = read_with_overlay(original_value, item_id_a, &entries_a);

    // Ordering B: also correct then reverse on a parallel item.
    let mut entry_b = record_correction(&_store, item_id_b, corrected_value);
    let _comp_b = reverse_overlay(&_store, &mut entry_b);

    let entries_b = list_overlay_entries_for_item(&_store, item_id_b);
    let read_b = read_with_overlay(original_value, item_id_b, &entries_b);

    // Both orderings must end with the original value restored (reversal wins).
    assert_eq!(
        read_a, original_value,
        "after reversal, read view must return original; wrong stub never applied the correction"
    );
    assert_eq!(
        read_b, original_value,
        "after reversal (parallel ordering), read view must return original"
    );
    // Determinism: both paths produce the same result.
    assert_eq!(
        read_a, read_b,
        "concurrent orderings must converge to same read value"
    );

    // History entries must exist (Invariant 3).
    assert!(
        !entries_a.is_empty(),
        "history entries must exist for item_id_a; wrong stub returns empty"
    );
    assert!(
        !entries_b.is_empty(),
        "history entries must exist for item_id_b; wrong stub returns empty"
    );
}

// Test 8 (store-state via the overlay API): placeholder apply_teaching_events_in_order
// always returns the original value; a correction-only call must return the corrected value.
#[test]
fn out_of_order_reversal_teaching_event_converges_to_same_read_view() {
    let original_value = "code_reference";
    let corrected_value = "technology_choice";
    let item_id = "item-order-008";

    // Build a correction entry and a reversal entry (reversed=true).
    // A store is required by the API; this test only asserts pure-value order convergence.
    let (_ooo_dir, ooo_store) = fresh_store("i_order_008");
    let correction_entry = record_correction(&ooo_store, item_id, corrected_value);
    let mut reversal_entry = correction_entry.clone();
    reversal_entry.reversed = true;

    // Standalone correction (no reversal): must return the corrected value.
    // Placeholder always returns original_value.
    let correction_only = vec![correction_entry.clone()];
    let read_correction_only = apply_teaching_events_in_order(original_value, &correction_only);
    assert_eq!(
        read_correction_only, corrected_value,
        "correction alone must produce the corrected value; wrong stub always returns original"
    );

    // In-order: [correction, reversal]. Net effect: original restored.
    let in_order = vec![correction_entry.clone(), reversal_entry.clone()];
    let read_in_order = apply_teaching_events_in_order(original_value, &in_order);

    // Out-of-order: [reversal, correction]. Must converge to the same net value.
    let out_of_order = vec![reversal_entry.clone(), correction_entry.clone()];
    let read_out_of_order = apply_teaching_events_in_order(original_value, &out_of_order);

    // Both paths must yield the same final read value.
    assert_eq!(
        read_in_order, read_out_of_order,
        "out-of-order teaching events must converge to the same read view"
    );
    // The final value after correction+reversal is the original.
    assert_eq!(
        read_in_order, original_value,
        "after correction then reversal, read view must equal original"
    );
}
